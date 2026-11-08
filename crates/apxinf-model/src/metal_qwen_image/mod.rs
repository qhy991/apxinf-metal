//! Native Qwen-Image-2.1 text-to-image on Apple Metal, following the mflux
//! `QwenImage21` txt2img path (MIT) with its `--low-ram` lifecycle: the
//! text encoder is read one layer at a time, and the DiT is released before
//! VAE decoding and read again by the next request. No Python execution.
//!
//! The checkpoint is the mflux 4-bit export (affine Q4 DiT, BF16 text
//! encoder, F32 VAE). Its weights are under the Qwen Research License; this
//! family ships no weights.
#[cfg(feature = "coreml")]
mod ane;
#[cfg(feature = "coreml")]
mod ane_sim;
mod dit;
mod encode_cache;
mod ops;
mod tables;
mod text;
mod vae;
mod weights;

use crate::image::{GeneratedImage, ImageRequest, ImageRuntime};
use crate::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use apxinf_mlx::fusions::MppBf16GemmNt;
use apxinf_mlx::qmm_bm64::QmmBm64;
use apxinf_mlx::{Array, MlxBackend, MlxDType, Stream};
use apxinf_tokenizer::Tokenizer;
use ops::Ops;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use weights::{Decoder, TextEncoderFiles, Transformer, DIM, LATENT, MLP};

/// The only schedule this family has validated: 40 linear steps, guidance 1.
pub const STEPS: usize = 40;
/// Owner-selected runtime tier for the TeaCache-style step cache; `0` on the
/// request disables it (rollback to the unchanged denoise loop).
pub const DEFAULT_STEP_CACHE_RATIO: f32 = 0.25;
/// Pixels per latent; the RoPE tables cover 2048 latent rows or columns.
const SCALE: usize = 16;
const MAX_SIDE: usize = 2048 * SCALE;
const MAX_IMAGE_TOKENS: usize = (i32::MAX as usize + 1) / MLP;

pub struct QwenImage {
    dir: PathBuf,
    tokenizer: Tokenizer,
    text: TextEncoderFiles,
    /// Loaded until the first VAE decode, then read again per request.
    transformer: Option<Transformer>,
    decoder: Decoder,
    ops: Ops,
    /// Present in auto mode when the synthetic MPP probe matches quantized matmul.
    gemm: Option<Rc<MppBf16GemmNt>>,
    /// Present in auto mode on an architecture where the fused image blocks
    /// were qualified; image-token counts in `dit::FUSED_TOKENS` use them.
    fused: Option<Rc<dit::Fused>>,
    /// The Neural Engine MLP share (on unless `APXINF_QWEN_IMAGE_ANE=0`),
    /// loaded once.
    #[cfg(feature = "coreml")]
    ane: Option<Rc<dit::AneShare>>,
    /// Present in auto mode when the BM64 retiled qmm's install probe matched
    /// stock quantized matmul; the non-fused image fallback uses it.
    qmm_bm64: Option<Rc<QmmBm64>>,
    /// `Qwen21Rope` cos/sin tables for the 16- and 56-wide axes.
    rope16: (Vec<f32>, Vec<f32>),
    rope56: (Vec<f32>, Vec<f32>),
}

/// Device-independent option admission, checked before the checkpoint is read.
fn admit(options: &LoadOptions) -> Result<()> {
    if !matches!(
        options.model_variant.as_deref(),
        None | Some("auto" | "reference")
    ) || !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) || options.text_weight_dtype.is_some()
        || options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.tuning_path.is_some()
        || options.autotune
        || options.uniform_fp8_scale.is_some()
        || !options.assets.is_empty()
    {
        return Err(Error::Contract(
            "Qwen-Image MLX loads the mflux 4-bit checkpoint as stored; only auto and reference \
             variants are supported, without CUDA tuning, calibration, synthetic weights or auxiliary assets",
        ));
    }
    Ok(())
}

pub fn load(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Metal(_)) {
        return Err(Error::UnsupportedDevice(device));
    }
    admit(options)?;
    let native = backend
        .as_any()
        .downcast_ref::<MlxBackend>()
        .ok_or(Error::Contract(
            "Qwen-Image requires the native MLX backend",
        ))?
        .clone();
    if native.device() != device {
        return Err(Error::DeviceMismatch {
            expected: device,
            got: native.device(),
        });
    }
    if !path.is_dir() {
        return Err(Error::Contract("Qwen-Image loads a checkpoint directory"));
    }
    let stream = native.stream().clone();
    let tokenizer = Tokenizer::from_file(path.join("processor/tokenizer.json"))?;
    let text = TextEncoderFiles::open(&path.join("text_encoder"))?;
    let transformer = Transformer::load(&stream, &path.join("transformer"))?;
    let decoder = Decoder::load(&stream, &path.join("vae"))?;
    let reference = options.model_variant.as_deref() == Some("reference");
    let fused = if reference {
        None
    } else {
        fused_blocks(&stream)?.map(Rc::new)
    };
    #[cfg(feature = "coreml")]
    let ane = ane_share(
        &stream,
        &transformer,
        &path.join("transformer"),
        fused.is_some(),
    )?;
    #[cfg(not(feature = "coreml"))]
    if std::env::var("APXINF_QWEN_IMAGE_ANE").is_ok_and(|v| v == "1") {
        eprintln!(
            "Qwen-Image: APXINF_QWEN_IMAGE_ANE=1 ignored: built without the coreml feature; GPU-only"
        );
    }
    Ok(LoadedModel::Image(Box::new(QwenImage {
        dir: path.to_path_buf(),
        tokenizer,
        text,
        transformer: Some(transformer),
        decoder,
        ops: Ops::new(&stream)?,
        gemm: if reference {
            None
        } else {
            probed_gemm(&stream).map(Rc::new)
        },
        fused,
        #[cfg(feature = "coreml")]
        ane,
        qmm_bm64: if reference {
            None
        } else {
            qmm_bm64_install(&stream)
        },
        rope16: tables::rope_axis(16, 10000.0),
        rope56: tables::rope_axis(56, 10000.0),
    })))
}

/// MLX architectures where the fused image blocks were qualified against
/// their reference implementation (pixels, memory, paired whole-image
/// timing): the M4. They do not reproduce mflux bit for bit (see the README).
/// The Neural Engine MLP share unless `APXINF_QWEN_IMAGE_ANE=0`, when the
/// fused blocks are in use and the memory gate admits it; otherwise GPU-only,
/// with the reason logged.
#[cfg(feature = "coreml")]
fn ane_share(
    stream: &Stream,
    transformer: &Transformer,
    dir: &Path,
    fused: bool,
) -> Result<Option<Rc<dit::AneShare>>> {
    if let Some(why) = ane::enable_refusal() {
        eprintln!("Qwen-Image: ANE MLP split disabled: {why}; GPU-only");
    }
    if !ane::requested() {
        return Ok(None);
    }
    if !fused {
        eprintln!(
            "Qwen-Image: ANE MLP split needs the fused image blocks (not selected here); GPU-only"
        );
        return Ok(None);
    }
    if let Some(why) = ane::split_refusal() {
        eprintln!("Qwen-Image: ANE MLP split disabled: {why}; GPU-only");
        return Ok(None);
    }
    if let Some(why) = ane_sim::refusal() {
        eprintln!("Qwen-Image: ANE MLP split disabled: {why}; GPU-only");
        return Ok(None);
    }
    match ane::memory_gate() {
        Ok(facts) => eprintln!("Qwen-Image: ANE MLP split admitted by the memory gate: {facts}"),
        Err(facts) => {
            eprintln!("Qwen-Image: ANE MLP split disabled by the memory gate: {facts}; GPU-only");
            return Ok(None);
        }
    }
    let offload = match ane_sim::requested() {
        Some(precision) => {
            eprintln!(
                "Qwen-Image: ANE MLP split SIMULATED on the GPU in {precision:?} (#236 H2): \
                 Core ML is not used, and this arm's timing is not comparable"
            );
            dit::Offload::Sim(ane_sim::Sim::share(transformer, precision)?)
        }
        None => dit::Offload::Ane(ane::Programs::install(transformer, dir, timing_enabled())?),
    };
    Ok(Some(Rc::new(dit::AneShare {
        offload,
        scales: ane::Scales::new(stream)?,
        spent: dit::AneSpent::default(),
    })))
}

const FUSED_QUALIFIED_ARCHITECTURES: &[&str] = &["applegpu_g16g"];

fn fused_blocks(stream: &Stream) -> Result<Option<dit::Fused>> {
    let architecture = stream
        .metal_caps()
        .map_or("", |caps| caps.architecture.as_str());
    if !FUSED_QUALIFIED_ARCHITECTURES.contains(&architecture) {
        return Ok(None);
    }
    dit::Fused::new(stream).map(Some)
}

/// MLX architectures where MPP image projections were measured faster than
/// quantized matmul: the M4 (paired whole-image timing). Other architectures keep quantized matmul until
/// measured there; the selection is structural, so it stays in this family.
const MPP_MEASURED_ARCHITECTURES: &[&str] = &["applegpu_g16g"];

/// Enables MPP on a measured architecture after a bit-identical probe of the
/// three block projection shapes `(K, N)` at `dit::MPP_MIN_TOKENS` rows, on
/// seeded device data. This does not qualify checkpoint weights, larger token
/// counts or image outputs.
/// MLX architectures where the BM64 qmm retile was measured faster than the
/// stock op: the M4. Other architectures keep the stock op until measured.
const QMM_BM64_MEASURED_ARCHITECTURES: &[&str] = &["applegpu_g16g"];

/// Rollback for the round-24 BM64 lever: `APXINF_QWEN_IMAGE_QMM_BM64=0`
/// restores the stock quantized matmul on the non-fused image tier.
fn qmm_bm64_disabled_from_env() -> bool {
    use std::sync::OnceLock;
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| {
        std::env::var("APXINF_QWEN_IMAGE_QMM_BM64")
            .is_ok_and(|v| matches!(v.as_str(), "0" | "false" | "off" | "no"))
    })
}

/// Enables the BM64 retiled qmm on a measured architecture after a
/// bit-identical probe against the stock op: the three block projection
/// shapes at a ragged row count (17*64 + 4, exercising the kernel's native
/// M-tail masking) plus a BM-divisible control, on seeded device data. The
/// retile is bit-exact by construction (same `qmm_t_impl`, only the compile
/// time tile changes); the probe guards against lane-specific surprises
/// before the fallback tier routes to it.
fn qmm_bm64_install(stream: &Stream) -> Option<Rc<QmmBm64>> {
    let architecture = stream
        .metal_caps()
        .map_or("", |caps| caps.architecture.as_str());
    if qmm_bm64_disabled_from_env() {
        eprintln!("Qwen-Image: BM64 qmm disabled (APXINF_QWEN_IMAGE_QMM_BM64=0); image fallback uses stock quantized matmul");
        return None;
    }
    if !QMM_BM64_MEASURED_ARCHITECTURES.contains(&architecture) {
        eprintln!(
            "Qwen-Image: BM64 qmm not used (not measured on {architecture:?}); image fallback uses stock quantized matmul"
        );
        return None;
    }
    let probe = || -> Result<Option<Rc<QmmBm64>>> {
        let bm64 = Rc::new(QmmBm64::new(stream)?);
        let normal = |shape: &[usize], seed: u64, scale: f32| -> Result<Array> {
            Array::random_normal(shape, MlxDType::F32, &Array::random_key(stream, seed)?)?
                .mul(&Array::scalar(stream, scale, MlxDType::F32)?)?
                .cast(MlxDType::BF16)
        };
        let shapes = [
            (17 * 64 + 4, DIM, DIM),
            (17 * 64 + 4, DIM, MLP * 2),
            (17 * 64 + 4, MLP, DIM),
            (16 * 64, DIM, DIM),
        ];
        for (seed, &(m, k, n)) in shapes.iter().enumerate() {
            let dense = normal(&[n, k], 2 * seed as u64, 0.02)?;
            let [weight, scales, biases] = dense.quantize(64, 4)?;
            let scales = scales.cast(MlxDType::BF16)?;
            let biases = biases.cast(MlxDType::BF16)?;
            let x = normal(&[1, m, k], 2 * seed as u64 + 1, 1.0)?;
            let reference = x.quantized_matmul(&weight, &scales, &biases, true, 64, 4)?;
            let out = bm64.call(&x, &weight, &scales, &biases)?;
            if out.to_bytes()? != reference.to_bytes()? {
                return Ok(None);
            }
        }
        Ok(Some(bm64))
    };
    match probe() {
        Ok(Some(bm64)) => Some(bm64),
        outcome => {
            let reason = match outcome {
                Err(e) => e.to_string(),
                _ => "results differ from quantized matmul".to_owned(),
            };
            eprintln!(
                "Qwen-Image: BM64 qmm not used ({reason}); image fallback uses stock quantized matmul"
            );
            None
        }
    }
}

fn probed_gemm(stream: &Stream) -> Option<MppBf16GemmNt> {
    let architecture = stream
        .metal_caps()
        .map_or("", |caps| caps.architecture.as_str());
    if !MPP_MEASURED_ARCHITECTURES.contains(&architecture) {
        eprintln!(
            "Qwen-Image: MPP GEMM not used (not measured on {architecture:?}); image projections use quantized matmul"
        );
        return None;
    }
    let probe = || -> Result<Option<MppBf16GemmNt>> {
        let gemm = MppBf16GemmNt::new(stream)?;
        let normal = |shape: &[usize], seed: u64, scale: f32| -> Result<Array> {
            Array::random_normal(shape, MlxDType::F32, &Array::random_key(stream, seed)?)?
                .mul(&Array::scalar(stream, scale, MlxDType::F32)?)?
                .cast(MlxDType::BF16)
        };
        let m = dit::MPP_MIN_TOKENS;
        for (seed, (k, n)) in [(4096, 4096), (4096, MLP), (MLP, 4096)]
            .into_iter()
            .enumerate()
        {
            let [q, scales, biases] = normal(&[n, k], 2 * seed as u64, 0.02)?.quantize(64, 4)?;
            let x = normal(&[1, m, k], 2 * seed as u64 + 1, 1.0)?;
            let reference = x.quantized_matmul(&q, &scales, &biases, true, 64, 4)?;
            let out = gemm.call(&x, &q.dequantize(&scales, &biases, 64, 4)?)?;
            if out.to_bytes()? != reference.to_bytes()? {
                return Ok(None);
            }
        }
        Ok(Some(gemm))
    };
    match probe() {
        Ok(Some(gemm)) => Some(gemm),
        outcome => {
            let reason = match outcome {
                Err(e) => e.to_string(),
                _ => "results differ from quantized matmul".to_owned(),
            };
            eprintln!(
                "Qwen-Image: MPP GEMM not used ({reason}); image projections use quantized matmul"
            );
            None
        }
    }
}

/// Env-gated (`APXINF_QWEN_IMAGE_TIMING=1`) stderr phase and per-step timing
/// for the long-prompt attention probe (EngineTailor #205): text encode,
/// text-prefix K/V build, compiled-step preparation, each denoise step, VAE
/// decode and the allocator peak, plus the text encoder's per-layer
/// load/compute split (EngineTailor #217). Off by default; when off, each
/// probe point costs one `Instant::now` pair whose delta is discarded.
pub(crate) fn timing_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("APXINF_QWEN_IMAGE_TIMING").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

/// Request admission, before any device work.
fn admit_request(request: &ImageRequest<'_>) -> Result<()> {
    if request.prompt.trim().is_empty() {
        return Err(Error::Contract("Qwen-Image needs a nonempty prompt"));
    }
    for side in [request.width, request.height] {
        if side == 0 || side % SCALE != 0 || side > MAX_SIDE {
            return Err(Error::Contract(
                "Qwen-Image width and height must be positive multiples of 16, at most 32768",
            ));
        }
    }
    (request.width / SCALE)
        .checked_mul(request.height / SCALE)
        .filter(|&tokens| tokens <= MAX_IMAGE_TOKENS)
        .ok_or(Error::Contract(
            "Qwen-Image area exceeds the MLP flattened i32 execution bound",
        ))?;
    if request.steps != STEPS {
        return Err(Error::Contract(
            "Qwen-Image supports exactly 40 denoising steps",
        ));
    }
    Ok(())
}

/// `LinearScheduler` sigmas with the Qwen-Image-2.1 shift and terminal
/// rescale, computed on the device as MLX computes them: FP32 `[steps + 1]`.
fn sigmas(ops: &Ops, width: usize, height: usize, steps: usize) -> Result<Array> {
    let s = ops.stream();
    let int = |v: usize| Array::from_i32(s, &[], &[v as i32]);
    // mx.linspace(1.0, 1 / steps, steps): t = arange / (n - 1); (1 - t) * a + t * b.
    let t = Array::arange(s, 0.0, steps as f32, 1.0, MlxDType::I32)?.div(&int(steps - 1)?)?;
    let linear = int(1)?
        .sub(&t)?
        .mul(&ops.f32(1.0)?)?
        .add(&t.mul(&ops.f32((1.0 / steps as f64) as f32)?)?)?;
    // shift_sigmas: mu from the image area, then the 0.02 terminal rescale.
    let m = (0.9 - 0.5) / (8192.0 - 256.0);
    let b = 0.5 - m * 256.0;
    let mu = m * width as f64 * height as f64 / 256.0 + b;
    let e = ops.f32(mu as f32)?.exp()?;
    let shifted = e.div(&e.add(&ops.f32(1.0)?.div(&linear)?.sub(&ops.f32(1.0)?)?)?)?;
    let one_minus = ops.f32(1.0)?.sub(&shifted)?;
    let scale = one_minus
        .slice(&[steps - 1], &[steps])?
        .reshape(&[])?
        .div(&ops.f32((1.0 - 0.02f64) as f32)?)?;
    let shifted = ops.f32(1.0)?.sub(&one_minus.div(&scale)?)?;
    let sigmas = Array::concat(&[&shifted, &Array::zeros(s, &[1], MlxDType::F32)?], 0)?;
    sigmas.eval()?;
    Ok(sigmas)
}

/// `Qwen21Rope` rows for `[text | image]`: FP32 `[rows, 64]` cos and sin.
/// Text advances one position on all axes; the image fixes the frame axis at
/// the text length and centers its height/width grid on zero.
fn rope(
    rope16: &(Vec<f32>, Vec<f32>),
    rope56: &(Vec<f32>, Vec<f32>),
    text: usize,
    lh: usize,
    lw: usize,
) -> (Vec<f32>, Vec<f32>) {
    let rows = text + lh * lw;
    let (mut cos, mut sin) = (Vec::with_capacity(rows * 64), Vec::with_capacity(rows * 64));
    let mut push = |position: [i64; 3]| {
        for (table, width, p) in [
            (rope16, 8, position[0]),
            (rope56, 28, position[1]),
            (rope56, 28, position[2]),
        ] {
            let r = tables::rope_row(p).expect("admitted sizes stay inside the RoPE tables");
            cos.extend_from_slice(&table.0[r * width..(r + 1) * width]);
            sin.extend_from_slice(&table.1[r * width..(r + 1) * width]);
        }
    };
    for p in 0..text as i64 {
        push([p; 3]);
    }
    let (top, left) = (-((lh - lh / 2) as i64), -((lw - lw / 2) as i64));
    for i in 0..lh as i64 {
        for j in 0..lw as i64 {
            push([text as i64, top + i, left + j]);
        }
    }
    (cos, sin)
}

impl ImageRuntime for QwenImage {
    fn generate(&mut self, request: &ImageRequest<'_>) -> Result<GeneratedImage> {
        admit_request(request)?;
        let timing = timing_enabled();
        let ops = &self.ops;
        let stream = ops.stream();
        let (width, height) = (request.width, request.height);
        let (lh, lw) = (height / SCALE, width / SCALE);
        let phase = std::time::Instant::now();
        let tokens = text::tokenize(&self.tokenizer, request.prompt)?;
        // Disk-persistent encode-output cache (round 22): a hit replaces the
        // whole text encode with a byte-identical load of a previous
        // evaluation's prompt states on the same weights.
        let prompt = match encode_cache::load(&self.text, &tokens, stream)? {
            Some(prompt) => prompt,
            None => {
                let prompt = text::encode(&self.text, &tokens, ops, lh * lw)?;
                encode_cache::store(&self.text, &tokens, &prompt)?;
                prompt
            }
        };
        if timing {
            eprintln!(
                "qimage-timing: text_encode {:.3}s raw_tokens={}",
                phase.elapsed().as_secs_f64(),
                prompt.shape()[1] + tokens.drop
            );
        }
        // The text encoder layers are freed; release their cached buffers as
        // mflux --low-ram does after deleting the encoder.
        apxinf_mlx::clear_cache()?;
        let key = Array::random_key(stream, request.seed)?;
        let mut latents = Array::random_normal(&[1, lh * lw, LATENT], MlxDType::F32, &key)?
            .cast(MlxDType::BF16)?;
        let sigmas = sigmas(ops, width, height, request.steps)?;
        let text_len = prompt.shape()[1];
        let (cos, sin) = rope(&self.rope16, &self.rope56, text_len, lh, lw);
        let split = text_len * 64;
        let table = |v: &[f32], rows: usize| Array::from_f32(stream, &[rows, 64], v);
        let transformer = match self.transformer.take() {
            Some(t) => t,
            None => Transformer::load(stream, &self.dir.join("transformer"))?,
        };
        let dit = dit::Dit::new(
            transformer,
            ops,
            self.gemm.clone(),
            self.fused.clone(),
            self.qmm_bm64.clone(),
        )?;
        #[cfg(feature = "coreml")]
        let dit = dit.with_ane(self.ane.clone());
        let mut dit = dit;
        let phase = std::time::Instant::now();
        let cache = dit.text_cache(
            &prompt,
            &table(&cos[..split], text_len)?,
            &table(&sin[..split], text_len)?,
        )?;
        if timing {
            eprintln!(
                "qimage-timing: text_cache {:.3}s text_len={} calls={}",
                phase.elapsed().as_secs_f64(),
                text_len,
                dit::text_cache_call_count()
            );
        }
        drop(prompt);
        dit.interleave_for_image(lh * lw)?;
        let (cos, sin) = (
            table(&cos[split..], lh * lw)?,
            table(&sin[split..], lh * lw)?,
        );
        latents.eval()?;
        let sigma = |i: usize| sigmas.slice(&[i], &[i + 1]);
        // TeaCache-style step reuse (mflux StepCache semantics): select the
        // steps whose consecutive sigma gap is smallest, outside the protected
        // head/tail window, then reuse the previous step's velocity there.
        let step_cache_ratio = match request.step_cache_ratio {
            Some(ratio) => ratio,
            None => DEFAULT_STEP_CACHE_RATIO,
        };
        let skip_steps = if step_cache_ratio > 0.0 {
            dit.select_skip_steps(&sigmas.slice(&[0], &[request.steps])?, step_cache_ratio)?
        } else {
            vec![false; request.steps]
        };
        let phase = std::time::Instant::now();
        let (step, first) = dit::Step::prepare(
            dit,
            &dit::Step::inputs(&latents, &sigma(0)?, &cos, &sin, &cache),
        )?;
        if timing {
            eprintln!(
                "qimage-timing: step_prepare {:.3}s (compile + first forward)",
                phase.elapsed().as_secs_f64()
            );
        }
        let mut first = Some(first);
        let mut previous_velocity: Option<Array> = None;
        let loop_start = std::time::Instant::now();
        for t in 0..request.steps {
            let step_start = std::time::Instant::now();
            let skipped = previous_velocity.is_some() && skip_steps[t];
            let velocity = match previous_velocity.clone() {
                Some(velocity) if skip_steps[t] => velocity,
                _ => {
                    let velocity = match first.take() {
                        Some(velocity) => velocity,
                        None => {
                            step.call(&dit::Step::inputs(&latents, &sigma(t)?, &cos, &sin, &cache))?
                        }
                    };
                    previous_velocity = Some(velocity.clone());
                    velocity
                }
            };
            let dt = sigma(t + 1)?
                .sub(&sigma(t)?)?
                .reshape(&[])?
                .cast(MlxDType::BF16)?;
            latents = latents.add(&velocity.cast(MlxDType::BF16)?.mul(&dt)?)?;
            latents.eval()?;
            if timing {
                eprintln!(
                    "qimage-timing: step {t} {:.1}ms {}",
                    step_start.elapsed().as_secs_f64() * 1e3,
                    if skipped { "skip" } else { "compute" },
                );
            }
        }
        if timing {
            eprintln!(
                "qimage-timing: denoise_loop {:.3}s text_cache_calls={}",
                loop_start.elapsed().as_secs_f64(),
                dit::text_cache_call_count()
            );
            #[cfg(feature = "coreml")]
            if let Some(s) = step.ane_spent() {
                eprintln!(
                    "qimage-timing: ane_block wait={:.3}s convert={:.3}s predict={:.3}s \
                     calls={} copies={} per_call={:.2}ms",
                    s.wait.get(),
                    s.convert.get(),
                    s.predict.get(),
                    s.calls.get(),
                    s.copies.get(),
                    1e3 * (s.wait.get() + s.convert.get() + s.predict.get())
                        / s.calls.get().max(1) as f64
                );
            }
        }
        // Release the DiT and its cached buffers before decoding, as mflux
        // --low-ram does.
        drop((cache, step));
        apxinf_mlx::clear_cache()?;
        let phase = std::time::Instant::now();
        let rgb = vae::decode(&self.decoder, ops, &latents, width, height)?;
        if timing {
            let stats = apxinf_mlx::memory_stats()?;
            eprintln!(
                "qimage-timing: vae_decode {:.3}s",
                phase.elapsed().as_secs_f64()
            );
            eprintln!(
                "qimage-timing: memory active={} cache={} peak={} bytes",
                stats.active_bytes, stats.cache_bytes, stats.peak_bytes
            );
        }
        Ok(GeneratedImage { width, height, rgb })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_variant_admission_needs_no_device() {
        for variant in [None, Some("auto"), Some("reference")] {
            assert!(
                admit(&LoadOptions {
                    model_variant: variant.map(str::to_owned),
                    ..LoadOptions::default()
                })
                .is_ok(),
                "rejected {variant:?}"
            );
        }
    }

    #[test]
    fn request_execution_bounds_need_no_device() {
        let request = ImageRequest {
            prompt: "a red bicycle",
            width: 1024,
            height: 1024,
            seed: 7,
            steps: STEPS,
            step_cache_ratio: None,
        };
        for (width, height) in [
            (16, 16),
            (512, 512),
            (1024, 576),
            (1024, 1024),
            (32768, 16),
            (16, 32768),
            (1824, 24528),
            (24528, 1824),
            (26880, 1664),
        ] {
            assert!(
                admit_request(&ImageRequest {
                    width,
                    height,
                    ..request
                })
                .is_ok(),
                "rejected {width}x{height}"
            );
        }
        for (width, height) in [
            (8192, 8192),
            (16384, 16384),
            (32768, 32768),
            (1824, 24544),
            (24544, 1824),
            (26880, 1680),
            (usize::MAX, usize::MAX),
            (usize::MAX - 15, 16),
            (16, usize::MAX - 15),
        ] {
            assert!(
                admit_request(&ImageRequest {
                    width,
                    height,
                    ..request
                })
                .is_err(),
                "accepted execution-invalid {width}x{height}"
            );
        }
    }

    #[test]
    fn admitted_schedules_remain_finite_and_decreasing_on_cpu() {
        let stream = Stream::cpu().unwrap();
        let ops = Ops::new(&stream).unwrap();
        for (width, height) in [
            (16, 16),
            (512, 512),
            (1024, 576),
            (1024, 1024),
            (1824, 24528),
            (26880, 1664),
        ] {
            let schedule = sigmas(&ops, width, height, STEPS)
                .unwrap()
                .to_f32_vec()
                .unwrap();
            assert_eq!(schedule.len(), STEPS + 1);
            assert_eq!(schedule[0], 1.0);
            assert!((schedule[STEPS - 1] - 0.02).abs() < 1e-6);
            assert_eq!(schedule[STEPS], 0.0);
            assert!(schedule.iter().all(|sigma| sigma.is_finite()));
            assert!(
                schedule.windows(2).all(|pair| pair[0] > pair[1]),
                "nondecreasing schedule for {width}x{height}: {schedule:?}"
            );
        }
    }

    #[test]
    fn admission_needs_no_device_or_checkpoint() {
        assert!(admit(&LoadOptions::default()).is_ok());
        for options in [
            LoadOptions {
                model_variant: Some("bf16".into()),
                ..LoadOptions::default()
            },
            LoadOptions {
                autotune: true,
                ..LoadOptions::default()
            },
        ] {
            assert!(admit(&options).is_err());
        }
        let request = ImageRequest {
            prompt: "a red bicycle",
            width: 1024,
            height: 576,
            seed: 7,
            steps: STEPS,
            step_cache_ratio: None,
        };
        assert!(admit_request(&request).is_ok());
        for bad in [
            ImageRequest {
                prompt: " \n",
                ..request
            },
            ImageRequest {
                width: 1000,
                ..request
            },
            ImageRequest {
                height: 0,
                ..request
            },
            ImageRequest {
                width: MAX_SIDE + 16,
                ..request
            },
            ImageRequest {
                steps: 20,
                ..request
            },
        ] {
            assert!(admit_request(&bad).is_err());
        }
    }

    #[test]
    #[ignore = "requires exclusive Apple Silicon Metal test slot and pinned MLX SDK"]
    fn mpp_gemm_is_probed_only_on_measured_architectures() {
        let stream = Stream::metal(0).unwrap();
        let architecture = stream.metal_caps().unwrap().architecture.clone();
        let measured = MPP_MEASURED_ARCHITECTURES.contains(&architecture.as_str());
        assert_eq!(probed_gemm(&stream).is_some(), measured, "{architecture}");
    }

    #[test]
    fn rope_positions_center_the_image_grid() {
        let tables = |dim| tables::rope_axis(dim, 10000.0);
        let (r16, r56) = (tables(16), tables(56));
        let (cos, _) = rope(&r16, &r56, 3, 2, 3);
        assert_eq!(cos.len(), (3 + 6) * 64);
        let row = |position: [i64; 3]| -> Vec<f32> {
            let mut out = Vec::new();
            for (t, w, p) in [
                (&r16, 8, position[0]),
                (&r56, 28, position[1]),
                (&r56, 28, position[2]),
            ] {
                let r = tables::rope_row(p).unwrap();
                out.extend_from_slice(&t.0[r * w..(r + 1) * w]);
            }
            out
        };
        assert_eq!(cos[2 * 64..3 * 64], row([2, 2, 2]));
        // First image token: frame 3, height -1, width -2; last: height 0, width 0.
        assert_eq!(cos[3 * 64..4 * 64], row([3, -1, -2]));
        assert_eq!(cos[8 * 64..9 * 64], row([3, 0, 0]));
    }
}
