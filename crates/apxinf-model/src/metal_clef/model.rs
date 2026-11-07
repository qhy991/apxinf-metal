//! Clef Flash forward pass: Qwen3.5 hybrid backbone in the stock or the
//! optimized GPU composition, and the BF16 joint schema head.
//!
//! The operator sequence and every rounding point follow the stock reference
//! (MLX-VLM 0.7.4 backbone, clef_mlx.py head, MLX 0.32.3), including its
//! compiled functions. The gated-delta recurrence is a Metal kernel with the
//! MLX-VLM kernel's arithmetic. The GPU composition replaces the MLP, the
//! gated-delta projections, convolution and recurrence, and the attention
//! projections with its own kernels. One request is one lazy graph evaluated
//! once; the optional Tier-1/Tier-2 prefix cache is the only state that can
//! survive between requests.
use super::config::{Config, MAX_INPUT_TOKENS};
use super::encode::{Encoded, EncodedQuestion};
use super::signpost;
use super::vision::{mrope_position_ids, Vision};
use super::weights::{
    Attention, Backbone, DecoderLayer, EvidenceLayer, GatedDelta, Head, Linear, Mha, Mixer, Mlp,
    Norm, Q4,
};
use apxinf_core::{Error, Result};
use apxinf_mlx::fusions::{
    padded_rows, CausalDepthwiseConv, F16GateSwiglu, F16Gemm, GatedDeltaLayout,
    GatedDeltaRecurrence, GdnConvSplit, MropeHalfSplitRope, PadRowsF16, Q4DecodeArithmetic,
    Q4RowDecodeF16, Q4TransposeDecode, RmsNormScale, SplitF16Gemm,
};
#[cfg(feature = "coreml")]
use apxinf_mlx::fusions::{AneTileHandoff, AneTileMerge, MppF16Gemm, SplitMppF16Gemm};
use apxinf_mlx::{Array, Compiled, MlxDType, Stream};
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};

const EPS: f32 = 1e-6;
const HEAD_EPS: f32 = 1e-5;
const G_BETA_WARM_LENGTHS: [usize; 3] = [1, 2, MAX_INPUT_TOKENS];
/// How many recently seen prepared-image identities Tier-2 may recognize. The
/// bound keeps a long-lived resident demo process from growing without limit.
const RECENT_IMAGE_KEYS: usize = 16;

/// `APXINF_CLEF_FUSION=1` enables the gated-delta q/k RMSNorm+scale fusion.
/// Default (unset) keeps the unfused `fast_rms_norm_unweighted` + `mul` path.
fn fusion_enabled() -> bool {
    fusion_flag(std::env::var("APXINF_CLEF_FUSION").ok())
}

/// The gate's value semantics, kept pure so the unit test can cover the
/// default and the enable spelling without mutating process env (which would
/// race across parallel tests).
fn fusion_flag(value: Option<String>) -> bool {
    value.as_deref() == Some("1")
}

/// The Tier-1 prefix-cache env gate (`APXINF_CLEF_PREFIX_CACHE`), default ON
/// (D20): unset fills and reuses the Tier-1 cache; only the explicit spelling
/// `0` disables it. The shared pure form keeps the unit test free of
/// process-env mutation (which would race across parallel tests), same as
/// [`fusion_flag`].
fn prefix_cache_flag(value: Option<String>) -> bool {
    value.as_deref() != Some("0")
}

fn prefix_cache_enabled() -> bool {
    prefix_cache_flag(std::env::var("APXINF_CLEF_PREFIX_CACHE").ok())
}

/// Tier-2's default after the reference-tier KEEP gate passed: ON. The
/// measurement runner still sets the gate explicitly on both arms; with this
/// const true the unset spelling means ON and only explicit `0` disables it,
/// mirroring Tier-1.
const TIER2_DEFAULT_ON: bool = true;

/// `APXINF_CLEF_PREFIX_CACHE_TIER2` gate. After KEEP the default is ON: unset
/// and lookalikes enable Tier-2 and only explicit `0` disables it, mirroring
/// [`prefix_cache_flag`]. Set [`TIER2_DEFAULT_ON`] to false to revert to the
/// qualification spelling (exact `1` only).
fn tier2_prefix_cache_flag(value: Option<String>) -> bool {
    if TIER2_DEFAULT_ON {
        value.as_deref() != Some("0")
    } else {
        value.as_deref() == Some("1")
    }
}

fn tier2_prefix_cache_enabled() -> bool {
    tier2_prefix_cache_flag(std::env::var("APXINF_CLEF_PREFIX_CACHE_TIER2").ok())
}

struct Kernels {
    recurrence: GatedDeltaRecurrence,
    /// Three-axis interleaved mRoPE; a text-only request has equal axes and
    /// degenerates to the previous half-split rotary exactly.
    rope: MropeHalfSplitRope,
    /// MLX-VLM `_compute_g_beta`, compiled shapeless as there: the fused
    /// kernel rounds exp/logaddexp differently from eager operations on some
    /// devices (one decay ulp per layer on the M1 Pro).
    g_beta: Compiled,
    /// Gated-delta q/k RMSNorm+scale; built only when `APXINF_CLEF_FUSION=1`.
    rms_norm_scale: Option<RmsNormScale>,
    /// Fusion #1: split the conv output into q/k/v in one pass; built only
    /// when `APXINF_CLEF_FUSION=1`.
    gdn_conv_split: Option<GdnConvSplit>,
}

/// Backbone composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Composition {
    /// MLX-VLM: Q4 matmuls, MLX convolution, the four-value recurrence.
    Reference,
    /// Optimized GPU kernels: gate GEMM with SwiGLU epilogue, wm1/wn2 split
    /// projections for the gated delta and attention q/o, one-pass k/v, the
    /// causal convolution kernel and the eight-lane recurrence. Its GDN
    /// reduction order differs from MLX-VLM's.
    Gpu,
    /// The GPU composition with intermediate MLP columns `[6656, 12288)` on
    /// the Neural Engine as one INT8 Core ML program per layer.
    #[cfg(feature = "coreml")]
    GpuAneColumns,
    /// The GPU composition's gated-delta and MLP kernels, the projections as
    /// Metal 4 MPP GEMMs, and each MLP split by neuron: 5632 INT8 and 1248
    /// FP16 neurons as one Neural Engine program per layer (128-token tiles,
    /// one batch prediction), the other 5408 on the GPU.
    #[cfg(feature = "coreml")]
    GpuAneNeurons,
}

/// The neuron split's projection GEMMs, in place of the Steel `split` and
/// `single`.
#[cfg(feature = "coreml")]
struct MppProjections {
    split: SplitMppF16Gemm,
    single: MppF16Gemm,
}

/// Kernels only the GPU composition uses.
struct GpuKernels {
    pad: PadRowsF16,
    /// Projection weights decoded in BF16, transposed to FP16.
    project_decode: Q4TransposeDecode,
    /// MLP down weight decoded in FP16, transposed.
    down_decode: Q4TransposeDecode,
    /// MLP gate and up weights decoded row-major.
    row_decode: Q4RowDecodeF16,
    /// Gate GEMM with the SwiGLU epilogue.
    gate_swiglu: F16GateSwiglu,
    /// Split projections (gated delta, attention q and o).
    split: SplitF16Gemm,
    /// Attention k and v in one pass.
    single: F16Gemm,
    /// Gated-delta convolution.
    conv: CausalDepthwiseConv,
    /// Eight value rows per SIMD group.
    recurrence: GatedDeltaRecurrence,
    /// Neuron split: the projections run as MPP GEMMs instead of `split`/`single`.
    #[cfg(feature = "coreml")]
    mpp: Option<MppProjections>,
}

impl GpuKernels {
    fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            pad: PadRowsF16::new(stream)?,
            project_decode: Q4TransposeDecode::new(stream, Q4DecodeArithmetic::Bf16)?,
            down_decode: Q4TransposeDecode::new(stream, Q4DecodeArithmetic::F16)?,
            row_decode: Q4RowDecodeF16::new(stream)?,
            gate_swiglu: F16GateSwiglu::new(stream)?,
            split: SplitF16Gemm::new(stream)?,
            single: F16Gemm::new(stream)?,
            conv: CausalDepthwiseConv::new(stream)?,
            recurrence: GatedDeltaRecurrence::with_layout(stream, GatedDeltaLayout::EightLanes)?,
            #[cfg(feature = "coreml")]
            mpp: None,
        })
    }

    /// Pad to 64 rows, decode each half of the Q4 weight, and run two wm1/wn2
    /// GEMMs writing one BF16 output.
    fn split_linear(&self, x: &Array, q: &Q4) -> Result<Array> {
        let tokens = x.shape()[1];
        let half = q.rows() / 2;
        let h = self.pad.call(x)?;
        #[cfg(feature = "coreml")]
        if let Some(mpp) = &self.mpp {
            let decode = |row0| {
                self.project_decode
                    .call(&q.weight, &q.scales, &q.biases, row0, half)
            };
            let low = mpp.split.low(&h, tokens, &decode(0)?)?;
            return mpp.split.high(&decode(half)?, &low, &h, tokens);
        }
        let low = self.split.low(
            &h,
            tokens,
            &self
                .project_decode
                .call(&q.weight, &q.scales, &q.biases, 0, half)?,
        )?;
        // Built after the low half and listed first, which fixes MLX's dispatch order.
        let high = self
            .project_decode
            .call(&q.weight, &q.scales, &q.biases, half, half)?;
        self.split.high(&high, &low, &h, tokens)
    }

    /// One full-width decode and one GEMM.
    fn single_linear(&self, x: &Array, q: &Q4) -> Result<Array> {
        let weight = self
            .project_decode
            .call(&q.weight, &q.scales, &q.biases, 0, q.rows())?;
        #[cfg(feature = "coreml")]
        if let Some(mpp) = &self.mpp {
            return mpp.single.call(&self.pad.call(x)?, x.shape()[1], &weight);
        }
        self.single.call(&self.pad.call(x)?, x.shape()[1], &weight)
    }

    /// The row-decoded up weight through an MLX GEMM, the gate GEMM with its
    /// SwiGLU epilogue, and the transposed down decode.
    fn mlp(&self, h: &Array, m: &Mlp) -> Result<Array> {
        let tokens = h.shape()[1];
        let x = if padded_rows(tokens) == tokens {
            h.cast(MlxDType::F16)?
        } else {
            self.pad.call(h)?
        };
        let decode = |q: &Q4| self.row_decode.call(&q.weight, &q.scales, &q.biases);
        let up = x.matmul(&decode(&m.up)?.transpose(&[1, 0])?)?;
        let activation = self.gate_swiglu.call(&x, &decode(&m.gate)?, &up)?;
        let down = self.down_decode.call(
            &m.down.weight,
            &m.down.scales,
            &m.down.biases,
            0,
            m.down.rows(),
        )?;
        activation
            .matmul(&down)?
            .slice_axis(1, 0, tokens)?
            .cast(MlxDType::BF16)
    }
}

/// The neuron split's per-layer MLP: the GPU neurons' gate rows (decoded per
/// call), their up and down
/// weights decoded once at load (FP16 `[5408, 4096]` each), and the Neural
/// Engine program for the other neurons.
#[cfg(feature = "coreml")]
struct NeuronMlp {
    gate: Q4,
    up: Array,
    down: Array,
    ane: super::ane::AneMlp,
}

#[cfg(feature = "coreml")]
impl NeuronMlp {
    fn new(c: &GpuKernels, cache: &apxinf_coreml::ProgramCache, mlp: &Mlp) -> Result<Self> {
        let dense = super::ane::DenseMlp::new(mlp)?;
        let (int8, fp16) = super::ane::neuron_counts()?;
        let split = dense.split(int8, fp16)?;
        let ane = super::ane::AneMlp::load_package(cache, &dense.program(&split)?)?;
        drop(dense);
        let stream = mlp.gate.weight.stream();
        let neurons: Vec<i32> = split.gpu.iter().map(|&n| n as i32).collect();
        let index = Array::from_i32(stream, &[neurons.len()], &neurons)?;
        let rows = |q: &Q4| -> Result<Q4> {
            Ok(Q4 {
                weight: q.weight.take(&index, 0)?,
                scales: q.scales.take(&index, 0)?,
                biases: q.biases.take(&index, 0)?,
            })
        };
        let gate = rows(&mlp.gate)?;
        let up = rows(&mlp.up)?;
        let up = c.row_decode.call(&up.weight, &up.scales, &up.biases)?;
        let d = &mlp.down;
        let down = c
            .down_decode
            .call(&d.weight, &d.scales, &d.biases, 0, d.rows())?
            .take(&index, 0)?;
        stream.eval(&[
            gate.weight.clone(),
            gate.scales.clone(),
            gate.biases.clone(),
            up.clone(),
            down.clone(),
        ])?;
        Ok(Self {
            gate,
            up,
            down,
            ane,
        })
    }
}

/// A layer MLP slot whose weights moved elsewhere (neuron split): empty arrays, so
/// the checkpoint's full Q4 weights are released.
#[cfg(feature = "coreml")]
fn moved_mlp(stream: &Stream) -> Result<Mlp> {
    let empty = || -> Result<Q4> {
        Ok(Q4 {
            weight: Array::zeros(stream, &[0, 0], MlxDType::U32)?,
            scales: Array::zeros(stream, &[0, 0], MlxDType::BF16)?,
            biases: Array::zeros(stream, &[0, 0], MlxDType::BF16)?,
        })
    };
    Ok(Mlp {
        gate: empty()?,
        up: empty()?,
        down: empty()?,
    })
}

/// `compute_g`: `exp(-exp(A_log) * softplus(a + dt_bias))`; beta: `sigmoid(b)`.
fn g_beta(args: &[Array]) -> Result<Vec<Array>> {
    let [a_log, a, b, dt_bias] = args else {
        unreachable!("g_beta takes four inputs")
    };
    let shifted = a.add(dt_bias)?;
    let softplus = shifted.logaddexp(&scalar(&shifted, 0.0)?)?;
    let decay = a_log
        .cast(MlxDType::F32)?
        .exp()?
        .neg()?
        .mul(&softplus)?
        .exp()?;
    Ok(vec![decay, b.sigmoid()?])
}

fn prepared_g_beta(stream: &Stream, value_heads: usize) -> Result<Compiled> {
    let callable = Compiled::with_shapeless(stream, 2, true, g_beta)?;
    for tokens in G_BETA_WARM_LENGTHS {
        let inputs = [
            Array::zeros(stream, &[value_heads], MlxDType::BF16)?,
            Array::zeros(stream, &[1, tokens, value_heads], MlxDType::BF16)?,
            Array::zeros(stream, &[1, tokens, value_heads], MlxDType::BF16)?,
            Array::zeros(stream, &[value_heads], MlxDType::BF16)?,
        ];
        drop(callable.prepare(&inputs)?);
    }
    Ok(callable)
}

pub struct Network {
    config: Config,
    stream: Stream,
    backbone: Backbone,
    head: Head,
    kernels: Kernels,
    gpu: Option<GpuKernels>,
    /// The column split's per-layer Neural Engine programs, in layer order.
    #[cfg(feature = "coreml")]
    ane: Vec<super::ane::AneMlp>,
    /// The neuron split's per-layer MLPs and its GPU/Neural Engine boundary kernels.
    #[cfg(feature = "coreml")]
    neurons: Vec<NeuronMlp>,
    #[cfg(feature = "coreml")]
    tiles: Option<(AneTileHandoff, AneTileMerge)>,
    /// MRoPE inverse frequencies, computed on the device as MLX-VLM does.
    inv_freq: Array,
    /// Which mRoPE axis feeds each frequency (`_interleaved_position_selector`).
    rope_selector: Array,
    /// Qwen3.5 vision tower, resident in every composition.
    vision: Vision,
    /// Tier-1 system/chat prefix cache (Q23/D18). Built lazily on the first
    /// request when `APXINF_CLEF_PREFIX_CACHE=1`; otherwise `None`, which is
    /// bit-identical to the pre-cache path. `RefCell` because the request
    /// network methods take `&self` while the first request must publish the
    /// fill. Single-threaded callers only (the worker and decide-server are
    /// serial), so no synchronization is needed.
    prefix_cache: RefCell<Option<PrefixCache>>,
    /// Tier-2 image-row prefix cache. It is filled only after the same
    /// prepared image has been seen at least once in this resident process;
    /// the first occurrence stays on the exact Tier-1 path so unique-image
    /// requests keep the established MLX parity and pay no fill penalty.
    tier2_cache: RefCell<Option<PrefixCache>>,
    /// Recently seen prepared-image identities, in request order. Bounded to
    /// [`RECENT_IMAGE_KEYS`]; membership decides whether Tier-2 may be used.
    recent_image_keys: RefCell<VecDeque<[u8; 32]>>,
    /// Prompt token IDs for the current request, kept so `backbone_cached`
    /// can hash and slice the prefix without re-reading device memory.
    last_prefix_ids: RefCell<Option<Vec<u32>>>,
    /// Prepared whole-backbone callables, keyed by `(chunk1 tokens, prefix
    /// tokens)`. Empty unless `APXINF_CLEF_COMPILE=1` and profiles were
    /// supplied; the eager path is the default and the fallback for any
    /// shape that was not prepared (the demo's dynamic lengths).
    compiled: RefCell<BTreeMap<(usize, usize), Compiled>>,
}

/// Per-layer state saved for the Tier-1 prefix cache (Q23): the F32 gated-delta
/// recurrence state and, for full-attention layers, the RoPE'd BF16 keys and
/// values. Both mixers also re-read the convolution state saved alongside it.
#[derive(Clone)]
struct CachedLayer {
    state: Array,
    conv: Array,
    keys: Option<Array>,
    values: Option<Array>,
}

/// Tier-1 system/chat prefix cache: the constant `<|im_start|>system ... STATE:\n`
/// plus the image start marker, whose inputs are identical across requests
/// (system prompt is a code constant, image content lives in later rows).
/// Layer-indexed to the full backbone; attention layers carry keys/values,
/// gated-delta layers leave them `None`.
#[derive(Clone)]
struct PrefixCache {
    prefix_tokens: usize,
    /// Bit-mix of the prefix token IDs; checked before reuse.
    prompt_hash: u64,
    /// Identity of the prepared vision input. Tier-2 compares this so two
    /// images with identical placeholder token ids never share a prefix.
    image_key: [u8; 32],
    /// The chunk-0 backbone output `[P, width]`, kept so a cache-hit request
    /// can still run `head_forward` over the full sequence: question/option
    /// spans, `memory`, `global` and the lexical rows are all indexed into the
    /// concatenated `[tokens, width]` hidden.
    prefix_hidden: Array,
    layers: Vec<CachedLayer>,
}

/// Stable-order bit-mix of the prefix token IDs. A change to the system/chat
/// text, media markers, or placeholder count changes the hash.
fn prefix_hash(ids: &[u32]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    ids.len().hash(&mut hasher);
    for &id in ids {
        id.hash(&mut hasher);
    }
    hasher.finish()
}

/// Whether a cached prefix may be reused for `prefix_end` prompt ids.
/// `image_key = None` skips the image comparison (Tier-1); `Some` requires the
/// prepared-image identity to match (Tier-2).
fn prefix_cache_matches(
    cache: &PrefixCache,
    prefix_end: usize,
    prompt_ids: &[u32],
    image_key: Option<&[u8; 32]>,
) -> bool {
    if cache.prefix_tokens != prefix_end {
        return false;
    }
    let end = prefix_end.min(prompt_ids.len());
    if cache.prompt_hash != prefix_hash(&prompt_ids[..end]) {
        return false;
    }
    if let Some(key) = image_key {
        if cache.image_key != *key {
            return false;
        }
    }
    true
}

/// Return the resident cache in `slot` when [`prefix_cache_matches`] accepts
/// it. `image_key = None` means "do not check the image", which is how Tier-1
/// stays independent of image content.
fn cached_prefix(
    slot: &RefCell<Option<PrefixCache>>,
    prefix_end: usize,
    prompt_ids: &[u32],
    image_key: Option<&[u8; 32]>,
) -> Option<PrefixCache> {
    let cache = slot.borrow().clone()?;
    prefix_cache_matches(&cache, prefix_end, prompt_ids, image_key).then_some(cache)
}

/// Python scalars combine with arrays in the array's dtype.
fn scalar(like: &Array, value: f64) -> Result<Array> {
    Array::scalar(like.stream(), value as f32, like.dtype())
}

fn silu(x: &Array) -> Result<Array> {
    x.mul(&x.sigmoid()?)
}

/// `nn.gelu`: `x * (1 + erf(x / sqrt(2))) / 2`.
fn gelu(x: &Array) -> Result<Array> {
    let inner = x.div(&scalar(x, std::f64::consts::SQRT_2)?)?.erf()?;
    x.mul(&scalar(x, 1.0)?.add(&inner)?)?.div(&scalar(x, 2.0)?)
}

fn qmm(x: &Array, q: &Q4) -> Result<Array> {
    x.quantized_matmul(&q.weight, &q.scales, &q.biases, true, 64, 4)
}

/// `nn.Linear` without bias: `x @ W.T`.
fn project(x: &Array, weight: &Array) -> Result<Array> {
    x.matmul(&weight.transpose(&[1, 0])?)
}

/// `nn.Linear` with bias: `addmm(b, x, W.T)`.
fn linear(x: &Array, l: &Linear) -> Result<Array> {
    l.bias.addmm(x, &l.weight.transpose(&[1, 0])?, 1.0, 1.0)
}

fn layer_norm(x: &Array, n: &Norm) -> Result<Array> {
    x.fast_layer_norm(Some(&n.weight), Some(&n.bias), HEAD_EPS)
}

/// `mx.stack(rows)` along a new leading axis.
fn stack(rows: &[Array]) -> Result<Array> {
    let expanded = rows
        .iter()
        .map(|r| {
            let mut shape = vec![1];
            shape.extend_from_slice(r.shape());
            r.reshape(&shape)
        })
        .collect::<Result<Vec<_>>>()?;
    Array::concat(&expanded.iter().collect::<Vec<_>>(), 0)
}

fn row(x: &Array, i: usize) -> Result<Array> {
    x.slice_axis(0, i, i + 1)?.reshape(&x.shape()[1..])
}

fn span_mean(x: &Array, (start, end): (usize, usize)) -> Result<Array> {
    x.slice_axis(0, start, end)?.mean(&[0], false)
}

/// `x / maximum(norm(x, axis=-1, keepdims=True), 1e-12)`.
fn l2_normalize(x: &Array) -> Result<Array> {
    let norm = x.norm(&[-1], true)?;
    x.div(&norm.maximum(&scalar(&norm, 1e-12)?)?)
}

/// `_interleaved_position_selector(mrope_section, freq_dim)`: frequency `i`
/// takes axis 0 unless the interleaved pattern of the later sections assigns
/// it (`i % 3 == axis` within that section's span).
fn mrope_selector(section: &[usize], frequencies: usize) -> Vec<i32> {
    let mut selector = vec![0i32; frequencies];
    let stride = section.len().max(1);
    for axis in 1..section.len() {
        let mut index = axis;
        while index < (section[axis] * stride).min(frequencies) {
            selector[index] = axis as i32;
            index += stride;
        }
    }
    selector
}

impl Network {
    pub fn new(
        config: Config,
        stream: Stream,
        #[cfg_attr(not(feature = "coreml"), allow(unused_mut))] mut backbone: Backbone,
        head: Head,
        vision: Vision,
        composition: Composition,
    ) -> Result<Self> {
        let kernels = Kernels {
            recurrence: GatedDeltaRecurrence::new(&stream)?,
            rope: MropeHalfSplitRope::new(&stream, config.rotary_dim())?,
            g_beta: prepared_g_beta(&stream, config.text.linear_num_value_heads)?,
            rms_norm_scale: fusion_enabled()
                .then(|| RmsNormScale::new(&stream, EPS))
                .transpose()?,
            gdn_conv_split: fusion_enabled()
                .then(|| GdnConvSplit::new(&stream))
                .transpose()?,
        };
        #[cfg_attr(not(feature = "coreml"), allow(unused_mut))]
        let mut gpu = match composition {
            Composition::Reference => None,
            _ => Some(GpuKernels::new(&stream)?),
        };
        // Column split: build each layer's program from the full MLP, then
        // keep only the GPU share's weights. The Neural Engine depth comes
        // from APXINF_CLEF_ANE_COLUMNS (unset: the qualified 6656).
        #[cfg(feature = "coreml")]
        let ane = if composition == Composition::GpuAneColumns {
            let columns = super::ane::gpu_columns()?;
            let cache = apxinf_coreml::ProgramCache::new(super::ane::cache_dir());
            let mut ane = Vec::with_capacity(backbone.layers.len());
            for layer in &mut backbone.layers {
                ane.push(super::ane::AneMlp::load_split(&cache, &layer.mlp, columns)?);
                layer.mlp = super::ane::gpu_share_with_columns(&layer.mlp, columns)?;
            }
            ane
        } else {
            Vec::new()
        };
        // Neuron split: split each MLP and build its program from the full
        // weights, keep the GPU share's weights, release the rest.
        #[cfg(feature = "coreml")]
        let (neurons, tiles) = match (composition, gpu.as_mut()) {
            (Composition::GpuAneNeurons, Some(c)) => {
                c.mpp = Some(MppProjections {
                    split: SplitMppF16Gemm::new(&stream)?,
                    single: MppF16Gemm::new(&stream)?,
                });
                let cache = apxinf_coreml::ProgramCache::new(super::ane::cache_dir());
                let mut neurons = Vec::with_capacity(backbone.layers.len());
                for layer in &mut backbone.layers {
                    neurons.push(NeuronMlp::new(c, &cache, &layer.mlp)?);
                    layer.mlp = moved_mlp(&stream)?;
                }
                let tiles = (AneTileHandoff::new(&stream)?, AneTileMerge::new(&stream)?);
                (neurons, Some(tiles))
            }
            _ => (Vec::new(), None),
        };
        // compute_inv_freq: 1 / (base ** (arange(0, dim, 2) / dim)) in FP32.
        let dim = config.rotary_dim();
        let exponents = Array::arange(&stream, 0.0, dim as f32, 2.0, MlxDType::F32)?
            .div(&Array::scalar(&stream, dim as f32, MlxDType::F32)?)?;
        let base = Array::scalar(
            &stream,
            config.text.rope_parameters.rope_theta,
            MlxDType::F32,
        )?;
        let inv_freq = Array::scalar(&stream, 1.0, MlxDType::F32)?.div(&base.pow(&exponents)?)?;
        stream.eval(&[inv_freq.clone()])?;
        // Interleaved 3-axis selector, from the admitted `mrope_section`.
        let frequencies = dim / 2;
        let section = config
            .text
            .rope_parameters
            .mrope_section
            .clone()
            .unwrap_or_else(|| vec![11, 11, 10]);
        let rope_selector = Array::from_i32(
            &stream,
            &[frequencies],
            &mrope_selector(&section, frequencies),
        )?;
        stream.eval(&[rope_selector.clone()])?;
        #[cfg(feature = "coreml")]
        for _ in 0..2 {
            for model in ane.iter().chain(neurons.iter().map(|layer| &layer.ane)) {
                model.warm()?;
            }
        }
        Ok(Self {
            config,
            stream,
            backbone,
            head,
            kernels,
            gpu,
            #[cfg(feature = "coreml")]
            ane,
            #[cfg(feature = "coreml")]
            neurons,
            #[cfg(feature = "coreml")]
            tiles,
            inv_freq,
            rope_selector,
            vision,
            prefix_cache: RefCell::new(None),
            tier2_cache: RefCell::new(None),
            recent_image_keys: RefCell::new(VecDeque::with_capacity(RECENT_IMAGE_KEYS)),
            last_prefix_ids: RefCell::new(None),
            compiled: RefCell::new(BTreeMap::new()),
        })
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Per-question logits for a text-only request, lazily; the caller
    /// evaluates them once.
    pub fn logits(&self, encoded: &Encoded) -> Result<Vec<Array>> {
        self.logits_with_vision(encoded, None)
    }

    /// Per-question logits for a request with optional image patches, lazily.
    ///
    /// The vision tower runs first, its features replace the prompt's image
    /// placeholder rows in order, and every attention layer uses the prompt's
    /// three-axis mRoPE positions. A request without images takes the same path
    /// with equal axes, so the text result is unchanged.
    ///
    /// Tier-2 is deliberately delayed until a prepared image has been seen at
    /// least once in this resident process. The first occurrence runs the
    /// exact Tier-1 path (same numerics, no image-prefix fill cost); a repeat
    /// may then fill or hit the Tier-2 cache. This keeps unique-image workloads
    /// — including the 42-case demo suite — on the established parity path.
    pub fn logits_with_vision(
        &self,
        encoded: &Encoded,
        vision: Option<(&Array, &[[u32; 3]], &[u8; 32])>,
    ) -> Result<Vec<Array>> {
        let tokens = encoded.input_ids.len();
        let width = self.config.text.hidden_size;
        let flat: Vec<i32> = encoded.input_ids.iter().map(|&id| id as i32).collect();
        if prefix_cache_enabled() && vision.is_some() {
            *self.last_prefix_ids.borrow_mut() = Some(encoded.input_ids.clone());
        }
        let ids = Array::from_i32(&self.stream, &[tokens], &flat)?;
        let e = &self.backbone.embedding;
        let mut hidden = e.weight.take(&ids, 0)?.dequantize(
            &e.scales.take(&ids, 0)?,
            &e.biases.take(&ids, 0)?,
            64,
            4,
        )?;

        // Positions are prompt/grid derived, so they can be computed before
        // the vision tower. Chunked RoPE must slice this full array; it is not
        // a serial arange after an image run.
        let positions = match vision {
            None => Self::text_positions(&self.stream, tokens)?,
            Some((_, grids, _)) => {
                let positions =
                    mrope_position_ids(&encoded.input_ids, grids, self.config.image_token_id)?;
                let mut flat = Vec::with_capacity(tokens * 3);
                for axis in 0..3 {
                    flat.extend(positions.iter().map(|position| position[axis]));
                }
                Array::from_i32(&self.stream, &[3, 1, tokens], &flat)?
            }
        };

        let image_key = vision.map(|(_, _, key)| *key);
        let repeat_image = image_key
            .map(|key| self.recent_image_keys.borrow().contains(&key))
            .unwrap_or(false);
        let tier2_prefix_end = (prefix_cache_enabled()
            && tier2_prefix_cache_enabled()
            && vision.is_some()
            && repeat_image)
            .then_some(encoded.media_end)
            .flatten()
            .filter(|&end| end < tokens);

        let tier2_hit_cache = tier2_prefix_end.and_then(|end| {
            vision.and_then(|(_, _, image_key)| {
                cached_prefix(&self.tier2_cache, end, &encoded.input_ids, Some(image_key))
            })
        });
        let tier2_hit = tier2_hit_cache.is_some();

        // A Tier-2 hit already contains the image-row prefix state, so the
        // vision tower and splice are skipped. On a Tier-2 fill (the repeat's
        // first request) or any Tier-1 request the normal vision path runs.
        if let Some((pixels, grids, _)) = vision {
            if !(tier2_prefix_end.is_some() && tier2_hit) {
                let features = self.vision.forward(pixels, grids)?;
                hidden = self.splice_vision(&encoded.input_ids, &hidden, &features)?;
            }
        }

        let hidden = if let Some(prefix_end) = tier2_prefix_end {
            self.backbone_cached(
                &self.tier2_cache,
                &hidden.reshape(&[1, tokens, width])?,
                &positions,
                tokens,
                prefix_end,
                vision.map(|(_, _, key)| key),
                tier2_hit_cache,
                true,
            )?
        } else if prefix_cache_enabled() && vision.is_some() {
            // Tier-1: the prefix ends at the first image placeholder, so the
            // image content is never part of the reused state. This is also
            // the path for a unique image's first occurrence.
            let prefix_end = encoded
                .input_ids
                .iter()
                .position(|&id| id == self.config.image_token_id)
                .unwrap_or(tokens);
            if prefix_end < tokens {
                self.backbone_cached(
                    &self.prefix_cache,
                    &hidden.reshape(&[1, tokens, width])?,
                    &positions,
                    tokens,
                    prefix_end,
                    vision.map(|(_, _, key)| key),
                    None,
                    false,
                )?
            } else {
                self.backbone_forward(
                    &hidden.reshape(&[1, tokens, width])?,
                    &positions,
                    tokens,
                    None,
                )?
            }
        } else {
            self.backbone_forward(
                &hidden.reshape(&[1, tokens, width])?,
                &positions,
                tokens,
                None,
            )?
        };

        // Remember the prepared image identity for the next request. The
        // deque is bounded, so a long-lived resident process cannot grow
        // without limit; repeated keys move to the back.
        if let Some(key) = image_key {
            let mut keys = self.recent_image_keys.borrow_mut();
            if let Some(position) = keys.iter().position(|seen| *seen == key) {
                keys.remove(position);
            }
            keys.push_back(key);
            while keys.len() > RECENT_IMAGE_KEYS {
                keys.pop_front();
            }
        }

        self.head_forward(&hidden.reshape(&[tokens, width])?, &ids, &encoded.questions)
    }

    /// Chunked backbone with a prefix cache. Tier-1 (`tier2 = false`) ends at
    /// the first image placeholder and ignores image identity on a hit; Tier-2
    /// (`tier2 = true`) ends after the media block and requires the prepared
    /// image key. On a miss the prefix is filled and its per-layer gated-delta
    /// state, conv history and attention KV are retained; on a hit the prefix
    /// chunk is skipped entirely. The two chunk outputs are concatenated so
    /// `head_forward` still sees one `[tokens, width]` hidden with the original
    /// absolute indexing.
    fn backbone_cached(
        &self,
        cache_slot: &RefCell<Option<PrefixCache>>,
        embeds: &Array,
        positions: &Array,
        tokens: usize,
        prefix_end: usize,
        image_key: Option<&[u8; 32]>,
        hit_cache: Option<PrefixCache>,
        tier2: bool,
    ) -> Result<Array> {
        let width = embeds.shape()[2];
        let hit_image_key = tier2.then_some(image_key).flatten();
        let cached = hit_cache.or_else(|| {
            cached_prefix(
                cache_slot,
                prefix_end,
                &self.prefix_ids(prefix_end),
                hit_image_key,
            )
        });
        let hit = cached.is_some();
        // `chunk0` = the cached prefix: on a hit it is skipped entirely (the
        // Tier-1/Tier-2 saving), on a miss it is the 32-layer fill. `chunk1`
        // = the dynamic rest, which always runs.
        let chunk0_start = std::time::Instant::now();
        let cache = match cached {
            Some(cache) => cache,
            None => {
                // Keep both chunks rank 3: the mixers shape their
                // intermediates from `[1, tokens, ...]`, and `prefix_hidden`
                // is the same rank so the two chunk outputs concat on the
                // token axis before `head_forward` flattens.
                let prefix = embeds.slice_axis(1, 0, prefix_end)?;
                self.fill_prefix_cache(cache_slot, &prefix, positions, prefix_end, image_key)?
            }
        };
        if let Some(ts0) = super::trace_json::to_us(chunk0_start) {
            super::trace_json::record(
                "chunk0",
                "chunk0",
                "prefix",
                ts0,
                chunk0_start.elapsed().as_secs_f64() * 1_000_000.0,
                serde_json::json!({
                    "cache_hit": hit,
                    "prefix_tokens": prefix_end,
                    "tier2": tier2,
                    "vision_skipped": tier2 && hit,
                }),
            );
        }
        let rest = embeds.slice_axis(1, prefix_end, tokens)?;
        let rest_positions = positions.slice_axis(2, prefix_end, tokens)?;
        let chunk1_start = std::time::Instant::now();
        let rest_hidden =
            self.backbone_forward(&rest, &rest_positions, tokens - prefix_end, Some(&cache))?;
        if let Some(ts0) = super::trace_json::to_us(chunk1_start) {
            super::trace_json::record(
                "chunk1",
                "chunk1",
                "rest",
                ts0,
                chunk1_start.elapsed().as_secs_f64() * 1_000_000.0,
                serde_json::json!({
                    "cache_hit": hit,
                    "prefix_tokens": prefix_end,
                    "rest_tokens": tokens - prefix_end,
                    "tier2": tier2,
                    "vision_skipped": tier2 && hit,
                }),
            );
        }
        Array::concat(&[&cache.prefix_hidden, &rest_hidden], 1)?.reshape(&[tokens, width])
    }

    /// Token IDs of the cached prefix (the same `input_ids` that fed the
    /// fill), read back from the resident ids array of the current request.
    fn prefix_ids(&self, prefix_end: usize) -> Vec<u32> {
        self.last_prefix_ids
            .borrow()
            .as_ref()
            .map(|ids| ids[..prefix_end.min(ids.len())].to_vec())
            .unwrap_or_default()
    }

    /// Run the cached prefix through the backbone once, capturing every
    /// layer's recurrence state, conv history and attention KV, plus the
    /// prefix's final hidden for `head_forward` on later hits. `image_key` is
    /// stored even for Tier-1 fills; Tier-1 simply does not compare it.
    fn fill_prefix_cache(
        &self,
        cache_slot: &RefCell<Option<PrefixCache>>,
        prefix: &Array,
        positions: &Array,
        prefix_end: usize,
        image_key: Option<&[u8; 32]>,
    ) -> Result<PrefixCache> {
        let prefix_positions = positions.slice_axis(2, 0, prefix_end)?;
        let mut layers = Vec::with_capacity(self.backbone.layers.len());
        let mut h = prefix.clone();
        for (index, layer) in self.backbone.layers.iter().enumerate() {
            // Same env-gated layer spans as `backbone_forward`, tagged
            // `chunk0` so the prefix-fill pass shows up on the layer lanes.
            let layer_start = std::time::Instant::now();
            let (mixer, mixer_c) = match &layer.mixer {
                Mixer::GatedDelta(_) => ("gated_delta", c"gated_delta"),
                Mixer::Attention(_) => ("attention", c"attention"),
            };
            // Host signpost for the chunk-0 prefix fill: index + mixer type.
            // `cached=0` — this pass seeds the cache rather than reading it.
            // The guard closes at the end of this iteration.
            let _sp =
                signpost_interval!(signpost::LAYER, index, mixer_c.as_ptr(), prefix_end, 0i32,);
            let x = h.fast_rms_norm(&layer.input_norm, EPS)?;
            let (r, state, conv, keys, values) = match &layer.mixer {
                Mixer::GatedDelta(g) => {
                    let mut conv = None;
                    let mut state = None;
                    // `gated_delta(save)` publishes the conv history first,
                    // then the recurrence's final state.
                    let r = self.gated_delta(&x, g, prefix_end, None, true, |a| {
                        if conv.is_none() {
                            conv = Some(a.clone());
                        } else {
                            state = Some(a.clone());
                        }
                        Ok(())
                    })?;
                    (
                        r,
                        state.expect("gated-delta state"),
                        conv.expect("conv history"),
                        None,
                        None,
                    )
                }
                Mixer::Attention(a) => {
                    let mut keys = None;
                    let mut values = None;
                    let r =
                        self.attention(&x, a, &prefix_positions, prefix_end, None, true, |a| {
                            if keys.is_none() {
                                keys = Some(a.clone());
                            } else {
                                values = Some(a.clone());
                            }
                            Ok(())
                        })?;
                    // Attention layers never read `state`/`conv`; these valid
                    // placeholders only satisfy the `CachedLayer` shape.
                    (
                        r,
                        Array::zeros(&self.stream, &[1, 1, 1, 1], MlxDType::F32)?,
                        Array::zeros(&self.stream, &[1, 1, 1], MlxDType::BF16)?,
                        Some(keys.expect("attention keys")),
                        Some(values.expect("attention values")),
                    )
                }
            };
            let h1 = h.add(&r)?;
            let m = self.mlp(&h1.fast_rms_norm(&layer.post_norm, EPS)?, &layer.mlp, index)?;
            h = h1.add(&m)?;
            layers.push(CachedLayer {
                state,
                conv,
                keys,
                values,
            });
            if let Some(ts0) = super::trace_json::to_us(layer_start) {
                super::trace_json::record(
                    "layer",
                    &format!("layer_{index}"),
                    mixer,
                    ts0,
                    layer_start.elapsed().as_secs_f64() * 1_000_000.0,
                    serde_json::json!({
                        "index": index,
                        "mixer": mixer,
                        "tokens": prefix_end,
                        "chunk": "chunk0",
                        "fill": true,
                    }),
                );
            }
        }
        let prefix_hidden = h.fast_rms_norm(&self.backbone.norm, EPS)?;
        let prefix_ids = self.last_prefix_ids.borrow().clone().unwrap_or_default();
        let cache = PrefixCache {
            prefix_tokens: prefix_end,
            prompt_hash: prefix_hash(&prefix_ids[..prefix_end.min(prefix_ids.len())]),
            image_key: image_key.copied().unwrap_or([0; 32]),
            prefix_hidden,
            layers,
        };
        *cache_slot.borrow_mut() = Some(cache.clone());
        Ok(cache)
    }

    /// Equal-axis positions of a text-only prompt.
    fn text_positions(stream: &Stream, tokens: usize) -> Result<Array> {
        let row = Array::arange(stream, 0.0, tokens as f32, 1.0, MlxDType::I32)?
            .reshape(&[1, 1, tokens])?;
        Array::concat(&[&row, &row, &row], 0)
    }

    /// Replace image placeholder rows with vision features, in prompt order.
    fn splice_vision(&self, input_ids: &[u32], hidden: &Array, features: &Array) -> Result<Array> {
        let tokens = hidden.shape()[0];
        let width = hidden.shape()[1];
        let rows = features.shape()[0];
        if features.shape() != [rows, width] {
            return Err(Error::Other(format!(
                "Clef vision features must be [{rows}, {width}], got {:?}",
                features.shape()
            )));
        }
        let mut index = Vec::with_capacity(tokens);
        let mut source = 0i32;
        let image = self.config.image_token_id;
        for (position, &id) in input_ids.iter().enumerate() {
            if id == image {
                index.push((tokens + source as usize) as i32);
                source += 1;
            } else {
                index.push(position as i32);
            }
        }
        if source as usize != rows {
            return Err(Error::Other(format!(
                "Clef prompt has {source} image placeholders for {rows} vision rows"
            )));
        }
        let combined = Array::concat(
            &[
                &hidden.reshape(&[1, tokens, width])?,
                &features.reshape(&[1, rows, width])?,
            ],
            1,
        )?;
        let index = Array::from_i32(&self.stream, &[tokens], &index)?;
        combined.take(&index, 1)?.reshape(&[tokens, width])
    }

    fn backbone_forward(
        &self,
        embeds: &Array,
        positions: &Array,
        tokens: usize,
        cache: Option<&PrefixCache>,
    ) -> Result<Array> {
        if let Some(cache) = cache {
            if let Some(hidden) = self.run_compiled(embeds, positions, tokens, cache)? {
                return Ok(hidden);
            }
        }
        self.forward_layers(embeds, positions, tokens, cache.map(|c| c.layers.as_slice()))
    }

    /// One prepared whole-backbone call for an exact (chunk1 tokens, prefix
    /// tokens) profile. Returns `None` when the profile was not prepared, so
    /// the caller uses the eager path (the demo's dynamic lengths).
    ///
    /// The compiled callable replays the 32-layer graph without running the
    /// host-side graph build; the prefix cache's per-layer state is passed as
    /// explicit inputs, in the same layer order the eager loop indexes.
    fn run_compiled(
        &self,
        embeds: &Array,
        positions: &Array,
        tokens: usize,
        cache: &PrefixCache,
    ) -> Result<Option<Array>> {
        let key = (tokens, cache.prefix_tokens);
        let compiled = self.compiled.borrow();
        let Some(callable) = compiled.get(&key) else {
            if std::env::var_os("APXINF_CLEF_COMPILE_TRACE").is_some() {
                eprintln!("Clef compiled backbone miss key={key:?}");
            }
            return Ok(None);
        };
        if std::env::var_os("APXINF_CLEF_COMPILE_TRACE").is_some() {
            eprintln!("Clef compiled backbone hit key={key:?}");
        }
        let mut inputs = Vec::with_capacity(2 + 2 * cache.layers.len());
        inputs.push(embeds.clone());
        inputs.push(positions.clone());
        for layer in &cache.layers {
            match (&layer.keys, &layer.values) {
                (Some(keys), Some(values)) => {
                    inputs.push(keys.clone());
                    inputs.push(values.clone());
                }
                _ => {
                    inputs.push(layer.state.clone());
                    inputs.push(layer.conv.clone());
                }
            }
        }
        let mut outputs = callable.call(&inputs)?;
        if outputs.len() != 1 {
            return Err(Error::Other(format!(
                "Clef compiled backbone returned {} outputs",
                outputs.len()
            )));
        }
        Ok(Some(outputs.remove(0)))
    }

    /// The eager layer loop, shared with the compiled body. `cached` is the
    /// per-layer prefix state in layer order; `None` is the uncached path.
    fn forward_layers(
        &self,
        embeds: &Array,
        positions: &Array,
        tokens: usize,
        cached: Option<&[CachedLayer]>,
    ) -> Result<Array> {
        let mut h = embeds.clone();
        // Env-gated per-layer spans (same flag as the rest of the trace).
        // These are host wall times of the lazy graph *build* per layer —
        // queue order plus mixer type/layer args, not GPU compute. The GPU
        // work lands in `logits_sync`; Perfetto overlays the layer lanes.
        let chrome = super::trace_json::enabled();
        let embed_start = std::time::Instant::now();
        let embed_ts = if chrome {
            super::trace_json::to_us(embed_start)
        } else {
            None
        };
        for (index, layer) in self.backbone.layers.iter().enumerate() {
            let layer_start = std::time::Instant::now();
            let cached = cached.and_then(|layers| layers.get(index));
            let (mixer, mixer_c) = match &layer.mixer {
                Mixer::GatedDelta(_) => ("gated_delta", c"gated_delta"),
                Mixer::Attention(_) => ("attention", c"attention"),
            };
            // Host signpost for the layer: index + mixer type, plus whether
            // this pass read a cached prefix state. It closes at the end of
            // this iteration and brackets the lazy-graph build.
            let _sp = signpost_interval!(
                signpost::LAYER,
                index,
                mixer_c.as_ptr(),
                tokens,
                cached.is_some() as i32,
            );
            let x = h.fast_rms_norm(&layer.input_norm, EPS)?;
            let r = match &layer.mixer {
                Mixer::GatedDelta(g) => {
                    self.gated_delta(&x, g, tokens, cached, false, |_| Ok(()))?
                }
                Mixer::Attention(a) => {
                    self.attention(&x, a, &positions, tokens, cached, false, |_| Ok(()))?
                }
            };
            let h1 = h.add(&r)?;
            let m = self.mlp(&h1.fast_rms_norm(&layer.post_norm, EPS)?, &layer.mlp, index)?;
            h = h1.add(&m)?;
            if let Some(ts0) = super::trace_json::to_us(layer_start) {
                super::trace_json::record(
                    "layer",
                    &format!("layer_{index}"),
                    mixer,
                    ts0,
                    layer_start.elapsed().as_secs_f64() * 1_000_000.0,
                    serde_json::json!({
                        "index": index,
                        "mixer": mixer,
                        "tokens": tokens,
                        "cached": cached.is_some(),
                    }),
                );
            }
        }
        let h = h.fast_rms_norm(&self.backbone.norm, EPS)?;
        if let Some(ts0) = embed_ts {
            // The `embed` span covers the whole backbone (norm + 32 layer
            // builds); the per-layer spans above sit on their own lanes.
            super::trace_json::record(
                "embed",
                "gpu",
                "backbone",
                ts0,
                embed_start.elapsed().as_secs_f64() * 1_000_000.0,
                serde_json::json!({"tokens": tokens, "layers": self.backbone.layers.len()}),
            );
        }
        Ok(h)
    }

    /// `Qwen3_5MLP`: `down(swiglu(gate(x), up(x)))` on Q4, or the GPU
    /// composition's MLP. With the column split, `m` is the GPU share and layer
    /// `index`'s program adds the rest: the tiles are read from the device, the
    /// GPU share is enqueued, and the Neural Engine predicts while the GPU runs.
    #[cfg_attr(not(feature = "coreml"), allow(unused_variables))]
    fn mlp(&self, h: &Array, m: &Mlp, index: usize) -> Result<Array> {
        // Neuron split: one kernel writes the GPU share's padded rows and the
        // Neural Engine tiles; the GPU share is queued behind it, the host
        // waits for the tiles only, the Neural Engine predicts every tile in
        // one batch, and one kernel adds its share to the GPU's.
        #[cfg(feature = "coreml")]
        if let (Some(c), Some((handoff, merge)), Some(layer_mlp)) =
            (&self.gpu, &self.tiles, self.neurons.get(index))
        {
            let tokens = h.shape()[1];
            let (x, packed) = handoff.call(h, super::ane::NEURON_TILE)?;
            let up = x.matmul(&layer_mlp.up.transpose(&[1, 0])?)?;
            let gate = c.row_decode.call(
                &layer_mlp.gate.weight,
                &layer_mlp.gate.scales,
                &layer_mlp.gate.biases,
            )?;
            let activation = c.gate_swiglu.call(&x, &gate, &up)?;
            let gpu = activation
                .matmul(&layer_mlp.down)?
                .slice_axis(1, 0, tokens)?;
            self.stream.async_eval(std::slice::from_ref(&packed))?;
            self.stream.async_eval(std::slice::from_ref(&gpu))?;
            let [y, gate_scale, up_scale] =
                layer_mlp
                    .ane
                    .predict_tiles(&self.stream, &packed, index, tokens)?;
            return merge.call(&gpu, &y, &gate_scale, &up_scale);
        }
        #[cfg(feature = "coreml")]
        if let (Some(c), Some(ane)) = (&self.gpu, self.ane.get(index)) {
            // The tile pack and the GPU share are both lazy graphs over `h`.
            // Queue the pack, then the GPU share, and block only on the pack:
            // the `to_bytes_ready` wait then leaves the GPU share running, so
            // the GPU and the Neural Engine overlap. Arithmetic and rounding points are
            // unchanged; only the issue order differs from the previous
            // (tiles read before any GPU work was queued) schedule.
            let tokens = h.shape()[1];
            let packed = super::ane::AneMlp::pack(h)?;
            let gpu = c.mlp(h, m)?;
            self.stream.async_eval(std::slice::from_ref(&packed))?;
            self.stream.async_eval(std::slice::from_ref(&gpu))?;
            let share = ane.predict(&self.stream, &packed, tokens, index)?;
            return gpu.add(&share);
        }
        match &self.gpu {
            Some(c) => c.mlp(h, m),
            None => qmm(&silu(&qmm(h, &m.gate)?)?.mul(&qmm(h, &m.up)?)?, &m.down),
        }
    }

    /// A gated-delta `in_proj_qkv`, `in_proj_z` or `out_proj`, or attention
    /// q/o: the GPU composition splits them in two halves.
    fn split_projection(&self, x: &Array, q: &Q4) -> Result<Array> {
        match &self.gpu {
            Some(c) => c.split_linear(x, q),
            None => qmm(x, q),
        }
    }

    /// Attention k/v: one pass in the GPU composition.
    fn single_projection(&self, x: &Array, q: &Q4) -> Result<Array> {
        match &self.gpu {
            Some(c) => c.single_linear(x, q),
            None => qmm(x, q),
        }
    }

    /// `save` publishes the recurrence's final F32 state and the convolution's
    /// trailing `taps - 1` rows for the prefix fill. Both are lazy graph
    /// outputs; `save` receives them synchronously after the recurrence
    /// returns, and the caller evaluates them when it materializes the cache.
    fn gated_delta(
        &self,
        x: &Array,
        g: &GatedDelta,
        tokens: usize,
        cached: Option<&CachedLayer>,
        save: bool,
        mut publish: impl FnMut(&Array) -> Result<()>,
    ) -> Result<Array> {
        let t = &self.config.text;
        let (hk, hv) = (t.linear_num_key_heads, t.linear_num_value_heads);
        let (dk, dv) = (t.linear_key_head_dim, t.linear_value_head_dim);
        let (key_dim, value_dim) = (hk * dk, hv * dv);
        let conv_dim = 2 * key_dim + value_dim;
        let mixed = self.split_projection(x, &g.qkv)?;
        let z = self
            .split_projection(x, &g.z)?
            .reshape(&[1, tokens, hv, dv])?;
        let (b, a) = (qmm(x, &g.b)?, qmm(x, &g.a)?);
        if save {
            let keep = t.linear_conv_kernel_dim - 1;
            publish(&mixed.slice_axis(1, tokens - keep, tokens)?)?;
        }
        let conv = match &self.gpu {
            Some(c) => silu(&c.conv.call_with_history(
                &mixed,
                &g.conv,
                cached.map(|c| &c.conv),
            )?)?,
            None => {
                let state = match cached {
                    Some(c) => c.conv.clone(),
                    None => Array::zeros(
                        &self.stream,
                        &[1, t.linear_conv_kernel_dim - 1, conv_dim],
                        MlxDType::BF16,
                    )?,
                };
                silu(&Array::concat(&[&state, &mixed], 1)?.conv1d(&g.conv, 1, 0, 1, conv_dim)?)?
            }
        };
        // Fusion #1 part A: one kernel splits the conv output into the three
        // contiguous q/k/v tensors (bit-preserving). Otherwise the three
        // slice+reshape pairs, whose consumers each force a row-contiguous
        // gather. Same values either way.
        let (q, k, v) = match &self.kernels.gdn_conv_split {
            Some(split) => split.call(&conv, hk, dk, hv, dv)?,
            None => (
                conv.slice_axis(2, 0, key_dim)?
                    .reshape(&[1, tokens, hk, dk])?,
                conv.slice_axis(2, key_dim, 2 * key_dim)?
                    .reshape(&[1, tokens, hk, dk])?,
                conv.slice_axis(2, 2 * key_dim, conv_dim)?
                    .reshape(&[1, tokens, hv, dv])?,
            ),
        };
        let inv_scale = (dk as f64).powf(-0.5);
        let norm_scale = |t: &Array, scale: f64| -> Result<Array> {
            let scale = scalar(t, scale)?;
            match &self.kernels.rms_norm_scale {
                Some(fused) => fused.call(t, &scale),
                // Unfused reference path: RMSNorm rounds to BF16, then the
                // BF16 * BF16 scalar multiply rounds to BF16.
                None => t.fast_rms_norm_unweighted(EPS)?.mul(&scale),
            }
        };
        let q = norm_scale(&q, inv_scale * inv_scale)?;
        let k = norm_scale(&k, inv_scale)?;
        let [decay, beta] = <[Array; 2]>::try_from(self.kernels.g_beta.call(&[
            g.a_log.clone(),
            a,
            b,
            g.dt_bias.clone(),
        ])?)
        .unwrap();
        let state = match cached {
            Some(c) => c.state.clone(),
            None => Array::zeros(&self.stream, &[1, hv, dv, dk], MlxDType::F32)?,
        };
        let recurrence = match &self.gpu {
            Some(c) => &c.recurrence,
            None => &self.kernels.recurrence,
        };
        let (y, next_state) = recurrence.call(&q, &k, &v, &decay, &beta, &state)?;
        if save {
            publish(&next_state)?;
        }
        // Gated RMSNorm: (silu(z) in FP32) * norm(y) in FP32, rounded to BF16.
        let normed = y.fast_rms_norm(&g.norm, EPS)?;
        let gated = silu(&z.cast(MlxDType::F32)?)?
            .mul(&normed.cast(MlxDType::F32)?)?
            .cast(MlxDType::BF16)?;
        self.split_projection(&gated.reshape(&[1, tokens, value_dim])?, &g.out)
    }

    /// `save` publishes the RoPE'd keys and values of the current chunk for
    /// the prefix fill; `publish` receives them in key-then-value order.
    fn attention(
        &self,
        x: &Array,
        a: &Attention,
        positions: &Array,
        tokens: usize,
        cached: Option<&CachedLayer>,
        save: bool,
        mut publish: impl FnMut(&Array) -> Result<()>,
    ) -> Result<Array> {
        let t = &self.config.text;
        let (heads, kv, d) = (t.num_attention_heads, t.num_key_value_heads, t.head_dim);
        let projected = self
            .split_projection(x, &a.q)?
            .reshape(&[1, tokens, heads, 2 * d])?;
        let (keys, values) = (
            self.single_projection(x, &a.k)?,
            self.single_projection(x, &a.v)?,
        );
        let queries = projected.slice_axis(3, 0, d)?;
        let gate = projected
            .slice_axis(3, d, 2 * d)?
            .reshape(&[1, tokens, heads * d])?;
        let queries = queries
            .fast_rms_norm(&a.q_norm, EPS)?
            .transpose(&[0, 2, 1, 3])?;
        let keys = keys
            .reshape(&[1, tokens, kv, d])?
            .fast_rms_norm(&a.k_norm, EPS)?
            .transpose(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[1, tokens, kv, d])?
            .transpose(&[0, 2, 1, 3])?;
        let rope = &self.kernels.rope;
        let queries = rope.call(&queries, positions, &self.inv_freq, &self.rope_selector)?;
        let keys = rope.call(&keys, positions, &self.inv_freq, &self.rope_selector)?;
        if save {
            publish(&keys)?;
            publish(&values)?;
        }
        let (keys, values) = match cached.and_then(|c| c.keys.as_ref().zip(c.values.as_ref())) {
            Some((cached_keys, cached_values)) => (
                Array::concat(&[cached_keys, &keys], 2)?,
                Array::concat(&[cached_values, &values], 2)?,
            ),
            None => (keys, values),
        };
        let out = queries
            .sdpa(&keys, &values, (d as f32).powf(-0.5), tokens > 1, None)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, tokens, heads * d])?;
        self.split_projection(&out.mul(&gate.sigmoid()?)?, &a.o)
    }

    /// `torch.nn.MultiheadAttention`, batch first, packed input projection.
    fn mha(&self, m: &Mha, q: &Array, k: &Array, v: &Array) -> Result<Array> {
        let width = self.config.head.width;
        let heads = self.config.head.heads;
        let head_dim = width / heads;
        let part = |x: &Array, i: usize| -> Result<Array> {
            let w = m.in_weight.slice_axis(0, i * width, (i + 1) * width)?;
            let b = m.in_bias.slice_axis(0, i * width, (i + 1) * width)?;
            let [batch, length, _] = *x.shape() else {
                unreachable!("head attention operands are rank 3")
            };
            project(x, &w)?
                .add(&b)?
                .reshape(&[batch, length, heads, head_dim])?
                .transpose(&[0, 2, 1, 3])
        };
        let (batch, length) = (q.shape()[0], q.shape()[1]);
        let o = part(q, 0)?.sdpa(
            &part(k, 1)?,
            &part(v, 2)?,
            (head_dim as f32).powf(-0.5),
            false,
            None,
        )?;
        let o = o
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[batch, length, width])?;
        linear(
            &o,
            &Linear {
                weight: m.out_weight.clone(),
                bias: m.out_bias.clone(),
            },
        )
    }

    fn evidence(&self, l: &EvidenceLayer, queries: &Array, memory: &Array) -> Result<Array> {
        let m = layer_norm(memory, &l.memory_norm)?;
        let queries = queries.add(&self.mha(
            &l.attention,
            &layer_norm(queries, &l.query_norm)?,
            &m,
            &m,
        )?)?;
        let ff = linear(
            &gelu(&linear(
                &layer_norm(&queries, &l.feedforward_norm)?,
                &l.fc1,
            )?)?,
            &l.fc2,
        )?;
        queries.add(&ff)
    }

    /// `torch.nn.TransformerDecoderLayer(norm_first=True, activation="gelu")`.
    fn decoder(&self, l: &DecoderLayer, x: &Array, memory: &Array) -> Result<Array> {
        let h = layer_norm(x, &l.norm1)?;
        let x = x.add(&self.mha(&l.self_attn, &h, &h, &h)?)?;
        let x = x.add(&self.mha(&l.cross_attn, &layer_norm(&x, &l.norm2)?, memory, memory)?)?;
        let ff = linear(
            &gelu(&linear(&layer_norm(&x, &l.norm3)?, &l.linear1)?)?,
            &l.linear2,
        )?;
        x.add(&ff)
    }

    /// `JointSchemaHead.__call__` for one record: hidden `[L, H]`.
    fn head_forward(
        &self,
        hidden: &Array,
        ids: &Array,
        questions: &[EncodedQuestion],
    ) -> Result<Vec<Array>> {
        // Production head span (previously only test instrumentation). Host
        // wall time of the lazy head build; the GPU drain is in `logits_sync`.
        let head_start = std::time::Instant::now();
        // Host signpost for the joint schema head: token and question count.
        let _sp = signpost_interval!(signpost::HEAD, hidden.shape()[0], questions.len(),);
        let hd = &self.head;
        let lm = &self.backbone.lm_head;
        let lexical = |span: (usize, usize)| -> Result<Array> {
            let rows = ids.slice_axis(0, span.0, span.1)?;
            lm.weight
                .take(&rows, 0)?
                .dequantize(
                    &lm.scales.take(&rows, 0)?,
                    &lm.biases.take(&rows, 0)?,
                    64,
                    4,
                )?
                .mean(&[0], false)
        };
        let h = layer_norm(hidden, &hd.hidden_norm)?;
        let length = h.shape()[0];
        let memory =
            project(&h, &hd.memory_projection)?.reshape(&[1, length, self.config.head.width])?;
        let global = row(&h, length - 1)?;
        let question_vectors = stack(
            &questions
                .iter()
                .map(|q| span_mean(&h, q.span))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let kinds: Vec<i32> = questions.iter().map(|q| q.kind as i32).collect();
        let kinds = Array::from_i32(&self.stream, &[kinds.len()], &kinds)?;

        let mut lexicals = Vec::with_capacity(questions.len());
        let mut option_queries = Vec::with_capacity(questions.len());
        for (i, q) in questions.iter().enumerate() {
            let context = stack(
                &q.option_spans
                    .iter()
                    .map(|&s| span_mean(&h, s))
                    .collect::<Result<Vec<_>>>()?,
            )?;
            let lex = stack(
                &q.option_spans
                    .iter()
                    .map(|&s| lexical(s))
                    .collect::<Result<Vec<_>>>()?,
            )?;
            let question = project(&row(&question_vectors, i)?, &hd.option_question_projection)?;
            let width = question.shape()[0];
            option_queries.push(
                project(&context, &hd.option_context_projection)?
                    .add(&project(&lex, &hd.option_lexical_projection)?)?
                    .add(&question.reshape(&[1, width])?)?,
            );
            lexicals.push(lex);
        }
        let mut routed = Array::concat(&option_queries.iter().collect::<Vec<_>>(), 0)?;
        let total = routed.shape()[0];
        let width = self.config.head.width;
        routed = routed.reshape(&[1, total, width])?;
        for layer in &hd.evidence {
            routed = self.evidence(layer, &routed, &memory)?;
        }
        let routed = routed.reshape(&[total, width])?;
        let mut split = Vec::with_capacity(questions.len());
        let mut start = 0;
        for q in questions {
            split.push(routed.slice_axis(0, start, start + q.option_spans.len())?);
            start += q.option_spans.len();
        }

        let base = project(&question_vectors, &hd.question_projection)?;
        let mut summaries = Vec::with_capacity(questions.len());
        for (i, options) in split.iter().enumerate() {
            let field = row(&base, i)?;
            let scores = options
                .matmul(&field)?
                .div(&scalar(options, (width as f64).sqrt())?)?;
            let w = scores.softmax_with(0, false)?;
            let n = w.shape()[0];
            summaries.push(w.reshape(&[n, 1])?.mul(options)?.sum(&[0], false)?);
        }
        let global_projected = project(&global, &hd.global_projection)?;
        let fields = base
            .add(&layer_norm(&stack(&summaries)?, &hd.option_summary_norm)?)?
            .add(&global_projected.reshape(&[1, width])?)?
            .add(&hd.type_embedding.take(&kinds, 0)?)?;
        let count = questions.len();
        let mut fields = fields.reshape(&[1, count, width])?;
        for layer in &hd.layers {
            fields = self.decoder(layer, &fields, &memory)?;
        }
        let fields = layer_norm(&fields.reshape(&[count, width])?, &hd.field_norm)?;

        let clamp = |s: &Array| -> Result<Array> { s.minimum(&scalar(s, 100f64.ln())?)?.exp() };
        let prior_scale = clamp(&hd.prior_logit_scale)?;
        let joint_scale = clamp(&hd.joint_logit_scale)?;
        let gate = hd.residual_gate.sigmoid()?;
        let mut logits = Vec::with_capacity(count);
        for i in 0..count {
            let field = row(&fields, i)?;
            let anchor = l2_normalize(&row(&question_vectors, i)?.add(&global)?)?;
            let prior = prior_scale.mul(&l2_normalize(&lexicals[i])?.matmul(&anchor)?)?;
            let options = layer_norm(&split[i], &hd.option_norm)?;
            let rf = field.reshape(&[1, width])?.broadcast_to(options.shape())?;
            let denominator = rf.norm(&[-1], false)?.mul(&options.norm(&[-1], false)?)?;
            let cosine = rf
                .mul(&options)?
                .sum(&[-1], false)?
                .div(&denominator.maximum(&scalar(&denominator, 1e-8)?)?)?;
            let features = Array::concat(
                &[&rf, &options, &rf.mul(&options)?, &rf.sub(&options)?.abs()?],
                1,
            )?;
            let n = options.shape()[0];
            let residual =
                linear(&gelu(&linear(&features, &hd.scorer1)?)?, &hd.scorer2)?.reshape(&[n])?;
            logits.push(prior.add(&gate.mul(&joint_scale.mul(&cosine)?.add(&residual)?)?)?);
        }
        if let Some(ts0) = super::trace_json::to_us(head_start) {
            super::trace_json::record(
                "head",
                "head",
                "schema",
                ts0,
                head_start.elapsed().as_secs_f64() * 1_000_000.0,
                serde_json::json!({
                    "tokens": hidden.shape()[0],
                    "questions": questions.len(),
                    "options": questions.iter().map(|q| q.option_spans.len()).sum::<usize>(),
                }),
            );
        }
        Ok(logits)
    }
}

/// `APXINF_CLEF_COMPILE=1` opts into the prepared whole-backbone callable.
/// Default OFF: the eager path is bit-for-bit the established implementation.
fn compiled_enabled() -> bool {
    matches!(
        std::env::var("APXINF_CLEF_COMPILE").as_deref(),
        Ok("1" | "true" | "on" | "yes")
    )
}

/// `APXINF_CLEF_COMPILE_PROFILES=L:P,L:P,...` lists the prompt-token and
/// prefix-token pairs to prepare at load. The 42-case suite needs the three
/// prompt lengths only; any other shape (the demo's dynamic lengths) falls
/// back to eager. An enabled compile with no profiles is a start-up error
/// rather than a silent eager run, so a measurement can never claim a
/// compiled arm it did not get.
fn compiled_profiles() -> Result<Option<Vec<(usize, usize)>>> {
    if !compiled_enabled() {
        return Ok(None);
    }
    let raw = std::env::var("APXINF_CLEF_COMPILE_PROFILES").map_err(|_| {
        Error::Contract(
            "APXINF_CLEF_COMPILE=1 requires APXINF_CLEF_COMPILE_PROFILES=L:P,...".into(),
        )
    })?;
    let mut profiles = Vec::new();
    for part in raw.split(',').map(str::trim).filter(|part| !part.is_empty()) {
        let (prompt, prefix) = part.split_once(':').ok_or_else(|| {
            Error::Other(format!(
                "APXINF_CLEF_COMPILE_PROFILES entry {part:?} is not L:P"
            ))
        })?;
        let prompt: usize = prompt.trim().parse().map_err(|_| {
            Error::Other(format!("APXINF_CLEF_COMPILE_PROFILES prompt {prompt:?} is not a length"))
        })?;
        let prefix: usize = prefix.trim().parse().map_err(|_| {
            Error::Other(format!("APXINF_CLEF_COMPILE_PROFILES prefix {prefix:?} is not a length"))
        })?;
        if prompt <= prefix {
            return Err(Error::Other(format!(
                "APXINF_CLEF_COMPILE_PROFILES {part:?} needs prompt > prefix"
            )));
        }
        profiles.push((prompt, prefix));
    }
    if profiles.is_empty() {
        return Err(Error::Contract(
            "APXINF_CLEF_COMPILE_PROFILES lists no profiles".into(),
        ));
    }
    Ok(Some(profiles))
}

/// Prepare one shape-specialized whole-backbone callable per profile. The
/// callable captures the network weakly: the compiled graph and the network
/// are one object, so a strong capture would be a reference cycle.
///
/// Called once after `Network::new` (load time). Each profile is traced and
/// replayed twice by `Compiled::prepare`, the established 2.2-2.8 s cost.
pub fn install_compiled(network: &Rc<Network>) -> Result<()> {
    let Some(profiles) = compiled_profiles()? else {
        return Ok(());
    };
    // The compiled body replays the reference layer loop (Q4 qmm + SDPA +
    // gated-delta kernels). The GPU/ANE compositions enqueue Core ML calls
    // that `mx.compile` cannot replay, so the flag is reference-only.
    if network.gpu.is_some() {
        return Err(Error::Other(
            "APXINF_CLEF_COMPILE supports the reference composition only".into(),
        ));
    }
    let count = profiles.len();
    for (prompt, prefix) in profiles {
        let rest = prompt - prefix;
        let inputs = network.compiled_inputs(rest, prefix)?;
        let weak: Weak<Network> = Rc::downgrade(network);
        let callable = Compiled::new(&network.stream, 1, move |args| {
            let network = weak.upgrade().ok_or_else(|| {
                Error::Other("Clef compiled backbone: network was dropped".into())
            })?;
            network.forward_compiled(args)
        })?;
        callable.prepare(&inputs)?;
        network
            .compiled
            .borrow_mut()
            .insert((rest, prefix), callable);
    }
    eprintln!(
        "Clef: prepared {count} whole-backbone compile profile(s) (APXINF_CLEF_COMPILE)"
    );
    Ok(())
}

impl Network {
    /// The exact input shapes of one prepared profile, in the order
    /// `run_compiled`/`forward_compiled` use: hidden, positions, then two
    /// arrays per layer (attention keys/values or gated-delta state/conv).
    fn compiled_inputs(&self, rest: usize, prefix: usize) -> Result<Vec<Array>> {
        let t = &self.config.text;
        let width = t.hidden_size;
        let mut inputs = Vec::with_capacity(2 + 2 * self.backbone.layers.len());
        inputs.push(Array::zeros(
            &self.stream,
            &[1, rest, width],
            MlxDType::BF16,
        )?);
        inputs.push(Array::zeros(
            &self.stream,
            &[3, 1, rest],
            MlxDType::I32,
        )?);
        let (hk, hv) = (t.linear_num_key_heads, t.linear_num_value_heads);
        let (dk, dv) = (t.linear_key_head_dim, t.linear_value_head_dim);
        let conv_dim = 2 * hk * dk + hv * dv;
        for layer in &self.backbone.layers {
            match &layer.mixer {
                Mixer::GatedDelta(_) => {
                    inputs.push(Array::zeros(
                        &self.stream,
                        &[1, hv, dv, dk],
                        MlxDType::F32,
                    )?);
                    inputs.push(Array::zeros(
                        &self.stream,
                        &[1, t.linear_conv_kernel_dim - 1, conv_dim],
                        MlxDType::BF16,
                    )?);
                }
                Mixer::Attention(_) => {
                    let (kv, d) = (t.num_key_value_heads, t.head_dim);
                    let shape = [1, kv, prefix, d];
                    inputs.push(Array::zeros(&self.stream, &shape, MlxDType::BF16)?);
                    inputs.push(Array::zeros(&self.stream, &shape, MlxDType::BF16)?);
                }
            }
        }
        Ok(inputs)
    }

    /// The compiled body: rebuild the per-layer cache view from the explicit
    /// inputs and run the same eager layer loop. It only executes while the
    /// profile is traced/prepared; replay goes straight to the compiled graph.
    fn forward_compiled(&self, args: &[Array]) -> Result<Vec<Array>> {
        let count = self.backbone.layers.len();
        if args.len() != 2 + 2 * count {
            return Err(Error::Other(format!(
                "Clef compiled backbone takes {} inputs, got {}",
                2 + 2 * count,
                args.len()
            )));
        }
        let mut layers = Vec::with_capacity(count);
        for (index, layer) in self.backbone.layers.iter().enumerate() {
            let first = &args[2 + 2 * index];
            let second = &args[3 + 2 * index];
            layers.push(match &layer.mixer {
                Mixer::GatedDelta(_) => CachedLayer {
                    state: first.clone(),
                    conv: second.clone(),
                    keys: None,
                    values: None,
                },
                Mixer::Attention(_) => CachedLayer {
                    state: Array::zeros(&self.stream, &[1, 1, 1, 1], MlxDType::F32)?,
                    conv: Array::zeros(&self.stream, &[1, 1, 1], MlxDType::BF16)?,
                    keys: Some(first.clone()),
                    values: Some(second.clone()),
                },
            });
        }
        let hidden = &args[0];
        let positions = &args[1];
        let tokens = hidden.shape()[1];
        Ok(vec![
            self.forward_layers(hidden, positions, tokens, Some(&layers))?
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g_beta_inputs(stream: &Stream, tokens: usize, variant: usize) -> Result<Vec<Array>> {
        let value_heads = 32;
        let make = |shape: &[usize], count: usize, offset: f32| {
            let values = (0..count)
                .map(|index| (index % 17) as f32 / 16.0 + variant as f32 / 8.0 + offset)
                .collect::<Vec<_>>();
            Array::from_f32(stream, shape, &values)?.cast(MlxDType::BF16)
        };
        Ok(vec![
            make(&[value_heads], value_heads, -2.0)?,
            make(&[1, tokens, value_heads], tokens * value_heads, -1.0)?,
            make(&[1, tokens, value_heads], tokens * value_heads, -0.5)?,
            make(&[value_heads], value_heads, -0.25)?,
        ])
    }

    fn check_g_beta_preparation(stream: Stream) -> Result<()> {
        let prepared = prepared_g_beta(&stream, 32)?;
        let original = Compiled::with_shapeless(&stream, 2, true, g_beta)?;
        let nonsingleton_first = Compiled::with_shapeless(&stream, 2, true, g_beta)?;
        let first_inputs = g_beta_inputs(&stream, 17, 2)?;
        let first_outputs = nonsingleton_first.call_and_eval(&first_inputs)?;
        let first_bytes = first_outputs
            .iter()
            .map(Array::to_bytes)
            .collect::<Result<Vec<_>>>()?;
        let mut retained = Vec::new();
        for (tokens, variant) in [
            (1, 0),
            (2, 1),
            (17, 2),
            (3, 3),
            (2, 4),
            (1, 0),
            (512, 3),
            (4096, 4),
        ] {
            let inputs = g_beta_inputs(&stream, tokens, variant)?;
            let before = stream.counters();
            let actual = prepared.call_and_eval(&inputs)?;
            let after = stream.counters();
            assert_eq!(after.trace_callbacks, before.trace_callbacks);
            assert_eq!(after.uploads_bytes, before.uploads_bytes);
            assert_eq!(after.downloads_bytes, before.downloads_bytes);
            assert_eq!(actual[0].shape(), &[1, tokens, 32]);
            assert_eq!(actual[0].dtype(), MlxDType::F32);
            assert_eq!(actual[1].dtype(), MlxDType::BF16);
            let expected = original.call_and_eval(&inputs)?;
            let expected_bytes = expected
                .iter()
                .map(Array::to_bytes)
                .collect::<Result<Vec<_>>>()?;
            for (index, (output, expected)) in actual.iter().zip(expected_bytes).enumerate() {
                let bytes = output.to_bytes()?;
                assert_eq!(bytes, expected);
                if tokens == 17 {
                    assert_eq!(bytes, first_bytes[index]);
                }
            }
            retained.push(actual);
        }
        let before_drop = retained
            .iter()
            .flatten()
            .map(Array::to_bytes)
            .collect::<Result<Vec<_>>>()?;
        drop(prepared);
        drop(original);
        drop(nonsingleton_first);
        let after_drop = retained
            .iter()
            .flatten()
            .map(Array::to_bytes)
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(after_drop, before_drop);
        Ok(())
    }

    #[test]
    fn g_beta_preparation_preserves_values_and_replays_without_host_io() -> Result<()> {
        check_g_beta_preparation(Stream::cpu()?)
    }

    #[test]
    fn g_beta_preparation_rejects_foreign_inputs_without_poisoning_replay() -> Result<()> {
        let stream = Stream::cpu()?;
        let foreign = Stream::cpu()?;
        let prepared = prepared_g_beta(&stream, 32)?;
        let foreign_prepared = prepared_g_beta(&foreign, 32)?;
        assert!(prepared.call(&g_beta_inputs(&foreign, 2, 1)?).is_err());
        assert!(foreign_prepared
            .call(&g_beta_inputs(&stream, 2, 1)?)
            .is_err());
        let inputs = g_beta_inputs(&stream, 2, 1)?;
        let before = stream.counters().trace_callbacks;
        let result = prepared.call_and_eval(&inputs)?;
        assert_eq!(stream.counters().trace_callbacks, before);
        assert!(result[0]
            .to_f32_vec()?
            .iter()
            .all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    #[ignore = "requires an Apple Metal GPU and the shared device lease"]
    fn g_beta_preparation_preserves_metal_values_and_replays_without_host_io() -> Result<()> {
        check_g_beta_preparation(Stream::metal(0)?)
    }

    /// The W1 column-split issue order (`pack → async_eval(packed) →
    /// async_eval(gpu) → host(packed)`) yields the same tile bytes as the
    /// eager read the previous (`tiles` before any GPU queueing) schedule used.
    /// The fixture is a stand-in for `GpuKernels::mlp` (an `Array` operand,
    /// constructed once as a fresh graph, queued second), since that kernel
    /// needs no checkpoint.
    #[cfg(feature = "coreml")]
    #[test]
    fn w1_column_issue_order_is_bit_equal() -> Result<()> {
        use super::super::ane::AneMlp;
        let stream = Stream::cpu()?;
        let hidden = 4096;
        for tokens in [1, 127, 128, 129, 323, 512, 513] {
            let h = Array::from_f32(
                &stream,
                &[1, tokens, hidden],
                &(0..tokens * hidden)
                    .map(|index| (index % 251) as f32 / 16.0)
                    .collect::<Vec<_>>(),
            )?
            .cast(MlxDType::BF16)?;
            // Previous schedule: read the pack before any other work is queued.
            let old = AneMlp::host(&AneMlp::pack(&h)?)?;
            // W1 schedule: queue both, then wait only for the pack.
            let packed = AneMlp::pack(&h)?;
            let gpu = h.mul(&scalar(&h, 0.0)?)?; // fresh graph, queues second
            stream.async_eval(std::slice::from_ref(&packed))?;
            stream.async_eval(std::slice::from_ref(&gpu))?;
            let new = AneMlp::host(&packed)?;
            assert_eq!(new, old, "tokens={tokens}");
            // The outstanding GPU share must still be evaluable afterwards;
            // BF16 (`h.mul` preserves the dtype) is 2 bytes per element.
            assert_eq!(
                gpu.to_bytes()?.len(),
                tokens * hidden * 2,
                "tokens={tokens}"
            );
        }
        Ok(())
    }

    /// Fusion #2 env gate semantics: the documented default (variable unset)
    /// is off, only the exact spelling `1` turns it on, and lookalikes stay
    /// off. Pure, no process-env mutation, so it is safe under parallel test
    /// execution. The byte-for-byte fused==unfused comparison is in
    /// `apxinf-mlx/tests/metal_fusions.rs` (needs a Metal stream).
    #[test]
    fn fusion_env_gate_defaults_to_off() {
        assert!(!fusion_flag(None), "unset must default to the unfused path");
        assert!(fusion_flag(Some("1".to_owned())));
        for off in ["", "0", "true", "yes", "2", "1 ", "01"] {
            assert!(
                !fusion_flag(Some(off.to_owned())),
                "{off:?} must not enable the fusion"
            );
        }
    }

    /// Q23 prefix-cache env gate (D20) defaults ON: unset or any lookalike
    /// fills and reuses the Tier-1 cache; only explicit `0` disables it.
    #[test]
    fn prefix_cache_env_gate_defaults_to_on() {
        assert!(prefix_cache_flag(None), "unset must enable the cache");
        assert!(!prefix_cache_flag(Some("0".to_owned())), "0 disables it");
        for on in ["", "1", "true", "yes", "2", "1 ", "01", "fill"] {
            assert!(
                prefix_cache_flag(Some(on.to_owned())),
                "{on:?} must not disable the prefix cache"
            );
        }
    }

    /// Tier-2 follows `TIER2_DEFAULT_ON`. After KEEP that const is true, so the
    /// gate mirrors Tier-1 (unset/lookalikes on, explicit `0` off). The same
    /// test also covers the OFF spelling if the const is reverted.
    #[test]
    fn tier2_prefix_cache_env_gate_follows_the_configured_default() {
        assert_eq!(
            tier2_prefix_cache_flag(None),
            TIER2_DEFAULT_ON,
            "unset must follow the current default"
        );
        if TIER2_DEFAULT_ON {
            assert!(!tier2_prefix_cache_flag(Some("0".to_owned())));
            for on in ["", "1", "true", "yes", "2", "1 ", "01", "fill"] {
                assert!(tier2_prefix_cache_flag(Some(on.to_owned())), "{on:?}");
            }
        } else {
            assert!(tier2_prefix_cache_flag(Some("1".to_owned())));
            for off in ["", "0", "true", "yes", "2", "1 ", "01", "fill"] {
                assert!(!tier2_prefix_cache_flag(Some(off.to_owned())), "{off:?}");
            }
        }
    }

    /// The cache match is the correctness boundary for both tiers: Tier-1
    /// ignores image identity, Tier-2 requires it, and both require the exact
    /// prefix length and prompt hash.
    #[test]
    fn prefix_cache_match_requires_boundary_hash_and_optional_image_key() -> Result<()> {
        let stream = Stream::cpu()?;
        let ids = [1u32, 2, 3, 4, 5];
        let image_key = [7u8; 32];
        let cache = PrefixCache {
            prefix_tokens: ids.len(),
            prompt_hash: prefix_hash(&ids),
            image_key,
            prefix_hidden: Array::zeros(&stream, &[1, ids.len(), 4], MlxDType::BF16)?,
            layers: Vec::new(),
        };
        // Tier-2: image identity participates.
        assert!(prefix_cache_matches(
            &cache,
            ids.len(),
            &ids,
            Some(&image_key)
        ));
        // Tier-1: image identity is deliberately not checked.
        assert!(prefix_cache_matches(&cache, ids.len(), &ids, None));
        assert!(!prefix_cache_matches(
            &cache,
            ids.len() - 1,
            &ids,
            Some(&image_key)
        ));
        assert!(!prefix_cache_matches(
            &cache,
            ids.len(),
            &[1, 2, 3, 4, 6],
            Some(&image_key)
        ));
        let other_key = [8u8; 32];
        assert!(!prefix_cache_matches(
            &cache,
            ids.len(),
            &ids,
            Some(&other_key)
        ));
        assert!(!prefix_cache_matches(
            &cache,
            ids.len(),
            &ids[..3],
            Some(&image_key)
        ));
        Ok(())
    }

    /// The full MLP call (column-split branch) is bit-identical before and
    /// after W1: the new issue order against a replay of the previous
    /// (`tiles` read before the GPU share is queued) order, same `Network`,
    /// same tile bytes. Needs the checkpoint, a Metal GPU and the column-split
    /// programs (`APXINF_CLEF_CHECKPOINT`; `APXINF_COREML_CACHE` should point
    /// at a fresh directory in the measurement window).
    #[cfg(feature = "coreml")]
    #[test]
    #[ignore = "requires the Clef checkpoint, compiled ANE programs and a Metal device"]
    fn w1_column_split_mlp_is_bit_equal() -> Result<()> {
        use super::super::ane::AneMlp;
        let var = |name: &str| {
            std::path::PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")))
        };
        let checkpoint = var("APXINF_CLEF_CHECKPOINT");
        let read = |name: &str| {
            std::fs::read_to_string(checkpoint.join(name))
                .map_err(|error| Error::Other(format!("cannot read {name}: {error}")))
        };
        let stream = Stream::metal(0)?;
        let config = Config::from_json(&read("config.json")?, &read("joint_head_config.json")?)?;
        let tokens = 323;
        let hidden = config.text.hidden_size;
        let backbone = Backbone::load(&config, &stream, &checkpoint)?;
        let head = Head::load(&config, &stream, &checkpoint.join("joint_head.safetensors"))?;
        let vision = Vision::load(&config, &stream, &checkpoint)?;
        let net = Network::new(
            config,
            stream.clone(),
            backbone,
            head,
            vision,
            Composition::GpuAneColumns,
        )?;
        let values = (0..tokens * hidden)
            .map(|index| (index % 251) as f32 / 16.0)
            .collect::<Vec<_>>();
        let h = Array::from_f32(&stream, &[1, tokens, hidden], &values)?.cast(MlxDType::BF16)?;
        let m = &net.backbone.layers[0].mlp;
        // W1 order, as `mlp` now runs it.
        let new = net.mlp(&h, m, 0)?.to_bytes()?;
        // The previous order: read the tiles before any GPU work is queued,
        // then evaluate the GPU share, then predict.
        let c = net.gpu.as_ref().expect("GpuAneColumns has the GPU kernels");
        let ane = net.ane.first().expect("GpuAneColumns has column programs");
        let packed = AneMlp::pack(&h)?;
        let gpu = c.mlp(&h, m)?;
        stream.async_eval(std::slice::from_ref(&gpu))?;
        let share = ane.predict(&stream, &packed, tokens, 0)?;
        let old = gpu.add(&share)?.to_bytes()?;
        assert_eq!(new, old, "W1 changed the column-split MLP's bytes");
        Ok(())
    }
}
