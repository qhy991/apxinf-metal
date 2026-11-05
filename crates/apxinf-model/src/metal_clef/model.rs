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
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

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

/// A1's env gate (`APXINF_CLEF_FUSE_KV=1`): concatenate each attention
/// layer's k and v into one projection. Read once per process at load.
fn fuse_kv_flag(value: Option<String>) -> bool {
    value.as_deref() == Some("1")
}

fn fuse_kv_enabled() -> bool {
    fuse_kv_flag(std::env::var("APXINF_CLEF_FUSE_KV").ok())
}

/// A2's env gate (`APXINF_CLEF_FUSE_GDN=1`): merge each gated-delta layer's
/// z, b and a projections into one GEMM. Read once per process at load.
fn fuse_gdn_flag(value: Option<String>) -> bool {
    value.as_deref() == Some("1")
}

fn fuse_gdn_enabled() -> bool {
    fuse_gdn_flag(std::env::var("APXINF_CLEF_FUSE_GDN").ok())
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
        let split = dense.split();
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
    Ok(Mlp {
        gate: empty_q4(stream)?,
        up: empty_q4(stream)?,
        down: empty_q4(stream)?,
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
    /// A1 (`APXINF_CLEF_FUSE_KV`): attention layers hold a concatenated k+v
    /// projection in `k` and an empty `v` slot; `attention()` issues it once.
    fuse_kv: bool,
    /// A2 (`APXINF_CLEF_FUSE_GDN`): gated-delta layers hold a concatenated
    /// z+b+a projection in `z` and empty `b`/`a` slots; `gated_delta()` issues
    /// it once. On GPU compositions the concatenated width is padded to a
    /// multiple of 128 for the split GEMM; the padded columns are discarded.
    fuse_gdn: bool,
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

/// An affine-Q4 slot whose weights moved into a fused projection: empty
/// arrays, so the checkpoint's originals are released (the same idiom as
/// [`moved_mlp`]). The fused slot keeps its own reference.
fn empty_q4(stream: &Stream) -> Result<Q4> {
    Ok(Q4 {
        weight: Array::zeros(stream, &[0, 0], MlxDType::U32)?,
        scales: Array::zeros(stream, &[0, 0], MlxDType::BF16)?,
        biases: Array::zeros(stream, &[0, 0], MlxDType::BF16)?,
    })
}

/// Concatenate affine-Q4 weights along the **output** axis and materialise
/// them once at load. Scales and biases are per output row (`[N, K/64]`), so
/// no 64-column scale grouping is shared between rows and none changes here.
/// `pad_rows` appends all-zero rows (zero weight, scale and bias dequantise
/// to 0) for kernels whose output width must be a fixed multiple — the GPU
/// composition's split GEMM needs `N % 128 == 0`.
fn concat_q4(stream: &Stream, parts: &[&Q4], pad_rows: usize) -> Result<Q4> {
    let first = parts
        .first()
        .ok_or(Error::Contract("concat_q4 needs at least one weight"))?;
    let [_, words] = *first.weight.shape() else {
        return Err(Error::Contract("concat_q4 weight must be rank 2"));
    };
    let k = words
        .checked_mul(8)
        .ok_or(Error::Contract("concat_q4 width overflows usize"))?;
    for q in parts {
        if q.weight.shape()[1] != words
            || q.weight.dtype() != MlxDType::U32
            || q.scales.shape() != [q.weight.shape()[0], k / 64]
            || q.biases.shape() != [q.weight.shape()[0], k / 64]
            || q.scales.dtype() != MlxDType::BF16
            || q.biases.dtype() != MlxDType::BF16
        {
            return Err(Error::Contract(
                "concat_q4 parts must share one affine-Q4 K and layout",
            ));
        }
    }
    let weights: Vec<&Array> = parts.iter().map(|q| &q.weight).collect();
    let scales: Vec<&Array> = parts.iter().map(|q| &q.scales).collect();
    let biases: Vec<&Array> = parts.iter().map(|q| &q.biases).collect();
    let mut weight = Array::concat(&weights, 0)?;
    let mut scales = Array::concat(&scales, 0)?;
    let mut biases = Array::concat(&biases, 0)?;
    if pad_rows > 0 {
        let zw = Array::zeros(stream, &[pad_rows, words], MlxDType::U32)?;
        let zs = Array::zeros(stream, &[pad_rows, k / 64], MlxDType::BF16)?;
        let zb = Array::zeros(stream, &[pad_rows, k / 64], MlxDType::BF16)?;
        weight = Array::concat(&[&weight, &zw], 0)?;
        scales = Array::concat(&[&scales, &zs], 0)?;
        biases = Array::concat(&[&biases, &zb], 0)?;
    }
    // Evaluate once: without this the concat would be re-enqueued on every
    // request, adding three kernels and erasing the fusion's purpose.
    stream.eval(&[weight.clone(), scales.clone(), biases.clone()])?;
    Ok(Q4 {
        weight,
        scales,
        biases,
    })
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
        // Matmul-level fusion (candidate A), built once at load. A1
        // concatenates each attention layer's k and v into one projection; A2
        // concatenates each gated-delta layer's z, b and a. The originals are
        // released into empty slots, so the fused path adds no device memory
        // beyond A2's padding on GPU compositions. The gate is read once: the
        // baseline and fused arms of a campaign run in separate processes.
        let fuse_kv = fuse_kv_enabled();
        let fuse_gdn = fuse_gdn_enabled();
        if fuse_kv || fuse_gdn {
            let heads = config.text.linear_num_value_heads;
            let z_dim = heads * config.text.linear_value_head_dim;
            let ab_dim = 2 * heads;
            // The GPU composition's split GEMM needs an output width that is
            // a multiple of 128 (two 64-multiple halves); the reference
            // composition's qmm has no such constraint.
            let pad = if fuse_gdn && gpu.is_some() {
                (128 - (z_dim + ab_dim) % 128) % 128
            } else {
                0
            };
            for layer in &mut backbone.layers {
                match &mut layer.mixer {
                    Mixer::Attention(a) if fuse_kv => {
                        let kv = concat_q4(&stream, &[&a.k, &a.v], 0)?;
                        a.k = kv;
                        a.v = empty_q4(&stream)?;
                    }
                    Mixer::GatedDelta(g) if fuse_gdn => {
                        let zba = concat_q4(&stream, &[&g.z, &g.b, &g.a], pad)?;
                        g.z = zba;
                        g.b = empty_q4(&stream)?;
                        g.a = empty_q4(&stream)?;
                    }
                    _ => {}
                }
            }
        }
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
            fuse_kv,
            fuse_gdn,
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
            let cached = cache.and_then(|c| c.layers.get(index));
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
        // A2: with the gate on, `g.z` holds the concatenated z+b+a projection
        // (`g.b` and `g.a` are empty slots). The reference composition ran all
        // three through `qmm`, so one `qmm` covers them; GPU compositions ran
        // z through the wn2 split GEMM, so the concat (padded to a 128-multiple
        // width at load) is decoded in two halves and issued as one split GEMM
        // — the padded columns are computed and discarded.
        let z_dim = value_dim;
        let (z, b, a) = if self.fuse_gdn {
            let out = if let Some(c) = &self.gpu {
                let total = g.z.weight.shape()[0];
                if total == 0 || total % 128 != 0 {
                    return Err(Error::Contract(
                        "fused gated-delta projection must have a 128-multiple width",
                    ));
                }
                let nw = total / 2;
                let h = c.pad.call(x)?;
                // Decode and queue the low half before decoding the high
                // half, the dispatch order `split_linear` pins for MLX.
                let low_w = c
                    .project_decode
                    .call(&g.z.weight, &g.z.scales, &g.z.biases, 0, nw)?;
                let low = c.split.low(&h, tokens, &low_w)?;
                let high_w =
                    c.project_decode
                        .call(&g.z.weight, &g.z.scales, &g.z.biases, nw, nw)?;
                c.split.high(&high_w, &low, &h, tokens)?
            } else {
                qmm(x, &g.z)?
            };
            (
                out.slice_axis(2, 0, z_dim)?.reshape(&[1, tokens, hv, dv])?,
                // b/a are tiny: make the sliced views row-contiguous before
                // the compiled `_compute_g_beta`, whose traced graph assumes
                // packed input rows.
                out.slice_axis(2, z_dim, z_dim + hv)?.contiguous()?,
                out.slice_axis(2, z_dim + hv, z_dim + 2 * hv)?
                    .contiguous()?,
            )
        } else {
            (
                self.split_projection(x, &g.z)?
                    .reshape(&[1, tokens, hv, dv])?,
                qmm(x, &g.b)?,
                qmm(x, &g.a)?,
            )
        };
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
        // A1: with the gate on, `a.k` holds the concatenated k+v projection
        // (`a.v` is an empty slot), so one projection issues and the halves
        // are sliced off. The slices are strided views (row stride
        // 2 * kv_dim); `reshape` keeps them views and the first consumers —
        // row reductions and the strides-aware Metal kernels — accept that.
        let kv_dim = kv * d;
        let (keys, values) = if self.fuse_kv {
            let kv_out = self.single_projection(x, &a.k)?;
            (
                kv_out.slice_axis(2, 0, kv_dim)?,
                kv_out.slice_axis(2, kv_dim, 2 * kv_dim)?,
            )
        } else {
            (
                self.single_projection(x, &a.k)?,
                self.single_projection(x, &a.v)?,
            )
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic affine-Q4 weight with deterministic in-range nibbles and
    /// per-row group-64 scales/biases, so the fusion tests do not need the
    /// checkpoint.
    fn synthetic_q4(stream: &Stream, out: usize, input: usize, seed: u32) -> Result<Q4> {
        let words: Vec<u32> = (0..out * (input / 8))
            .map(|word| {
                let mut packed = 0u32;
                for nibble in 0..8 {
                    let value = ((word * 8 + nibble) as u32).wrapping_mul(17 + seed) % 15;
                    packed |= value << (nibble * 4);
                }
                packed
            })
            .collect();
        let weight_bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
        let weight = Array::from_bytes(stream, &[out, input / 8], MlxDType::U32, &weight_bytes)?;
        let scales: Vec<f32> = (0..out * (input / 64))
            .map(|index| 0.01 + ((index as u32 + seed) % 7) as f32 * 0.005)
            .collect();
        let biases: Vec<f32> = (0..out * (input / 64))
            .map(|index| -0.02 + ((index as u32 + seed * 3) % 5) as f32 * 0.01)
            .collect();
        let scales = Array::from_f32(stream, &[out, input / 64], &scales)?.cast(MlxDType::BF16)?;
        let biases = Array::from_f32(stream, &[out, input / 64], &biases)?.cast(MlxDType::BF16)?;
        Ok(Q4 {
            weight,
            scales,
            biases,
        })
    }

    #[test]
    fn matmul_fusion_gates_need_the_explicit_spelling() {
        for value in [
            None,
            Some(String::new()),
            Some("0".into()),
            Some("true".into()),
        ] {
            assert!(!fuse_kv_flag(value.clone()), "{value:?}");
            assert!(!fuse_gdn_flag(value.clone()), "{value:?}");
        }
        assert!(fuse_kv_flag(Some("1".into())));
        assert!(fuse_gdn_flag(Some("1".into())));
    }

    #[test]
    fn concat_q4_concatenates_rows_and_zero_pads() -> Result<()> {
        let stream = Stream::cpu()?;
        let k = synthetic_q4(&stream, 8, 64, 1)?;
        let v = synthetic_q4(&stream, 8, 64, 5)?;
        let fused = concat_q4(&stream, &[&k, &v], 16)?;
        assert_eq!(fused.weight.shape(), [32, 8]);
        assert_eq!(fused.scales.shape(), [32, 1]);
        assert_eq!(fused.biases.shape(), [32, 1]);
        // The 16 padding rows are all zero: a dequantise of them is 0.
        let bytes = fused.weight.to_bytes()?;
        assert!(bytes[16 * 8 * 4..].iter().all(|&byte| byte == 0));
        assert!(fused.scales.to_f32_vec()?[16..].iter().all(|&x| x == 0.0));
        assert!(fused.biases.to_f32_vec()?[16..].iter().all(|&x| x == 0.0));
        // The originals are unchanged, so the concat is purely additive.
        assert_eq!(k.weight.to_bytes()?[..8 * 8 * 4], bytes[..8 * 8 * 4]);
        Ok(())
    }

    /// Differences between two same-length BF16 buffers: count of differing
    /// elements, largest absolute difference and largest relative difference
    /// (BF16 bit patterns widened to F32). Used instead of printing megabytes
    /// of bytes when a fused projection is compared to the separate one.
    fn bf16_diffs(a: &[u8], b: &[u8]) -> (usize, f32, f32) {
        assert_eq!(a.len(), b.len(), "compared BF16 buffers must match in size");
        let mut count = 0usize;
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
            if x == y {
                continue;
            }
            count += 1;
            let xf = f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16);
            let yf = f32::from_bits((u16::from_le_bytes([y[0], y[1]]) as u32) << 16);
            let delta = (xf - yf).abs();
            max_abs = max_abs.max(delta);
            max_rel = max_rel.max(delta / xf.abs().max(yf.abs()).max(1e-6));
        }
        (count, max_abs, max_rel)
    }

    /// Gather `width` consecutive BF16 columns of every row from a row-major
    /// `[rows, total]` byte buffer, so a column slice of a fused `[1, T, N]`
    /// output can be compared with a standalone `[1, T, width]` projection.
    fn gather_columns(
        bytes: &[u8],
        rows: usize,
        total: usize,
        start: usize,
        width: usize,
    ) -> Vec<u8> {
        assert!(start + width <= total, "column range exceeds the row");
        let mut out = Vec::with_capacity(rows * width * 2);
        let row_bytes = total * 2;
        for row in 0..rows {
            out.extend_from_slice(
                &bytes[row * row_bytes + start * 2..row * row_bytes + (start + width) * 2],
            );
        }
        out
    }

    /// Largest absolute BF16 value in a byte buffer, used to scale an ulp
    /// bound when comparing projections whose accumulation order differs.
    fn bf16_scale(bytes: &[u8]) -> f32 {
        bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).abs())
            .fold(0.0f32, f32::max)
    }

    /// A1 and A2 at the `qmm` level on the real device. `quantized_matmul` is
    /// deterministic for identical inputs (the Tier-2 A/A established this
    /// across processes for the reference composition), but its **shape**
    /// changes the GEMM tiling and with it the FP32 accumulation order — the
    /// fused `[.., 2048]` projection is therefore *not* bit-identical to two
    /// separate `[.., 1024]` projections, it differs in the last BF16 bit
    /// (measured: ~68 % of elements flip, max absolute gap within one BF16
    /// ulp of the buffer scale; relative error is only meaningful away from
    /// zero). This contradicts the "A1 expected bit-identical" hypothesis, so
    /// the campaign must run A1 through the same 42-case parity gate as A2.
    /// This test pins the measured relationship: hard determinism, and an
    /// absolute ulp bound on the fused-vs-separate gap.
    #[test]
    fn fused_projection_slices_track_the_separate_path() -> Result<()> {
        let stream = Stream::metal(0)?;
        let tokens = 17usize;
        let input = 4096usize;
        let hidden: Vec<f32> = (0..tokens * input)
            .map(|index| ((index * 7) % 251) as f32 / 251.0 - 0.5)
            .collect();
        let x = Array::from_f32(&stream, &[1, tokens, input], &hidden)?.cast(MlxDType::BF16)?;

        // A1: attention k and v, 1024 rows each (the Clef geometry).
        let k = synthetic_q4(&stream, 1024, input, 3)?;
        let v = synthetic_q4(&stream, 1024, input, 9)?;
        let k_bytes = qmm(&x, &k)?.to_bytes()?;
        // Determinism: two identical projections must be byte-equal (this is
        // what makes the campaign's cross-process A/A meaningful).
        let (det, _, _) = bf16_diffs(&k_bytes, &qmm(&x, &k)?.to_bytes()?);
        assert_eq!(det, 0, "quantized_matmul must be deterministic");

        let kv = concat_q4(&stream, &[&k, &v], 0)?;
        let out = qmm(&x, &kv)?;
        let fused = out.to_bytes()?;
        assert_eq!(fused.len(), 2 * k_bytes.len());
        let v_bytes = qmm(&x, &v)?.to_bytes()?;
        // The fused output is row-major `[1, T, 2048]`; take each row's
        // first/second 1024 columns (a flat prefix would cut rows in half).
        let fused_k = gather_columns(&fused, tokens, 2048, 0, 1024);
        let fused_v = gather_columns(&fused, tokens, 2048, 1024, 1024);
        let (dk, abs_k, rel_k) = bf16_diffs(&fused_k, &k_bytes);
        let (dv, abs_v, rel_v) = bf16_diffs(&fused_v, &v_bytes);
        println!(
            "A1 fused-vs-separate: k {dk}/{} elems differ max|d|={abs_k:.3e} (scale {:.3e}) rel={rel_k:.3e}; \
             v {dv}/{} elems differ max|d|={abs_v:.3e} (scale {:.3e}) rel={rel_v:.3e}",
            k_bytes.len() / 2,
            bf16_scale(&k_bytes),
            v_bytes.len() / 2,
            bf16_scale(&v_bytes),
        );
        // The fused projection stays within a couple of BF16 spacings of the
        // separate one (~68 % of elements flip in the last bits); relative
        // error near zero is meaningless, so the bound is absolute and scaled
        // by the buffer magnitude. A wiring/scale bug would be far larger.
        let ulp_k = bf16_scale(&k_bytes) * 2f32.powi(-5);
        let ulp_v = bf16_scale(&v_bytes) * 2f32.powi(-5);
        assert!(
            abs_k <= ulp_k && abs_v <= ulp_v,
            "fused k/v must stay within ~4 BF16 spacings: {abs_k:.3e} vs {ulp_k:.3e}, {abs_v:.3e} vs {ulp_v:.3e}"
        );

        // The k half also has to survive the layout chain the attention path
        // uses: strided slice -> reshape -> row reduction.
        let norm: Vec<f32> = (0..256)
            .map(|index| 0.5 + (index % 17) as f32 / 34.0)
            .collect();
        let norm = Array::from_f32(&stream, &[256], &norm)?.cast(MlxDType::BF16)?;
        let fused_norm = out
            .slice_axis(2, 0, 1024)?
            .reshape(&[1, tokens, 4, 256])?
            .fast_rms_norm(&norm, EPS)?
            .to_bytes()?;
        let separate_norm = qmm(&x, &k)?
            .reshape(&[1, tokens, 4, 256])?
            .fast_rms_norm(&norm, EPS)?
            .to_bytes()?;
        let (dn, abs_n, rel_n) = bf16_diffs(&fused_norm, &separate_norm);
        println!(
            "A1 rms_norm through the strided slice: {dn}/{} elems differ max|d|={abs_n:.3e} (scale {:.3e}) rel={rel_n:.3e}",
            separate_norm.len() / 2,
            bf16_scale(&separate_norm),
        );
        assert!(
            abs_n <= bf16_scale(&separate_norm) * 2f32.powi(-5),
            "rms_norm over the fused slice must stay within ~4 BF16 spacings"
        );

        // A2: gated-delta z (4096 rows) + b and a (32 rows each). The concat
        // is padded by 64 rows so a GPU-composition split GEMM (width % 128)
        // could consume it; the slices ignore the padding.
        let z = synthetic_q4(&stream, 4096, input, 1)?;
        let b = synthetic_q4(&stream, 32, input, 4)?;
        let a = synthetic_q4(&stream, 32, input, 8)?;
        let zba = concat_q4(&stream, &[&z, &b, &a], 64)?;
        assert_eq!(zba.weight.shape()[0], 4224);
        let out = qmm(&x, &zba)?.to_bytes()?;
        let z_bytes = qmm(&x, &z)?.to_bytes()?;
        let b_bytes = qmm(&x, &b)?.to_bytes()?;
        let a_bytes = qmm(&x, &a)?.to_bytes()?;
        let fused_z = gather_columns(&out, tokens, 4224, 0, 4096);
        let fused_b = gather_columns(&out, tokens, 4224, 4096, 32);
        let fused_a = gather_columns(&out, tokens, 4224, 4128, 32);
        let (dz, abs_z, rel_z) = bf16_diffs(&fused_z, &z_bytes);
        let (db, abs_b, rel_b) = bf16_diffs(&fused_b, &b_bytes);
        let (da, abs_a, rel_a) = bf16_diffs(&fused_a, &a_bytes);
        println!(
            "A2 fused-vs-separate: z {dz}/{} max|d|={abs_z:.3e} (scale {:.3e}) rel={rel_z:.3e}; \
             b {db}/{} max|d|={abs_b:.3e} (scale {:.3e}) rel={rel_b:.3e}; \
             a {da}/{} max|d|={abs_a:.3e} (scale {:.3e}) rel={rel_a:.3e}",
            z_bytes.len() / 2,
            bf16_scale(&z_bytes),
            b_bytes.len() / 2,
            bf16_scale(&b_bytes),
            a_bytes.len() / 2,
            bf16_scale(&a_bytes),
        );
        let ulp_z = bf16_scale(&z_bytes) * 2f32.powi(-5);
        let ulp_b = bf16_scale(&b_bytes) * 2f32.powi(-5);
        let ulp_a = bf16_scale(&a_bytes) * 2f32.powi(-5);
        assert!(
            abs_z <= ulp_z && abs_b <= ulp_b && abs_a <= ulp_a,
            "fused z/b/a must stay within ~4 BF16 spacings: {abs_z:.3e}/{ulp_z:.3e}, \
             {abs_b:.3e}/{ulp_b:.3e}, {abs_a:.3e}/{ulp_a:.3e}"
        );
        Ok(())
    }

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
