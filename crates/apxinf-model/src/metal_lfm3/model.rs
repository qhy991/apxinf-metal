//! Native LFM3 forward and request-owned state (prefill, decode and KV/conv caches).
//!
//! The execution recipe is a line-by-line reproduction of the verified reference
//! path, because that path is the acceptance oracle: sandwiched BF16 weighted
//! RMSNorm, explicit FP32 zero-tail rotary tables, FP32 depthwise convolution
//! with a six-column input history, per-head sigmoid output gate, and a dense KV
//! row for full-attention layers with a 256-token read window for sliding ones.
//! The one deliberate exception is the single-token convolve: its seven-tap
//! window is convolved by one fused causal-depthwise kernel (the same primitive
//! `metal_clef` and `metal_d3_lite` use), and the reference's per-tap host-side
//! accumulation is kept only as the test-only parity reference. Every other
//! operation names the backend primitive the reference itself calls.

use super::{
    config::{LayerKind, Lfm3Config},
    kkernel::KKernel,
    weights::{LayerWeights, Mixer, Weights},
};
use crate::llm_trait::{
    LlmInput, LlmTrait, TextPreparationState, TextPreparationStatus, TokenSequence,
};
use crate::mlx_decode::{DecodeFinish, DeviceDecode, GreedySequence};
use apxinf_core::{Backend, Device, Error, Result, Tensor, TokenSampler, TokenSamplingParams};
use apxinf_loader::ModelConfig;
use apxinf_mlx::{fusions::CausalDepthwiseConv, Array, MlxBackend, MlxDType, Stream};
use std::{collections::HashMap, sync::Arc, time::Instant};

fn err(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

/// `APXINF_LFM3_DECODE_STAGES=1` splits every one-token step into host graph
/// construction and device evaluation on stderr. Reaches the worker through
/// the gateway's `--worker-env` forwarding (the supervisor clears the
/// environment when it spawns the worker).
fn decode_stages_enabled() -> bool {
    std::env::var_os("APXINF_LFM3_DECODE_STAGES").is_some_and(|value| value == "1")
}

/// `APXINF_LFM3_PREFILL_FAST=1` reuses the last prefill graph for a repeated
/// static shape instead of rebuilding it per prompt (QUEUE #3,
/// `doc/lfm3-short-prompt-prefill.md`). Same env source the worker banner
/// reports, and the model captures it once at construction, so the unset
/// baseline arm takes the unchanged path (empty-arm gate).
pub fn prefill_fast_enabled() -> bool {
    std::env::var_os("APXINF_LFM3_PREFILL_FAST").is_some_and(|value| value == "1")
}

/// `APXINF_LFM3_DECODE_ASYNC_EVAL=1` replaces the single-token step's blocking
/// `Stream::eval` with `async_eval` + `synchronize` (QUEUE #4,
/// `doc/lfm3-decode-eval-sync-split.md`): the same required arrays, stream and
/// graph, only the host wait boundary moves. Same env source the worker banner
/// reports; the model captures it once at construction so the unset arm keeps
/// the blocking `eval` unchanged (empty-arm gate).
pub fn decode_async_eval_enabled() -> bool {
    std::env::var_os("APXINF_LFM3_DECODE_ASYNC_EVAL").is_some_and(|value| value == "1")
}

// Platform C math library, used only to generate the fixed rotary tables on the
// host. Declared here instead of relying on Rust's `f32::sin`/`f32::cos`, which
// are correctly rounded and therefore not bit-identical to the reference lane's
// float32 numpy path (see `rope_table`).
extern "C" {
    #[link_name = "cosf"]
    fn libm_cosf(x: f32) -> f32;
    #[link_name = "sinf"]
    fn libm_sinf(x: f32) -> f32;
}

/// Weighted RMSNorm over the last axis on BF16 activations.
fn norm(x: &Array, weight: &Array, epsilon: f32) -> Result<Array> {
    if weight.dtype() != MlxDType::BF16 || x.dtype() != MlxDType::BF16 {
        return Err(err("LFM3 RMSNorm requires BF16 activations and weights"));
    }
    x.fast_rms_norm(weight, epsilon)
}

/// HF `rotate_half` over the last axis: `[x1, x2] -> [-x2, x1]`.
fn rotate_half(x: &Array, dim: usize) -> Result<Array> {
    let half = dim / 2;
    Array::concat(
        &[
            &x.slice_axis(3, half, dim)?.neg()?,
            &x.slice_axis(3, 0, half)?,
        ],
        3,
    )
}

/// Reference rotary application: FP32 arithmetic against the precomputed
/// zero-tail tables, then a single cast back to the activation dtype.
fn apply_rope(x: &Array, cos: &Array, sin: &Array, dim: usize) -> Result<Array> {
    let xf = x.cast(MlxDType::F32)?;
    xf.mul(cos)?
        .add(&rotate_half(&xf, dim)?.mul(sin)?)?
        .cast(x.dtype())
}

/// `silu(gate) * up` expressed exactly as the reference's `nn.silu(g) * u`,
/// dispatched through the armed custom kernel when the switch is on.
fn silu_mul(gate: &Array, up: &Array, kkernel: &KKernel) -> Result<Array> {
    kkernel.silu_mul(gate, up)
}

/// Fused one-token depthwise convolution over a seven-column window.
///
/// `window` is `[1, kernel, hidden]` BF16 in tap order (oldest first) and
/// `taps_bf16` is `[hidden, kernel]` BF16. One kernel launch writes every row;
/// row `kernel - 1` is the current token's convolution (already in the
/// `[1, 1, hidden]` layout the gate multiplies it in) and rows `0..kernel - 1`
/// are the six columns the next step reads. Returns
/// `([1, 1, hidden], [1, hidden, kernel - 1])`.
fn fused_conv_window(
    conv: &CausalDepthwiseConv,
    window: &Array,
    taps_bf16: &Array,
) -> Result<(Array, Array)> {
    let [1, kernel, hidden] = *window.shape() else {
        return Err(err("LFM3 conv window must be [1, kernel, hidden]"));
    };
    let out = conv.call(window, &taps_bf16.reshape(&[hidden, kernel, 1])?)?;
    // The current row is already `[1, 1, hidden]`; the state is kept in the
    // family's `[1, hidden, kernel - 1]` layout.
    let current = out.slice_axis(1, kernel - 1, kernel)?;
    let next = window.slice_axis(1, 1, kernel)?.transpose(&[0, 2, 1])?;
    Ok((current, next))
}

/// Pre-migration one-token convolution arithmetic: seven host-side tap
/// accumulations, kept only as the test reference for `fused_conv_window`.
///
/// The decode path never calls it. Its operands are the FP32 tap table
/// widened from the checkpoint's BF16 bytes, so the residual against the
/// fused path is the accumulation form, not a precision class.
#[cfg(test)]
fn legacy_window_conv(window: &Array, taps_f32: &Array) -> Result<(Array, Array)> {
    let [1, kernel, hidden] = *window.shape() else {
        return Err(err("LFM3 conv window must be [1, kernel, hidden]"));
    };
    let legacy_window = window.transpose(&[0, 2, 1])?;
    // Every operand of the accumulation is `[1, hidden, 1]`; adding a
    // `[1, 1, hidden]` accumulator to it would broadcast into `[1, h, h]`.
    let mut conv = Array::zeros(window.stream(), &[1, hidden, 1], MlxDType::F32)?;
    for tap in 0..kernel {
        let weight = taps_f32
            .slice_axis(1, tap, tap + 1)?
            .reshape(&[1, hidden, 1])?;
        conv = conv.add(&weight.mul(&legacy_window.slice_axis(2, tap, tap + 1)?)?)?;
    }
    let conv = conv.transpose(&[0, 2, 1])?.cast(MlxDType::BF16)?;
    Ok((
        conv,
        window.slice_axis(1, 1, kernel)?.transpose(&[0, 2, 1])?,
    ))
}

#[derive(Clone)]
struct KvState {
    keys: Array,
    values: Array,
}

/// Per-layer state a completed layer returns to the request.
enum LayerState {
    Attention(KvState),
    Conv(Array),
}

/// QUEUE #3 (doc §Decision): the last prefill graph and the static shape it
/// was traced for. Holding the graph plus its initial states skips the host
/// re-trace of the 24 layers when the next prompt has the same shape.
/// `len` is the token count, `start` the absolute position (0 for the whole
/// prompt), and `continuing` whether conv windows are concatenated.
struct PrefillCache {
    key: (usize, usize, bool),
    logits: Array,
    states: Vec<LayerState>,
}

pub struct Lfm3Model {
    config: Lfm3Config,
    backend: MlxBackend,
    weights: Weights,
    /// Compiled `apxinf_causal_depthwise_conv` pipeline for the decode step.
    causal_conv: CausalDepthwiseConv,
    kinds: Vec<LayerKind>,
    /// One dense row per attention layer, `[1, kv_heads, capacity, head_dim]`.
    initial_kv: Vec<Option<KvState>>,
    kv: Vec<Option<KvState>>,
    /// Conv input history `[1, hidden, conv_L_cache - 1]` per conv layer.
    initial_conv: Vec<Option<Array>>,
    conv: Vec<Option<Array>>,
    cos_full: Option<Array>,
    sin_full: Option<Array>,
    cos_sliding: Option<Array>,
    sin_sliding: Option<Array>,
    capacity: usize,
    position: usize,
    invalid: Option<String>,
    /// Per-step conv-window snapshot for the pipelined greedy path: the
    /// lookahead step shifts the 13 sliding `[1, hidden, kernel-1]` windows,
    /// and `finish_decode` must be able to move them back. MLX arrays are
    /// immutable, so keeping the pre-step references is the rollback.
    conv_snapshot: Option<Vec<Option<Array>>>,
    stages: bool,
    stage_seq: usize,
    /// QUEUE #3 prefill fast arm, captured once at construction from the same
    /// env the banner reports (`prefill_fast_enabled`). When unset the model
    /// never builds or touches a cached prefill graph (empty-arm gate).
    prefill_fast: bool,
    /// Last built prefill graph plus the static shape it was traced for
    /// (`(len, start, continuing)`). Kept only while the fast arm is on and
    /// only for the multi-token path; `None` means no reuse this request.
    prefill_cache: Option<PrefillCache>,
    /// Test-only fault injection for T3: skip the conv restore in
    /// `finish_decode` and prove the byte-level state check catches it.
    #[cfg(test)]
    disable_conv_restore: bool,
    /// Decode-step custom kernels (QUEUE #2; inert unless `APXINF_LFM3_KKERNEL=1`).
    kkernel: KKernel,
    /// QUEUE #4 probe: move the single-token step's host wait from inside
    /// `Stream::eval` to an explicit `synchronize` after `async_eval`. Captured
    /// once at construction from the same env the banner reports; when unset the
    /// step keeps the blocking `eval` unchanged (empty-arm gate).
    decode_async_eval: bool,
    /// QUEUE #4 E4 test seam: records the armed split's call order
    /// (`"async_eval"` then `"synchronize"`) so the negative control can prove
    /// the branch ran and ran in the right order. Empty on the baseline arm.
    /// Test-only: no production state or cost.
    #[cfg(test)]
    test_split_trace: Vec<&'static str>,
}

impl Lfm3Model {
    pub fn new(
        config: Lfm3Config,
        tensors: HashMap<String, Tensor>,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        config.validate()?;
        config.validate_checkpoint_scope()?;
        let backend = backend
            .as_any()
            .downcast_ref::<MlxBackend>()
            .ok_or_else(|| err("LFM3 native implementation requires an MLX backend"))?
            .clone();
        let weights = Weights::load(&config, tensors, backend.stream())?;
        let causal_conv = CausalDepthwiseConv::new(backend.stream())?;
        let kkernel = KKernel::new(backend.stream())?;
        let kinds = config.layer_kinds()?;
        let layers = config.num_hidden_layers;
        let mut model = Self {
            config,
            backend,
            weights,
            causal_conv,
            kinds,
            initial_kv: vec![None; layers],
            kv: vec![None; layers],
            initial_conv: vec![None; layers],
            conv: vec![None; layers],
            cos_full: None,
            sin_full: None,
            cos_sliding: None,
            sin_sliding: None,
            capacity: 0,
            position: 0,
            invalid: None,
            conv_snapshot: None,
            stages: decode_stages_enabled(),
            stage_seq: 0,
            prefill_fast: prefill_fast_enabled(),
            prefill_cache: None,
            kkernel,
            decode_async_eval: decode_async_eval_enabled(),
            #[cfg(test)]
            disable_conv_restore: false,
            #[cfg(test)]
            test_split_trace: Vec::new(),
        };
        model.prepare(1, 0)?;
        Ok(model)
    }

    fn report_stage(&mut self, path: &str, build_ms: f64, eval_ms: f64) {
        self.stage_seq += 1;
        eprintln!(
            "{{\"kind\":\"decode_stage\",\"impl\":\"metal_lfm3\",\"path\":\"{path}\",\"seq\":{},\"build_ms\":{build_ms:.4},\"eval_ms\":{eval_ms:.4}}}",
            self.stage_seq,
        );
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Prepare the total position capacity for a prompt plus its output budget.
    /// Existing capacity is retained when it already covers the request.
    pub fn prepare(&mut self, prompt_len: usize, max_new_tokens: usize) -> Result<()> {
        if let Err(error) = self.ensure_capacity(prompt_len, max_new_tokens) {
            self.invalid = Some(error.to_string());
            return Err(error);
        }
        Ok(())
    }

    fn ensure_capacity(&mut self, prompt_len: usize, max_new_tokens: usize) -> Result<()> {
        if let Some(reason) = &self.invalid {
            return Err(err(format!(
                "LFM3 session invalidated: {reason}; reset required"
            )));
        }
        let required = self.config.request_capacity(prompt_len, max_new_tokens)?;
        if required <= self.capacity && !self.kv.is_empty() {
            return Ok(());
        }
        if self.position != 0 {
            return Err(err(
                "reset LFM3 before growing its prepared position capacity",
            ));
        }
        self.allocate(required)
    }

    /// Allocate KV rows, conv history and both rotary tables for `capacity`.
    fn allocate(&mut self, capacity: usize) -> Result<()> {
        let stream = self.backend.stream().clone();
        let c = &self.config;
        let row = [1, c.num_key_value_heads, capacity, c.head_dim];
        let zero_kv = Array::zeros(&stream, &row, MlxDType::BF16)?;
        let keep = c.conv_state_len();
        let zero_conv = Array::zeros(&stream, &[1, c.hidden_size, keep], MlxDType::BF16)?;
        let mut kv = Vec::with_capacity(c.num_hidden_layers);
        let mut conv = Vec::with_capacity(c.num_hidden_layers);
        for kind in &self.kinds {
            if kind.is_conv() {
                kv.push(None);
                conv.push(Some(zero_conv.clone()));
            } else {
                kv.push(Some(KvState {
                    keys: zero_kv.clone(),
                    values: zero_kv.clone(),
                }));
                conv.push(None);
            }
        }
        let cos_full = Self::rope_table(&stream, c, LayerKind::FullAttention, capacity, true)?;
        let sin_full = Self::rope_table(&stream, c, LayerKind::FullAttention, capacity, false)?;
        let cos_sliding =
            Self::rope_table(&stream, c, LayerKind::SlidingAttention, capacity, true)?;
        let sin_sliding =
            Self::rope_table(&stream, c, LayerKind::SlidingAttention, capacity, false)?;
        stream.eval(&[
            zero_kv,
            zero_conv,
            cos_full.clone(),
            sin_full.clone(),
            cos_sliding.clone(),
            sin_sliding.clone(),
        ])?;
        self.initial_kv = kv.clone();
        self.kv = kv;
        self.initial_conv = conv.clone();
        self.conv = conv;
        self.cos_full = Some(cos_full);
        self.sin_full = Some(sin_full);
        self.cos_sliding = Some(cos_sliding);
        self.sin_sliding = Some(sin_sliding);
        self.capacity = capacity;
        Ok(())
    }

    /// `cos`/`sin` tables `[capacity, head_dim]` in FP32.
    ///
    /// The angle table is `concat(freqs, freqs)` like the reference, so the
    /// zero-tail frequencies arrive as `cos=1, sin=0` and the same `rotate()`
    /// formula is exact for both rotary schemes.
    fn rope_table(
        stream: &Stream,
        config: &Lfm3Config,
        kind: LayerKind,
        capacity: usize,
        cosine: bool,
    ) -> Result<Array> {
        let scheme = config.rope(kind)?;
        let dim = config.head_dim;
        let half = dim / 2;
        let inv_freq = scheme.inv_freq(dim);
        // The table is built on the host in FP32 with the platform `sinf`/`cosf`.
        // That is what the reference's `np.sin`/`np.cos` on a float32 angle table
        // compiles to on this platform, and matching it bit-for-bit matters:
        // Rust's `f32::sin`/`f32::cos` are correctly rounded and differ from the
        // platform libm by one ULP on ~5-12% of entries, which is enough to flip
        // a BF16 rounding of `q`/`k` and lose parity in the first sliding layer
        // (measured: 1014/1024 argmax agreement with a correctly rounded table,
        // 1024/1024 with these tables). The device transcendental kernel differs
        // the same way, so the table is not built from MLX ops either.
        let mut table = vec![0f32; capacity * dim];
        for position in 0..capacity {
            let base = position * dim;
            let angle_base = position as f32;
            for j in 0..half {
                let angle = angle_base * inv_freq[j];
                let value = if cosine {
                    unsafe { libm_cosf(angle) }
                } else {
                    unsafe { libm_sinf(angle) }
                };
                table[base + j] = value;
                table[base + half + j] = value;
            }
        }
        Array::from_f32(stream, &[capacity, dim], &table)
    }

    fn cos_sin(&self, kind: LayerKind, start: usize, len: usize) -> Result<(Array, Array)> {
        let (cos, sin) = match kind {
            LayerKind::FullAttention => (self.cos_full.as_ref(), self.sin_full.as_ref()),
            LayerKind::SlidingAttention => (self.cos_sliding.as_ref(), self.sin_sliding.as_ref()),
            LayerKind::Conv => return Err(err("LFM3 conv layers have no rotary embedding")),
        };
        let cos = cos.ok_or_else(|| err("LFM3 rotary tables are not prepared"))?;
        let sin = sin.ok_or_else(|| err("LFM3 rotary tables are not prepared"))?;
        let dim = self.config.head_dim;
        Ok((
            cos.slice_axis(0, start, start + len)?
                .reshape(&[1, 1, len, dim])?,
            sin.slice_axis(0, start, start + len)?
                .reshape(&[1, 1, len, dim])?,
        ))
    }

    /// Boolean attention mask `[1, 1, q_len, k_len]` over absolute positions.
    fn band_mask(
        &self,
        q_start: usize,
        q_len: usize,
        k_start: usize,
        k_len: usize,
        window: Option<usize>,
    ) -> Result<Array> {
        let stream = self.backend.stream();
        let q_pos = Array::arange(
            stream,
            q_start as f32,
            (q_start + q_len) as f32,
            1.,
            MlxDType::I32,
        )?
        .reshape(&[q_len, 1])?;
        let k_pos = Array::arange(
            stream,
            k_start as f32,
            (k_start + k_len) as f32,
            1.,
            MlxDType::I32,
        )?
        .reshape(&[1, k_len])?;
        let causal = k_pos.less_equal(&q_pos)?;
        let mask = match window {
            None => causal,
            Some(window) => {
                // `q_pos - k_pos < window` as an integer comparison.
                let distance = q_pos.sub(&k_pos)?;
                let inside = distance.less_equal(&Array::scalar(
                    stream,
                    (window - 1) as f32,
                    MlxDType::I32,
                )?)?;
                causal
                    .cast(MlxDType::I32)?
                    .mul(&inside.cast(MlxDType::I32)?)?
                    .cast(MlxDType::Bool)?
            }
        };
        mask.reshape(&[1, 1, q_len, k_len])
    }

    /// Short-conv branch. `continuing` selects the stateful window continuation
    /// the reference uses for a chunk that is not the prompt's first.
    fn conv_layer(
        &self,
        n: &Array,
        layer: &LayerWeights,
        state: &Array,
        continuing: bool,
    ) -> Result<(Array, Array)> {
        let Mixer::Conv { in_proj, out_proj, taps, .. } = &layer.mixer
        else {
            return Err(err("LFM3 conv layer has an attention weight plan"));
        };
        let c = &self.config;
        let h = c.hidden_size;
        let taps_kernel = c.conv_l_cache;
        let keep = c.conv_state_len();
        let len = n.shape()[1];
        let bcx = in_proj.call(n)?;
        let b = bcx.slice_axis(2, 0, h)?;
        let gate = bcx.slice_axis(2, h, 2 * h)?;
        let z = bcx.slice_axis(2, 2 * h, 3 * h)?;
        let z = b.mul(&z)?;
        let z_stream = if continuing {
            Array::concat(&[&state.transpose(&[0, 2, 1])?, &z], 1)?
        } else {
            z
        };
        let stream_len = z_stream.shape()[1];
        if stream_len < len {
            return Err(err("LFM3 conv input shorter than its chunk"));
        }
        let weight = taps.reshape(&[h, taps_kernel, 1])?;
        let conv = z_stream.cast(MlxDType::F32)?.conv1d(&weight, 1, taps_kernel - 1, 1, h)?;
        let y = gate.mul(&conv.slice_axis(1, stream_len - len, stream_len)?.cast(MlxDType::BF16)?)?;
        let op = out_proj.call(&y)?;
        let first = stream_len.saturating_sub(keep);
        let mut next = z_stream.slice_axis(1, first, stream_len)?.transpose(&[0, 2, 1])?;
        let width = next.shape()[2];
        if width < keep {
            next = next.pad(&[0, 0, keep - width], &[0, 0, 0])?;
        }
        Ok((op, next))
    }

    /// One-token short-conv step. The six retained input columns plus the new
    /// column are one `[1, kernel, hidden]` window, and one fused
    /// causal-depthwise kernel writes all of its rows; the last row is this
    /// token's convolution and the leading `kernel - 1` rows are the next state.
    ///
    /// The pre-migration implementation accumulated the taps one at a time on
    /// the host graph (7 slices + 7 multiplies + 6 adds per conv layer). It was
    /// replaced by the single kernel launch; see `legacy_window_conv` for the
    /// preserved reference and `crates/apxinf-mlx/native/causal_depthwise_conv.metal`
    /// for the kernel's contract.
    fn conv_step(&self, n: &Array, layer: &LayerWeights, state: &Array) -> Result<(Array, Array)> {
        let Mixer::Conv {
            in_proj,
            out_proj,
            taps_bf16,
            ..
        } = &layer.mixer
        else {
            return Err(err("LFM3 conv layer has an attention weight plan"));
        };
        let bcx = in_proj.call(n)?;
        let h = n.shape()[2];
        let b = bcx.slice_axis(2, 0, h)?;
        let gate = bcx.slice_axis(2, h, 2 * h)?;
        let z = bcx.slice_axis(2, 2 * h, 3 * h)?;
        let z = b.mul(&z)?;
        // Build the window in the kernel's `[1, L, C]` layout instead of
        // concatenating `[1, hidden, kernel]` and transposing the copy.
        let window = Array::concat(&[&state.transpose(&[0, 2, 1])?, &z], 1)?;
        let (conv, next) = fused_conv_window(&self.causal_conv, &window, taps_bf16)?;
        let y = gate.mul(&conv)?;
        let op = out_proj.call(&y)?;
        Ok((op, next))
    }

    /// Attention branch for a prompt chunk (`len >= 1`, `continuing` when the
    /// position counter is already advanced).
    #[allow(clippy::too_many_arguments)]
    fn attention_layer(
        &self,
        n: &Array,
        layer: &LayerWeights,
        kind: LayerKind,
        kv: &KvState,
        start: usize,
        len: usize,
    ) -> Result<(Array, KvState)> {
        let Mixer::Attention {
            q: q_proj,
            k: k_proj,
            v: v_proj,
            o: o_proj,
            q_norm,
            k_norm,
            gate,
        } = &layer.mixer
        else {
            return Err(err("LFM3 attention layer has a conv weight plan"));
        };
        let c = &self.config;
        let heads = c.num_attention_heads;
        let dim = c.head_dim;
        let scale = (dim as f64).powf(-0.5) as f32;
        let continuing = start > 0;
        let q = q_proj.call(n)?.reshape(&[1, len, heads, dim])?;
        let q = norm(&q, q_norm, c.norm_eps)?.transpose(&[0, 2, 1, 3])?;
        let k = k_proj
            .call(n)?
            .reshape(&[1, len, c.num_key_value_heads, dim])?;
        let k = norm(&k, k_norm, c.norm_eps)?.transpose(&[0, 2, 1, 3])?;
        let v = v_proj
            .call(n)?
            .reshape(&[1, len, c.num_key_value_heads, dim])?
            .transpose(&[0, 2, 1, 3])?;
        let (cos, sin) = self.cos_sin(kind, start, len)?;
        let q = apply_rope(&q, &cos, &sin, dim)?;
        let k = apply_rope(&k, &cos, &sin, dim)?;
        let index = Array::from_i32(self.backend.stream(), &[1], &[start as i32])?;
        let keys = kv.keys.slice_update(&k, &index, &[2])?;
        let values = kv.values.slice_update(&v, &index, &[2])?;
        let end = start + len;
        let window = match kind {
            LayerKind::SlidingAttention => Some(c.window()),
            _ => None,
        };
        let attention = if continuing {
            // A continued chunk needs the absolute-position mask; sliding caches
            // stay dense here, so the window is expressed by the mask.
            let mask = self.band_mask(start, len, 0, end, window)?;
            q.sdpa(
                &keys.slice_axis(2, 0, end)?,
                &values.slice_axis(2, 0, end)?,
                scale,
                false,
                Some(&mask),
            )?
        } else if let Some(window) = window {
            if len <= window {
                q.sdpa(
                    &keys.slice_axis(2, 0, end)?,
                    &values.slice_axis(2, 0, end)?,
                    scale,
                    len > 1,
                    None,
                )?
            } else {
                self.windowed_sdpa(&q, &keys, &values, len, window, scale)?
            }
        } else {
            q.sdpa(
                &keys.slice_axis(2, 0, end)?,
                &values.slice_axis(2, 0, end)?,
                scale,
                len > 1,
                None,
            )?
        };
        // Per-head sigmoid output gate: `[1, len, heads] -> [1, heads, len, 1]`.
        let gate = gate.call(n)?.sigmoid()?.transpose(&[0, 2, 1])?;
        let gated = attention.mul(&gate.reshape(&[1, heads, len, 1])?)?;
        let merged = gated
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, len, heads * dim])?;
        let op = o_proj.call(&merged)?;
        Ok((op, KvState { keys, values }))
    }

    /// Exact sliding-window attention over query blocks, mirroring the
    /// reference's decomposition so no dense `[T, T]` mask is materialised.
    fn windowed_sdpa(
        &self,
        q: &Array,
        keys: &Array,
        values: &Array,
        len: usize,
        window: usize,
        scale: f32,
    ) -> Result<Array> {
        let mut blocks = Vec::with_capacity(len.div_ceil(window));
        let mut start = 0;
        while start < len {
            let end = (start + window).min(len);
            let low = start.saturating_sub(window - 1);
            let mask = self.band_mask(start, end - start, low, end - low, Some(window))?;
            blocks.push(q.slice_axis(2, start, end)?.sdpa(
                &keys.slice_axis(2, low, end)?,
                &values.slice_axis(2, low, end)?,
                scale,
                false,
                Some(&mask),
            )?);
            start = end;
        }
        let refs: Vec<&Array> = blocks.iter().collect();
        Array::concat(&refs, 2)
    }

    /// One-token attention step: the cache already contains exactly the keys
    /// this position may see, so no mask is needed.
    fn attention_step(
        &self,
        n: &Array,
        layer: &LayerWeights,
        kind: LayerKind,
        kv: &KvState,
        start: usize,
    ) -> Result<(Array, KvState)> {
        let (cos, sin) = self
            .cos_sin(kind, start, 1)
            .map_err(|e| err(format!("rope slice start={start}: {e}")))?;
        let stream = self.backend.stream();
        let heads = self.config.num_attention_heads;
        let eps = self.config.norm_eps;
        let dim = self.config.head_dim;
        Self::attention_step_parts(n, layer, kind, kv, start, &cos, &sin, stream, heads, eps, dim)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_step_parts(
        n: &Array,
        layer: &LayerWeights,
        kind: LayerKind,
        kv: &KvState,
        start: usize,
        cos: &Array,
        sin: &Array,
        stream: &Stream,
        heads: usize,
        eps: f32,
        dim: usize,
    ) -> Result<(Array, KvState)> {
        let Mixer::Attention {
            q: q_proj,
            k: k_proj,
            v: v_proj,
            o: o_proj,
            q_norm,
            k_norm,
            gate,
        } = &layer.mixer
        else {
            return Err(err("LFM3 attention layer has a conv weight plan"));
        };
        let kv_heads = kv.keys.shape()[1];
        let scale = (dim as f64).powf(-0.5) as f32;
        let q = q_proj
            .call(n)
            .map_err(|e| err(format!("q_proj: {e}")))?
            .reshape(&[1, 1, heads, dim])
            .map_err(|e| err(format!("q reshape n={:?}: {e}", n.shape())))?;
        let q = norm(&q, q_norm, eps)?.transpose(&[0, 2, 1, 3])?;
        let k = k_proj
            .call(n)?
            .reshape(&[1, 1, kv_heads, dim])
            .map_err(|e| err(format!("k reshape: {e}")))?;
        let k = norm(&k, k_norm, eps)?.transpose(&[0, 2, 1, 3])?;
        let v = v_proj
            .call(n)?
            .reshape(&[1, 1, kv_heads, dim])
            .map_err(|e| err(format!("v reshape: {e}")))?
            .transpose(&[0, 2, 1, 3])?;
        let q = apply_rope(&q, cos, sin, dim)?;
        let k = apply_rope(&k, cos, sin, dim)?;
        let index = Array::from_i32(stream, &[1], &[start as i32])?;
        let keys = kv.keys.slice_update(&k, &index, &[2])?;
        let values = kv.values.slice_update(&v, &index, &[2])?;
        let end = start + 1;
        // Sliding attention reads exactly the 256 positions before `end` (the
        // window is a fixed family constant; see `Lfm3Config::window`).
        let low = match kind {
            LayerKind::SlidingAttention => end.saturating_sub(256),
            _ => 0,
        };
        let attention = q.sdpa(
            &keys.slice_axis(2, low, end)?,
            &values.slice_axis(2, low, end)?,
            scale,
            false,
            None,
        )?;
        let gate = gate.call(n)?.sigmoid()?.transpose(&[0, 2, 1])?;
        let gated = attention.mul(
            &gate
                .reshape(&[1, heads, 1, 1])
                .map_err(|e| err(format!("gate reshape gate={:?}: {e}", gate.shape())))?,
        )?;
        let merged = gated
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, 1, heads * dim])?;
        let op = o_proj.call(&merged)?;
        Ok((op, KvState { keys, values }))
    }

    /// Sandwich-norm residual composition shared by both branches.
    fn mix_and_residual(
        &self,
        x: &Array,
        op: &Array,
        layer: &LayerWeights,
    ) -> Result<Array> {
        let eps = self.config.norm_eps;
        let op = norm(op, &layer.post_operator_norm, eps)?;
        let hidden = x.add(&op)?;
        let ffn = norm(&hidden, &layer.ffn_norm, eps)?;
        let gate = layer.w1.call(&ffn)?;
        let up = layer.w3.call(&ffn)?;
        let activated = silu_mul(&gate, &up, &self.kkernel)?;
        let down = layer.w2.call(&activated)?;
        let down = norm(&down, &layer.post_ffn_norm, eps)?;
        hidden.add(&down)
    }

    /// Build and evaluate one layer. `None` means a single-token step whose
    /// state the caller evaluates once with the logits.
    fn run_layer(
        &mut self,
        x: &Array,
        layer_idx: usize,
        start: usize,
        len: usize,
        evaluate: bool,
        states: Option<&mut Vec<LayerState>>,
    ) -> Result<Array> {
        let eps = self.config.norm_eps;
        let kind = self.kinds[layer_idx];
        let layer = &self.weights.layers[layer_idx];
        let n = norm(x, &layer.operator_norm, eps)?;
        let (op, state) = if kind.is_conv() {
            let state = self.conv[layer_idx]
                .clone()
                .ok_or_else(|| err("LFM3 conv state is missing"))?;
            let (op, next) = if len == 1 {
                self.conv_step(&n, layer, &state)?
            } else {
                self.conv_layer(&n, layer, &state, start > 0)?
            };
            (op, LayerState::Conv(next))
        } else {
            let kv = self.kv[layer_idx]
                .clone()
                .ok_or_else(|| err("LFM3 KV state is missing"))?;
            let (op, next) = if len == 1 {
                self.attention_step(&n, layer, kind, &kv, start)?
            } else {
                self.attention_layer(&n, layer, kind, &kv, start, len)?
            };
            (op, LayerState::Attention(next))
        };
        let output = self.mix_and_residual(x, &op, layer)?;
        if let Some(states) = states {
            // The fast arm publishes the captured snapshot into the live slots
            // inside `build_graph` so both stay in lockstep; skip the eval here
            // and let the caller evaluate the finished graph once.
            states.push(state);
        } else {
            self.publish_layer_state(layer_idx, state, &output, evaluate)?;
        }
        Ok(output)
    }

    /// Publish a completed layer's state to the request (and optionally eval
    /// it alongside the output). Shared by `run_layer` and the cached prefill
    /// graph so the store ordering cannot drift between the two paths.
    fn publish_layer_state(
        &mut self,
        layer_idx: usize,
        state: LayerState,
        output: &Array,
        evaluate: bool,
    ) -> Result<()> {
        match state {
            LayerState::Attention(next) => {
                if evaluate {
                    let needed = [output.clone(), next.keys.clone(), next.values.clone()];
                    self.backend.stream().eval(&needed)?;
                }
                self.kv[layer_idx] = Some(next);
            }
            LayerState::Conv(next) => {
                if evaluate {
                    let needed = [output.clone(), next.clone()];
                    self.backend.stream().eval(&needed)?;
                }
                self.conv[layer_idx] = Some(next);
            }
        }
        Ok(())
    }

    /// Restore request state from a prefill cache snapshot and re-evaluate the
    /// cached logits (QUEUE #3 cache-last-graph reuse).
    fn run_cached_prefill(
        &mut self,
        cache: &PrefillCache,
        len: usize,
        start: usize,
    ) -> Result<Array> {
        self.kv = cache
            .states
            .iter()
            .map(|s| match s {
                LayerState::Attention(kv) => Some(kv.clone()),
                LayerState::Conv(_) => None,
            })
            .collect();
        self.conv = cache
            .states
            .iter()
            .map(|s| match s {
                LayerState::Conv(c) => Some(c.clone()),
                LayerState::Attention(_) => None,
            })
            .collect();
        let stream = self.backend.stream().clone();
        stream.eval(std::slice::from_ref(&cache.logits))?;
        self.position = start + len;
        Ok(cache.logits.clone())
    }

    fn validate_input(&self, token_ids: &[u32], start_pos: usize) -> Result<usize> {
        if let Some(reason) = &self.invalid {
            return Err(err(format!(
                "LFM3 session invalidated: {reason}; reset required"
            )));
        }
        let end = self
            .position
            .checked_add(token_ids.len())
            .ok_or_else(|| err("LFM3 position overflow"))?;
        if token_ids.is_empty()
            || start_pos != self.position
            || end > self.capacity
            || token_ids
                .iter()
                .any(|&id| id as usize >= self.config.vocab_size)
        {
            return Err(err(
                "LFM3 forward requires nonempty in-vocabulary IDs, matching position and prepared capacity",
            ));
        }
        Ok(end)
    }

    /// Consume `token_ids` and return logits, either `[len, vocab]` or the last
    /// row only.
    fn execute(&mut self, token_ids: &[u32], start_pos: usize, last_only: bool) -> Result<Tensor> {
        if let Err(error) = self.validate_input(token_ids, start_pos) {
            self.invalid = Some(error.to_string());
            return Err(error);
        }
        match self.execute_inner(token_ids, last_only) {
            Ok(tensor) => Ok(tensor),
            Err(error) => {
                self.invalid = Some(error.to_string());
                Err(error)
            }
        }
    }

    /// Build the step graph from device-resident ids and update the KV/conv
    /// state references. Does not evaluate anything; the caller decides
    /// whether the logits (whose evaluation covers every state array this
    /// step produced) are pulled synchronously or by the pipelined driver.
    fn build_graph(
        &mut self,
        ids: &Array,
        start: usize,
        len: usize,
        last_only: bool,
    ) -> Result<Array> {
        let mut x = self
            .weights
            .embed(ids)?
            .reshape(&[1, len, self.config.hidden_size])?;
        // A prompt chunk evaluates every layer boundary (bounded peak memory);
        // a single-token step keeps one graph and evaluates once by the caller.
        let per_layer_eval = len > 1;
        // QUEUE #3: capture each layer's next state so the built graph can be
        // cached and replayed for the same static shape. Decode (len == 1) and
        // the unset arm never allocate or fill this vector (empty-arm gate).
        let mut cached_states = (self.prefill_fast && len > 1).then(Vec::new);
        for layer_idx in 0..self.config.num_hidden_layers {
            x = self
                .run_layer(&x, layer_idx, start, len, per_layer_eval, cached_states.as_mut())
                .map_err(|e| err(format!("layer {layer_idx}: {e}")))?;
        }
        let hidden = norm(&x, &self.weights.final_norm, self.config.norm_eps)
            .map_err(|e| err(format!("final norm: {e}")))?;
        let rows = if last_only {
            hidden.slice_axis(1, len - 1, len)?
        } else {
            hidden
        };
        let rows_shape = rows.shape().to_vec();
        let logits = self
            .weights
            .project(&rows, &self.kkernel)
            .map_err(|e| err(format!("head projection: {e}")))?
            .reshape(&[rows_shape[1], self.config.vocab_size])
            .map_err(|e| err(format!("head reshape rows={rows_shape:?}: {e}")))?;
        // Publish the captured states into both the request's live slots and
        // the cache snapshot, so the two paths stay in lockstep (a later
        // same-shape prompt must see the same state a rebuild would produce).
        if let Some(states) = cached_states {
            for (layer_idx, state) in states.iter().enumerate() {
                match state {
                    LayerState::Attention(next) => self.kv[layer_idx] = Some(next.clone()),
                    LayerState::Conv(next) => self.conv[layer_idx] = Some(next.clone()),
                }
            }
            self.prefill_cache = Some(PrefillCache {
                key: (len, start, start > 0),
                logits: logits.clone(),
                states,
            });
        }
        Ok(logits)
    }

    /// The step's state arrays: evaluating these alongside the logits keeps
    /// the single-token graph evaluated exactly once per step.
    fn step_state_arrays(&self) -> Vec<Array> {
        let mut required = Vec::new();
        for state in &self.kv {
            if let Some(kv) = state {
                required.push(kv.keys.clone());
                required.push(kv.values.clone());
            }
        }
        for state in &self.conv {
            if let Some(conv) = state {
                required.push(conv.clone());
            }
        }
        required
    }

    fn execute_inner(&mut self, token_ids: &[u32], last_only: bool) -> Result<Tensor> {
        let stream = self.backend.stream().clone();
        let len = token_ids.len();
        let start = self.position;
        let end = start + len;
        let ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
        let ids = Array::from_i32(&stream, &[len], &ids)?;
        // QUEUE #3 armed arm: when the static shape matches the cached
        // prefill graph, restore the request state and re-evaluate the cached
        // logits instead of rebuilding the 24-layer graph. Decode (len == 1)
        // and the unset arm always take the original build path (empty-arm).
        // The cache is taken out of `self` so the replay/build below can borrow
        // `self` mutably; a miss rebuilds and re-populates it.
        let cached = self
            .prefill_fast
            .then(|| self.prefill_cache.take())
            .flatten();
        let (cache, reuse) = match cached {
            Some(cache) if cache.key == (len, start, start > 0) => (Some(cache), true),
            other => (other, false),
        };
        let stages = self.stages && len == 1 && start > 0;
        let build_started = stages.then(Instant::now);
        let logits = if let Some(cache) = cache.as_ref() {
            self.run_cached_prefill(cache, len, start)?
        } else {
            self.build_graph(&ids, start, len, last_only)?
        };
        let build_ms = build_started.map(|t| t.elapsed().as_secs_f64() * 1000.);
        let mut required: Vec<Array> = vec![logits.clone()];
        if len == 1 {
            required.extend(self.step_state_arrays());
        }
        let eval_started = stages.then(Instant::now);
        if !reuse {
            if self.decode_async_eval && len == 1 && start > 0 {
                // QUEUE #4 probe: identical required arrays/stream/graph as the
                // blocking eval above, only the host wait moves out of MLX's
                // eval path to an explicit synchronize immediately after.
                #[cfg(test)]
                self.test_split_trace.push("async_eval");
                stream.async_eval(&required)?;
                #[cfg(test)]
                self.test_split_trace.push("synchronize");
                stream.synchronize()?;
            } else {
                stream.eval(&required)?;
            }
        }
        // A replayed cache is still valid for the same static key (its
        // snapshots are absolute-position state, independent of the prompt),
        // so put it back for the next same-shape prompt; a miss was rebuilt by
        // `build_graph` and already stored its fresh snapshot.
        if reuse {
            self.prefill_cache = cache;
        }
        let eval_ms = eval_started.map(|t| t.elapsed().as_secs_f64() * 1000.);
        if let (Some(build_ms), Some(eval_ms)) = (build_ms, eval_ms) {
            self.report_stage("sync", build_ms, eval_ms);
        }
        self.position = end;
        self.backend.from_array(logits)
    }

    fn restore_conv_snapshot(&mut self) {
        if let Some(snapshot) = self.conv_snapshot.take() {
            self.conv = snapshot;
        }
    }

    /// QUEUE #3 test toggle (P1/P3 harness): flip the fast arm on an already
    /// loaded model so the token gate does not pay for a second 3.4 GB weight
    /// load. Clearing the cache when switching off keeps the empty-arm
    /// assertion (`prefill_cache.is_none()`) load-bearing.
    #[cfg(test)]
    fn set_prefill_fast(&mut self, on: bool) {
        self.prefill_fast = on;
        if !on {
            self.prefill_cache = None;
        }
    }

    /// QUEUE #4 test toggle (E1/E3 harness): flip the async-eval split on an
    /// already loaded model so both arms share one weight set; the env is
    /// still reset at entry/exit so `decode_async_eval_enabled()` (banner
    /// source) cannot be poisoned by a prior panicked test.
    #[cfg(test)]
    fn set_decode_async_eval(&mut self, on: bool) {
        self.decode_async_eval = on;
    }
}

impl DeviceDecode for Lfm3Model {
    fn stream(&self) -> &Stream {
        self.backend.stream()
    }

    fn decode(&mut self, token: &Array) -> Result<Array> {
        if self.invalid.is_some() || self.position >= self.capacity {
            return Err(err("LFM3 decode requires valid prepared capacity"));
        }
        if token.shape() != [1] {
            return Err(err(format!(
                "LFM3 decode expects one device-resident token id, got {:?}",
                token.shape()
            )));
        }
        // The lookahead step may be discarded by `finish_decode`; the conv
        // windows it shifts must move back, so keep the pre-step references.
        self.conv_snapshot = Some(self.conv.clone());
        let start = self.position;
        let build_started = self.stages.then(Instant::now);
        let logits = match self.build_graph(token, start, 1, true) {
            Ok(logits) => logits,
            Err(error) => {
                self.invalid = Some(error.to_string());
                return Err(error);
            }
        };
        if let Some(build_ms) = build_started.map(|t| t.elapsed().as_secs_f64() * 1000.) {
            self.report_stage("device", build_ms, 0.0);
        }
        self.position = start + 1;
        Ok(logits)
    }

    fn finish_decode(&mut self, outcome: DecodeFinish) -> Result<()> {
        if let Err(error) = self.backend.stream().synchronize() {
            self.invalid = Some(error.to_string());
            return Err(error);
        }
        match outcome {
            DecodeFinish::PublishedPosition(position)
                if self.invalid.is_none()
                    && position <= self.position
                    && self.position - position <= 1 =>
            {
                if self.position != position {
                    // The discarded lookahead step wrote KV row `position`,
                    // which the next write overwrites and attention never
                    // reads past `[low, end)`. The conv windows, however,
                    // slid: restore the snapshot taken before that step.
                    #[cfg(test)]
                    if !self.disable_conv_restore {
                        self.restore_conv_snapshot();
                    }
                    #[cfg(not(test))]
                    self.restore_conv_snapshot();
                    self.position = position;
                }
                Ok(())
            }
            DecodeFinish::Invalidated => {
                self.invalid
                    .get_or_insert_with(|| "greedy request failed or was abandoned".into());
                Ok(())
            }
            _ => {
                self.invalid = Some("invalid LFM3 greedy rollback position".into());
                Err(err("invalid LFM3 greedy rollback position"))
            }
        }
    }
}

impl LlmTrait for Lfm3Model {
    fn preparation_status(&self) -> TextPreparationStatus {
        let ready = self.invalid.is_none()
            && self.capacity > 0
            && self.kv.len() == self.config.num_hidden_layers
            && self.position <= self.capacity
            && self.cos_full.is_some();
        TextPreparationStatus {
            state: if self.invalid.is_some() {
                TextPreparationState::Invalidated
            } else if ready {
                TextPreparationState::Ready
            } else {
                TextPreparationState::Unprepared
            },
            implementation: "metal_lfm3",
            variant: Some(VARIANT),
            // This port composes eager device primitives only: no compiled or
            // captured scope is claimed.
            compiled_scopes: Vec::new(),
            prepared_prompt_tokens: None,
            prepared_sequence_lengths: Vec::new(),
            kv_capacity: (self.capacity > 0).then_some(self.capacity),
            max_decode_rows: ready.then_some(1),
            error: self.invalid.clone(),
        }
    }

    fn load(
        _config: ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self> {
        Err(err(
            "metal_lfm3 requires its checkpoint config and layer plan; use AutoModel or Lfm3Model::new",
        ))
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        self.execute(token_ids, start_pos as usize, false)
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        if input.image.is_some() {
            return Err(err("LFM3 is a text-only model"));
        }
        if self.position != 0 {
            return Err(err(
                "LFM3 prefill starts a new request; reset the session first",
            ));
        }
        if let Err(error) = self.ensure_capacity(input.token_ids.len(), 0) {
            self.invalid = Some(error.to_string());
            return Err(error);
        }
        self.execute(input.token_ids, 0, true)
    }

    fn backend(&self) -> &dyn Backend {
        &self.backend
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn reset(&mut self) {
        self.kv = self.initial_kv.clone();
        self.conv = self.initial_conv.clone();
        self.conv_snapshot = None;
        self.position = 0;
        self.invalid = None;
    }

    /// Greedy decode runs on the pipelined device sequence unless the
    /// operator forces the legacy synchronous path with
    /// `APXINF_MLX_SYNC_DECODE=1` (forwarded into the worker with the
    /// gateway's `--worker-env`); anything but penalty-free greedy keeps the
    /// shared synchronous sequence.
    fn token_sequence<'a>(
        &'a mut self,
        first: Tensor,
        sampler: Box<dyn TokenSampler>,
        params: &TokenSamplingParams,
        prompt_end: u32,
    ) -> Result<Box<dyn TokenSequence + 'a>> {
        if crate::mlx_decode::enabled(params) {
            let first = self.backend.array(&first)?;
            Ok(Box::new(GreedySequence::new(self, first, prompt_end)?))
        } else {
            Ok(Box::new(crate::llm_trait::SyncTokenSequence::new(
                self, first, sampler, prompt_end,
            )))
        }
    }

    /// Drop everything the finished request owned: KV rows, conv history, the
    /// rotary tables and the position capacity. Weights stay resident.
    fn release_request_state(&mut self) -> Result<()> {
        self.backend.stream().synchronize()?;
        let layers = self.config.num_hidden_layers;
        self.kv = vec![None; layers];
        self.conv = vec![None; layers];
        self.initial_kv = vec![None; layers];
        self.initial_conv = vec![None; layers];
        self.cos_full = None;
        self.sin_full = None;
        self.cos_sliding = None;
        self.sin_sliding = None;
        self.capacity = 0;
        self.position = 0;
        self.invalid = None;
        self.conv_snapshot = None;
        self.backend.stream().synchronize()
    }

    /// Keep the model-owned position capacity and rotary tables, drop only the
    /// request's KV/conv content.
    fn release_request_state_retaining_preparation(&mut self) -> Result<()> {
        self.backend.stream().synchronize()?;
        self.kv = self.initial_kv.clone();
        self.conv = self.initial_conv.clone();
        self.conv_snapshot = None;
        self.position = 0;
        self.invalid = None;
        self.backend.stream().synchronize()
    }

    fn prewarm_decode(&mut self, prompt_len: usize, max_new_tokens: usize) {
        if let Err(error) = self.prepare(prompt_len, max_new_tokens) {
            self.invalid = Some(error.to_string());
        }
    }
}

/// Implementation identity reported through `preparation_status`.
pub const VARIANT: &str = "bf16-eager";

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random values in `[-scale, scale)`; enough entropy
    /// to exercise the rounding paths without a checkpoint.
    fn values(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                (unit * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// The decode short-conv arithmetic: one fused causal-depthwise kernel over
    /// the seven-column window versus the pre-migration seven-tap host-side
    /// accumulation, on the same window and the same checkpoint-shaped tap
    /// table.
    ///
    /// The two arms issue different kernels, so this test does not *require*
    /// bit-exactness; it measures the residual and the element-exact fraction
    /// against the migration's 2e-3 numeric gate. Both operands are BF16, so
    /// every tap product is exactly representable in F32 (8 + 8 mantissa bits
    /// < 24), and both arms add those exact products in tap order and round once
    /// to BF16; the expected residual is therefore zero, and the measured values
    /// are reported below rather than asserted a priori.
    #[test]
    #[ignore = "requires shared Metal qualification lock"]
    fn fused_decode_conv_matches_the_per_tap_reference() -> Result<()> {
        let stream = Stream::metal(0)?;
        let conv = CausalDepthwiseConv::new(&stream)?;
        let (hidden, kernel) = (2048, 7);
        // The tap table as `Weights::load` produces it: BF16 checkpoint bytes
        // widened to F32, then handed to the fused kernel as BF16 again (an
        // exact round trip, not a second quantization).
        let taps_f32 = Array::from_f32(
            &stream,
            &[hidden, kernel],
            &values(hidden * kernel, 7, 0.25),
        )?
        .cast(MlxDType::BF16)?
        .cast(MlxDType::F32)?;
        let taps_bf16 = taps_f32.cast(MlxDType::BF16)?;
        let window = Array::from_f32(
            &stream,
            &[1, kernel, hidden],
            &values(kernel * hidden, 19, 2.0),
        )?
        .cast(MlxDType::BF16)?;

        let (fused, next) = fused_conv_window(&conv, &window, &taps_bf16)?;
        let (reference, reference_next) = legacy_window_conv(&window, &taps_f32)?;
        stream.eval(&[fused.clone(), reference.clone(), next.clone()])?;

        let got = fused.cast(MlxDType::F32)?.to_f32_vec()?;
        let want = reference.cast(MlxDType::F32)?.to_f32_vec()?;
        let (mut max_abs, mut exact) = (0f64, 0usize);
        for (a, b) in got.iter().zip(&want) {
            exact += usize::from(a == b);
            max_abs = max_abs.max((*a as f64 - *b as f64).abs());
        }
        eprintln!(
            "decode-conv fused vs per-tap: n={} max_abs_diff={max_abs:.3e} element_exact={exact}/{} ({:.3}%)",
            got.len(),
            got.len(),
            100.0 * exact as f64 / got.len() as f64
        );
        assert!(
            max_abs <= 2e-3,
            "decode-conv parity residual {max_abs:.3e} exceeds the 2e-3 numeric gate"
        );
        // The next-step state is the leading rows of the same window: both
        // implementations must hand the caller those identical columns.
        assert_eq!(next.shape(), reference_next.shape());
        assert_eq!(next.to_f32_vec()?, reference_next.to_f32_vec()?);
        Ok(())
    }

    // ---- Pipelined greedy decode acceptance tests (issue #122). Real ----
    // checkpoint, real Metal: `APXINF_LFM3_TEST_MODEL=<snapshot dir> cargo
    // test -p apxinf-model --features mlx,coreml metal_lfm3 -- --ignored
    // --test-threads=1` under the shared Metal qualification lock. The tests
    // mutate process env to select the schedule, so they must run
    // single-threaded.

    use crate::generation_config::GenerationOptions;
    use crate::llm_trait::GenerationCancellation;

    const SYNC_ENV: &str = "APXINF_MLX_SYNC_DECODE";
    const NO_RESTORE_ENV: &str = "APXINF_LFM3_TEST_NO_CONV_RESTORE";
    const KKERNEL_ENV: &str = "APXINF_LFM3_KKERNEL";
    const PREFILL_FAST_ENV: &str = "APXINF_LFM3_PREFILL_FAST";
    const DECODE_ASYNC_EVAL_ENV: &str = "APXINF_LFM3_DECODE_ASYNC_EVAL";

    fn test_checkpoint() -> Result<std::path::PathBuf> {
        std::env::var_os("APXINF_LFM3_TEST_MODEL")
            .map(std::path::PathBuf::from)
            .ok_or_else(|| err("set APXINF_LFM3_TEST_MODEL to the LFM3 snapshot dir"))
    }

    fn load_checkpoint_model() -> Result<Lfm3Model> {
        let path = test_checkpoint()?;
        let json = std::fs::read_to_string(path.join("config.json"))
            .map_err(|e| err(format!("cannot read LFM3 config: {e}")))?;
        let config = Lfm3Config::from_json(&json)?;
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(&path)
            .map_err(|e| err(format!("cannot load LFM3 checkpoint: {e}")))?;
        let mut model = Lfm3Model::new(config, tensors, Arc::new(MlxBackend::new(0)?))?;
        if std::env::var_os(NO_RESTORE_ENV).is_some() {
            model.disable_conv_restore = true;
        }
        Ok(model)
    }

    fn pseudo_prompt(len: usize, seed: u64) -> Vec<u32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 33) as u32) % 120_000
            })
            .collect()
    }

    fn set_sync_schedule(on: bool) {
        if on {
            std::env::set_var(SYNC_ENV, "1");
        } else {
            std::env::remove_var(SYNC_ENV);
        }
    }

    fn set_kkernel(on: bool) {
        if on {
            std::env::set_var(KKERNEL_ENV, "1");
        } else {
            std::env::remove_var(KKERNEL_ENV);
        }
    }

    fn set_prefill_fast(on: bool) {
        if on {
            std::env::set_var(PREFILL_FAST_ENV, "1");
        } else {
            std::env::remove_var(PREFILL_FAST_ENV);
        }
    }

    fn set_decode_async_eval(on: bool) {
        if on {
            std::env::set_var(DECODE_ASYNC_EVAL_ENV, "1");
        } else {
            std::env::remove_var(DECODE_ASYNC_EVAL_ENV);
        }
    }

    fn run_greedy(model: &mut Lfm3Model, prompt: &[u32], budget: usize) -> Result<Vec<u32>> {
        let options = GenerationOptions::greedy(budget, None);
        let output = model.generate_streaming_with_options(
            crate::llm_trait::GenerationRequest {
                input: LlmInput::text(prompt),
                options: &options,
            },
            |_| {},
        )?;
        Ok(output.token_ids())
    }

    /// The production greedy rule (NaN/+inf/-inf normalisation, first-index
    /// ties) applied to a forward's logits row.
    fn greedy_from_logits(model: &Lfm3Model, logits: &Tensor) -> Result<u32> {
        let array = model.backend.array(logits)?;
        let (_feed, checked) = apxinf_mlx::greedy_token_arrays(&array)?;
        let bytes = checked.to_bytes_ready()?;
        let token = u32::from_ne_bytes(
            bytes
                .try_into()
                .map_err(|_| err("greedy token must contain one u32"))?,
        );
        if token == u32::MAX {
            return Err(err("all token logits are invalid"));
        }
        Ok(token)
    }

    /// Byte snapshot of every attention layer's KV rows `[0, position)` and
    /// every conv layer's window — the continuation state a request hands to
    /// its sequel.
    fn state_bytes(
        model: &Lfm3Model,
        position: usize,
    ) -> Result<(Vec<Option<Vec<u8>>>, Vec<Option<Vec<u8>>>)> {
        model.backend.stream().synchronize()?;
        let kv = model
            .kv
            .iter()
            .map(|state| {
                state
                    .as_ref()
                    .map(|kv| -> Result<Vec<u8>> {
                        let keys = kv
                            .keys
                            .slice_axis(2, 0, position)?
                            .contiguous()?
                            .to_bytes_ready()?;
                        let values = kv
                            .values
                            .slice_axis(2, 0, position)?
                            .contiguous()?
                            .to_bytes_ready()?;
                        let mut bytes = keys;
                        bytes.extend_from_slice(&values);
                        Ok(bytes)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let conv = model
            .conv
            .iter()
            .map(|state| {
                state
                    .as_ref()
                    .map(|array| array.contiguous()?.to_bytes_ready())
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((kv, conv))
    }

    /// Drive `take` pipelined reads with lookahead, then finish as the driver
    /// does on an EOS break after `take` publishes: one step is rolled back.
    fn drive_pipelined_prefix(
        model: &mut Lfm3Model,
        prompt: &[u32],
        take: usize,
    ) -> Result<Vec<u32>> {
        let prompt_end = prompt.len() as u32;
        let first = model.prefill(LlmInput::text(prompt))?;
        let first = model.backend.array(&first)?;
        let mut seq = Box::new(GreedySequence::new(model, first, prompt_end)?);
        let mut published = Vec::with_capacity(take);
        for _ in 0..take {
            published.push(seq.next(true)?.token_id);
        }
        seq.finish(take)?;
        Ok(published)
    }

    /// T1: pipelined and synchronous greedy emit identical token ids on every
    /// prompt, at the full 128-token decode depth of the benchmark cell.
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn t1_pipelined_greedy_matches_sync_token_for_token() -> Result<()> {
        let prompts: Vec<Vec<u32>> = (0..4).map(|i| pseudo_prompt(24, 101 + i)).collect();
        let mut model = load_checkpoint_model()?;
        set_sync_schedule(true);
        let mut expected = Vec::with_capacity(prompts.len());
        for prompt in &prompts {
            expected.push(run_greedy(&mut model, prompt, 128)?);
        }
        set_sync_schedule(false);
        for (index, prompt) in prompts.iter().enumerate() {
            let got = run_greedy(&mut model, prompt, 128)?;
            assert_eq!(
                got, expected[index],
                "prompt {index}: pipelined greedy diverged from sync"
            );
        }
        Ok(())
    }

    // ---- QUEUE #2 custom-kernel gates (doc/lfm3-lm-head-custom-kernel.md
    // §9). Same real-checkpoint Metal lock and `--test-threads=1` discipline as
    // T1/T2/T3; they mutate `APXINF_LFM3_KKERNEL`, which selects the kernels at
    // model construction.

    /// K1: the custom lm_head GEMV reproduces MLX's `[1,1,K]×[K,V]` matmul
    /// bitwise (doc §9 K1's strict form). If MLX reduces in a different order,
    /// the failure message carries the numeric delta to be recorded as the
    /// amendment the doc permits — the hard gate is tokens, not logits (K3).
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn k1_lm_head_kernel_matches_matmul_bitwise() -> Result<()> {
        use apxinf_mlx::fusions::LmHeadGemv;
        let stream = Stream::metal(0)?;
        let gemv = LmHeadGemv::new(&stream)?;
        for (k, v, seed) in [(8usize, 1usize, 23u64), (32, 64, 29)] {
            let x = Array::from_f32(&stream, &[1, 1, k], &values(k, seed, 0.5))?
                .cast(MlxDType::BF16)?;
            let w = Array::from_f32(&stream, &[k, v], &values(k * v, seed + 1, 0.25))?
                .cast(MlxDType::BF16)?;
            let custom = gemv.call(&x, &w)?;
            let reference = x.matmul(&w)?;
            assert_eq!(custom.shape(), reference.shape(), "K1: output shape must match");
            stream.eval(&[custom.clone(), reference.clone()])?;
            let got = custom.cast(MlxDType::F32)?.to_f32_vec()?;
            let want = reference.cast(MlxDType::F32)?.to_f32_vec()?;
            let (mut max_abs, mut exact) = (0f64, 0usize);
            for (a, b) in got.iter().zip(&want) {
                exact += usize::from(a == b);
                max_abs = max_abs.max((*a as f64 - *b as f64).abs());
            }
            eprintln!(
                "K1 lm_head vs matmul k={k} v={v}: max_abs_diff={max_abs:.3e} element_exact={exact}/{}",
                got.len()
            );
            assert_eq!(
                got, want,
                "K1: lm_head GEMV must match MLX matmul bitwise (k={k} v={v}, max_abs_diff={max_abs:.3e}, exact={exact}/{}); if MLX reduces in a different order, record this delta as the §9 K1 amendment — the hard gate is tokens, not logits",
                got.len()
            );
        }
        Ok(())
    }

    /// K2: the fused `silu(gate)*up` matches the unfused graph on the exact
    /// decode shape within the registered `rtol=1e-6, atol=1e-6` (doc §9 K2).
    /// The delta and exact-element fraction are printed so a failure is
    /// diagnostic rather than opaque.
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn k2_fused_silu_mul_matches_graph_reference() -> Result<()> {
        use apxinf_mlx::fusions::SiluMulFused;
        let stream = Stream::metal(0)?;
        let fused = SiluMulFused::new(&stream)?;
        let gate = Array::from_f32(&stream, &[1, 1, 2048], &values(2048, 37, 2.0))?
            .cast(MlxDType::BF16)?;
        let up = Array::from_f32(&stream, &[1, 1, 2048], &values(2048, 41, 2.0))?
            .cast(MlxDType::BF16)?;
        let custom = fused.call(&gate, &up)?;
        let reference = gate.mul(&gate.sigmoid()?)?.mul(&up)?;
        assert_eq!(custom.shape(), reference.shape(), "K2: output shape must match");
        stream.eval(&[custom.clone(), reference.clone()])?;
        let got = custom.cast(MlxDType::F32)?.to_f32_vec()?;
        let want = reference.cast(MlxDType::F32)?.to_f32_vec()?;
        let (mut max_abs, mut exact) = (0f64, 0usize);
        for (a, b) in got.iter().zip(&want) {
            exact += usize::from(a == b);
            max_abs = max_abs.max((*a as f64 - *b as f64).abs());
        }
        eprintln!(
            "K2 fused silu vs graph: max_abs_diff={max_abs:.3e} element_exact={exact}/{}",
            got.len()
        );
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (*a - *b).abs() <= 1e-6 + 1e-6 * b.abs(),
                "K2: element {i} fused={a} graph={b} exceeds rtol=1e-6 atol=1e-6 (max_abs_diff={max_abs:.3e}); record the numeric delta as an amendment — the hard gate is tokens (K3)"
            );
        }
        Ok(())
    }

    /// K3 — the hard gate (doc §5.1): with both custom kernels armed, greedy
    /// decode emits the identical token ids on every prompt at the full
    /// 128-token depth. One differing token ⇒ INVALID. The env switch is
    /// cleared at entry as well as exit so a prior panicked test cannot poison
    /// the baseline arm (the empty-arm trap depends on it).
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn k3_armed_kkernel_greedy_matches_baseline_token_for_token() -> Result<()> {
        let prompts: Vec<Vec<u32>> = (0..4).map(|i| pseudo_prompt(24, 101 + i)).collect();

        set_kkernel(false);
        let mut baseline = load_checkpoint_model()?;
        assert!(!baseline.kkernel.lm_head_armed(), "K3: baseline arm must not construct the lm_head kernel");
        assert!(!baseline.kkernel.silu_mul_armed(), "K3: baseline arm must not construct the silu_mul kernel");
        let mut expected = Vec::with_capacity(prompts.len());
        for prompt in &prompts {
            expected.push(run_greedy(&mut baseline, prompt, 128)?);
        }

        set_kkernel(true);
        let mut armed = load_checkpoint_model()?;
        assert!(armed.kkernel.lm_head_armed(), "K3: empty-arm trap — lm_head kernel not constructed");
        assert!(armed.kkernel.silu_mul_armed(), "K3: empty-arm trap — silu_mul kernel not constructed");
        for (index, prompt) in prompts.iter().enumerate() {
            let got = run_greedy(&mut armed, prompt, 128)?;
            assert_eq!(
                got, expected[index],
                "prompt {index}: armed kkernel diverged from baseline — INVALID (doc §5.1)"
            );
        }

        set_kkernel(false);
        Ok(())
    }

    /// K4 — the shape guard is load-bearing (doc §5.2 / §9 K4): padded,
    /// mis-rank, K-mismatched, shape-mismatched, or widened (non-BF16) inputs
    /// must fail loudly, while the exact registered geometry must succeed, so
    /// the `Err` assertions above cannot be vacuous.
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn k4_shape_guard_rejects_padded_and_mismatched_inputs() -> Result<()> {
        use apxinf_mlx::fusions::{LmHeadGemv, SiluMulFused};
        let stream = Stream::metal(0)?;
        let gemv = LmHeadGemv::new(&stream)?;
        let silu = SiluMulFused::new(&stream)?;

        // Exact geometry must dispatch and preserve the output shape.
        let x = Array::from_f32(&stream, &[1, 1, 8], &values(8, 53, 0.5))?
            .cast(MlxDType::BF16)?;
        let w = Array::from_f32(&stream, &[8, 16], &values(128, 59, 0.25))?
            .cast(MlxDType::BF16)?;
        assert_eq!(gemv.call(&x, &w)?.shape(), &[1, 1, 16], "K4: exact geometry must dispatch");

        // lm_head guards: rank, K match, dtype.
        let bad_rank = Array::from_f32(&stream, &[1, 16], &values(16, 61, 0.5))?
            .cast(MlxDType::BF16)?;
        assert!(gemv.call(&bad_rank, &w).is_err(), "K4: non-[1,1,K] input must be rejected");
        let w_k_mismatch = Array::from_f32(&stream, &[7, 16], &values(112, 67, 0.25))?
            .cast(MlxDType::BF16)?;
        assert!(gemv.call(&x, &w_k_mismatch).is_err(), "K4: K mismatch must be rejected");
        let x_f32 = Array::from_f32(&stream, &[1, 1, 8], &values(8, 71, 0.5))?;
        assert!(gemv.call(&x_f32, &w).is_err(), "K4: widened (non-BF16) input must be rejected");

        // silu_mul guards: shape and dtype.
        let gate = Array::from_f32(&stream, &[1, 1, 8], &values(8, 73, 1.0))?
            .cast(MlxDType::BF16)?;
        let up_ok = Array::from_f32(&stream, &[1, 1, 8], &values(8, 79, 1.0))?
            .cast(MlxDType::BF16)?;
        assert_eq!(silu.call(&gate, &up_ok)?.shape(), &[1, 1, 8], "K4: exact silu geometry must dispatch");
        let up_wide = Array::from_f32(&stream, &[1, 1, 16], &values(16, 83, 1.0))?
            .cast(MlxDType::BF16)?;
        assert!(silu.call(&gate, &up_wide).is_err(), "K4: silu_mul shape mismatch must be rejected");
        let gate_f32 = Array::from_f32(&stream, &[1, 1, 8], &values(8, 89, 1.0))?;
        let up_f32 = Array::from_f32(&stream, &[1, 1, 8], &values(8, 97, 1.0))?;
        assert!(silu.call(&gate_f32, &up_f32).is_err(), "K4: silu_mul widened (non-BF16) inputs must be rejected");
        Ok(())
    }

    /// T2: EOS and budget edges return the same sequences as sync and land on
    /// `position == P+N-1`. The first-token-EOS case is driven at the
    /// sequence level so it is deterministic; cancel is driven through the
    /// shared generation loop.
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn t2_budget_eos_and_cancel_boundaries_match_sync() -> Result<()> {
        let prompt = pseudo_prompt(16, 7);
        let mut model = load_checkpoint_model()?;

        // Budget exhaustion: 1 and 8 tokens, both schedules, same prompt.
        for budget in [1usize, 8] {
            set_sync_schedule(true);
            let sync_ids = run_greedy(&mut model, &prompt, budget)?;
            let sync_pos = model.position();
            set_sync_schedule(false);
            let pipe_ids = run_greedy(&mut model, &prompt, budget)?;
            let pipe_pos = model.position();
            assert_eq!(sync_ids, pipe_ids, "budget {budget}: published tokens");
            assert_eq!(pipe_pos, sync_pos, "budget {budget}: final position");
            assert_eq!(
                sync_pos,
                prompt.len() + sync_ids.len() - 1,
                "budget {budget}: position must be P+N-1"
            );
        }

        // First-token EOS: synthetic prefill logits whose argmax is EOS. The
        // driver-side break after one publish must roll the lookahead step
        // back, leaving exactly the prefill state (the sync outcome).
        let prompt_end = prompt.len() as u32;
        model.reset();
        model.prefill(LlmInput::text(&prompt))?; // real prefill state; first logits unused
        let mut values = vec![-1e30f32; model.config.vocab_size];
        values[124_894] = 1.0; // any in-vocab id stands in for EOS here
        let synthetic = Array::from_f32(model.backend.stream(), &[1, values.len()], &values)?;
        let mut seq = Box::new(GreedySequence::new(&mut model, synthetic, prompt_end)?);
        let token = seq.next(true)?;
        assert_eq!(
            token.token_id, 124_894,
            "synthetic EOS must be the first token"
        );
        seq.finish(1)?;
        assert_eq!(
            model.position(),
            prompt.len(),
            "first-token EOS must roll back to the prefill position"
        );
        let (kv, conv) = state_bytes(&model, model.position())?;
        // Sync reference for this scenario is prefill-only state.
        model.reset();
        model.prepare(prompt.len(), 0)?;
        model.prefill(LlmInput::text(&prompt))?;
        let (kv_ref, conv_ref) = state_bytes(&model, model.position())?;
        assert_eq!(kv, kv_ref, "first-token EOS: KV must equal prefill state");
        assert_eq!(
            conv, conv_ref,
            "first-token EOS: conv must equal prefill state"
        );

        // Cancel right after the first publish: the lookahead decode is
        // discarded and both schedules agree on the published prefix.
        set_sync_schedule(true);
        let cancel_sync = GenerationCancellation::default();
        let mut published_sync = Vec::new();
        let options = GenerationOptions::greedy(16, None);
        let cancel_flag = cancel_sync.clone();
        model.generate_streaming_cancellable_dyn(
            crate::llm_trait::GenerationRequest {
                input: LlmInput::text(&prompt),
                options: &options,
            },
            &cancel_sync,
            &mut |token| {
                published_sync.push(token.token_id);
                cancel_flag.cancel();
            },
        )?;
        let sync_pos = model.position();
        set_sync_schedule(false);
        let cancel_pipe = GenerationCancellation::default();
        let mut published_pipe = Vec::new();
        let cancel_flag = cancel_pipe.clone();
        model.generate_streaming_cancellable_dyn(
            crate::llm_trait::GenerationRequest {
                input: LlmInput::text(&prompt),
                options: &options,
            },
            &cancel_pipe,
            &mut |token| {
                published_pipe.push(token.token_id);
                cancel_flag.cancel();
            },
        )?;
        assert_eq!(published_pipe, published_sync, "cancel: published prefix");
        assert_eq!(model.position(), sync_pos, "cancel: final position");
        assert_eq!(
            model.position(),
            prompt.len() + published_pipe.len() - 1,
            "cancel: position must be P+N-1"
        );
        Ok(())
    }

    /// T3: after a pipelined run whose lookahead step was rolled back, every
    /// KV row `[0, position)` and all 13 conv windows are byte-identical to
    /// the synchronous reference, and a 16-token continuation matches too.
    /// The manual early stop makes the rollback deterministic.
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn t3_pipelined_continuation_state_matches_sync_byte_for_byte() -> Result<()> {
        let prompt = pseudo_prompt(24, 11);
        let take = 12usize;
        let mut pipe = load_checkpoint_model()?;

        // Pipelined arm: `take` publishes plus one lookahead step, finished
        // like an EOS break so the extra step is rolled back.
        let published = drive_pipelined_prefix(&mut pipe, &prompt, take)?;
        let position = pipe.position();
        assert_eq!(
            position,
            prompt.len() + take - 1,
            "rolled-back position must be P+N-1"
        );
        let (kv_pipe, conv_pipe) = state_bytes(&pipe, position)?;

        // Continuation on the pipe arm, seeded with the last published token.
        let mut cont_pipe = Vec::new();
        let mut prev = published[take - 1];
        for _ in 0..16 {
            let logits = pipe.forward(&[prev], pipe.position() as u32)?;
            prev = greedy_from_logits(&pipe, &logits)?;
            cont_pipe.push(prev);
        }

        // Synchronous reference: prefill, then consume the same published
        // trajectory for `take - 1` steps (the token read from the last
        // legit logits is the continuation seed, not a consumed input).
        let mut reference = load_checkpoint_model()?;
        reference.prepare(prompt.len(), take + 16)?;
        reference.prefill(LlmInput::text(&prompt))?;
        for &token in &published[..take - 1] {
            reference.forward(&[token], reference.position() as u32)?;
        }
        assert_eq!(reference.position(), position);
        let (kv_ref, conv_ref) = state_bytes(&reference, position)?;
        assert_eq!(
            kv_pipe, kv_ref,
            "KV rows [0, position) must match sync byte-for-byte"
        );
        assert_eq!(
            conv_pipe, conv_ref,
            "13 conv windows must match sync byte-for-byte after the rollback"
        );

        let mut cont_ref = Vec::new();
        let mut prev = published[take - 1];
        for _ in 0..16 {
            let logits = reference.forward(&[prev], reference.position() as u32)?;
            prev = greedy_from_logits(&reference, &logits)?;
            cont_ref.push(prev);
        }
        assert_eq!(
            cont_pipe, cont_ref,
            "16-token continuation after the rollback must match sync"
        );
        Ok(())
    }

    /// T3's fault proof: with the conv restore disabled, the byte-level state
    /// check fails. `APXINF_LFM3_TEST_NO_CONV_RESTORE=1 cargo test ...
    /// t3_pipelined` makes the state equality in T3 fail — that output is the
    /// PR evidence. This companion test pins the positive direction without
    /// the env knob.
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn t3_fault_injection_without_conv_restore_is_detected() -> Result<()> {
        let prompt = pseudo_prompt(24, 11);
        let take = 12usize;
        let mut faulted = load_checkpoint_model()?;
        faulted.disable_conv_restore = true;
        drive_pipelined_prefix(&mut faulted, &prompt, take)?;
        let (_, conv_faulted) = state_bytes(&faulted, faulted.position())?;

        let mut reference = load_checkpoint_model()?;
        reference.disable_conv_restore = false;
        drive_pipelined_prefix(&mut reference, &prompt, take)?;
        let (_, conv_reference) = state_bytes(&reference, reference.position())?;

        assert_ne!(
            conv_faulted, conv_reference,
            "fault injection left the conv windows identical — the T3 state check would not catch the missing restore"
        );
        let diff = conv_faulted
            .iter()
            .zip(&conv_reference)
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "fault injection: {diff}/{} conv layers diverged after the skipped restore",
            conv_reference.len()
        );
        Ok(())
    }

    /// P1 — the QUEUE #3 hard gate (doc §Gate.1): with the fast arm on, greedy
    /// decode emits the identical token ids on every prompt at the full
    /// 128-token depth. One differing token ⇒ INVALID. The fast toggle flips
    /// the already-loaded model so both arms share one weight set; the env is
    /// still reset at entry and exit so `prefill_fast_enabled()` (and thus the
    /// banner source) cannot be poisoned by a prior panicked test.
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn p1_fast_prefill_greedy_matches_baseline_token_for_token() -> Result<()> {
        let prompts: Vec<Vec<u32>> = (0..4).map(|i| pseudo_prompt(24, 201 + i)).collect();

        set_prefill_fast(false);
        let mut model = load_checkpoint_model()?;
        model.set_prefill_fast(false);
        assert!(!model.prefill_fast, "P1: baseline arm must be off");
        assert!(model.prefill_cache.is_none(), "P1: baseline arm must hold no cached graph");
        let mut expected = Vec::with_capacity(prompts.len());
        for prompt in &prompts {
            expected.push(run_greedy(&mut model, prompt, 128)?);
        }

        model.set_prefill_fast(true);
        assert!(model.prefill_fast, "P1: fast arm toggle did not take");
        for (index, prompt) in prompts.iter().enumerate() {
            let got = run_greedy(&mut model, prompt, 128)?;
            assert_eq!(
                got, expected[index],
                "prompt {index}: fast prefill diverged from baseline — INVALID (doc §Gate.1)"
            );
        }

        model.set_prefill_fast(false);
        set_prefill_fast(false);
        Ok(())
    }

    /// P2 — the empty-arm gate (doc §Gate.4): with the env unset, the model
    /// captured the fast arm off, holds no cached graph after a full request,
    /// and `prefill_fast_enabled()` (the same source the worker banner reads)
    /// reports false. If the fast path dispatched while the switch was off,
    /// `prefill_cache` would be `Some` here.
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn p2_fast_arm_stays_off_when_switch_unset() -> Result<()> {
        set_prefill_fast(false);
        assert!(!prefill_fast_enabled(), "P2: helper must report false when the env is unset");
        let mut model = load_checkpoint_model()?;
        assert!(!model.prefill_fast, "P2: model captured the fast arm while the env was unset");
        let prompt = pseudo_prompt(24, 301);
        let ids = run_greedy(&mut model, &prompt, 32)?;
        assert_eq!(ids.len(), 32, "P2: budget must be honoured");
        assert!(model.prefill_cache.is_none(), "P2: empty-arm — a cached prefill graph was built with the switch off");
        Ok(())
    }

    /// P3 — cache correctness (doc §Tests P3): the armed model runs two
    /// different prompts back to back. The second prompt (a different length,
    /// so the static key differs) must match a fresh single-prompt run token
    /// for token. This is the stale-cache canary for the review-time finding
    /// that a cache hit replays the previous prompt's logits array without
    /// re-binding the new ids.
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn p3_cached_second_prompt_matches_fresh_run() -> Result<()> {
        let first = pseudo_prompt(24, 401);
        let second = pseudo_prompt(128, 402);

        // Armed model: two different prompts in one process. The second has a
        // different static key, so it must rebuild (or be provably equivalent).
        set_prefill_fast(true);
        let mut model = load_checkpoint_model()?;
        model.set_prefill_fast(true);
        let armed_first = run_greedy(&mut model, &first, 64)?;
        let armed_second = run_greedy(&mut model, &second, 64)?;

        // Fresh armed model, only the second prompt — the single-run reference.
        let mut fresh = load_checkpoint_model()?;
        fresh.set_prefill_fast(true);
        let fresh_second = run_greedy(&mut fresh, &second, 64)?;
        assert_eq!(
            armed_first.len(), 64,
            "P3: first prompt budget must be honoured before the second starts"
        );
        assert_eq!(
            armed_second, fresh_second,
            "P3: second prompt in-process diverged from a fresh run — stale cache (doc §Tests P3)"
        );

        // Same-shape replay: two equal-length prompts in a row must not return
        // the first prompt's logits either — run peer_b in the already-warm
        // model and compare against peer_b in a fresh model.
        let peer_a = pseudo_prompt(24, 501);
        let peer_b = pseudo_prompt(24, 502);
        run_greedy(&mut fresh, &peer_a, 64)?;
        let replayed_b = run_greedy(&mut fresh, &peer_b, 64)?;
        let mut fresh_b = load_checkpoint_model()?;
        fresh_b.set_prefill_fast(true);
        let expected_b = run_greedy(&mut fresh_b, &peer_b, 64)?;
        assert_eq!(
            replayed_b, expected_b,
            "P3: same-shape replay of a different prompt diverged from a fresh run — stale cache on cache-hit (doc §Tests P3)"
        );

        model.set_prefill_fast(false);
        set_prefill_fast(false);
        Ok(())
    }

    // ---- QUEUE #4 eval/sync split gates (doc/lfm3-decode-eval-sync-split.md
    // §9). The split lives in `execute_inner`'s single-token eval, which only
    // the **sync** schedule reaches (`SyncTokenSequence` → `forward` →
    // `execute`); the pipelined `GreedySequence` builds the step graph through
    // `DeviceDecode::decode` and never calls it. Every armed arm therefore
    // forces `APXINF_MLX_SYNC_DECODE=1` and asserts `test_split_trace` fired —
    // a token-equal result from a switch that never ran is the empty-arm trap,
    // not a pass. The schedule/arm mismatch this implies for the measurement
    // window is recorded in STATUS for lead adjudication.

    /// E1 — the §5.1 token gate. Baseline (split off) and armed (split on)
    /// must emit identical token ids for 4 prompts × 128 tokens under the sync
    /// schedule, and the armed arm must have run `async_eval` → `synchronize`
    /// on every decode step. Entry and exit both reset the env so a prior
    /// panic cannot poison `decode_async_eval_enabled()` (banner source).
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn e1_split_greedy_matches_baseline_token_for_token() -> Result<()> {
        let prompts: Vec<Vec<u32>> = (0..4).map(|i| pseudo_prompt(24, 721 + i)).collect();
        set_sync_schedule(true);
        set_decode_async_eval(false);

        let mut model = load_checkpoint_model()?;
        model.set_decode_async_eval(false);
        assert!(!model.decode_async_eval, "E1: baseline arm must start with the split off");
        model.test_split_trace.clear();
        let mut expected = Vec::with_capacity(prompts.len());
        for prompt in &prompts {
            expected.push(run_greedy(&mut model, prompt, 128)?);
        }
        assert!(
            model.test_split_trace.is_empty(),
            "E1: baseline arm touched the split seam — blocking eval must stay the only path"
        );

        model.set_decode_async_eval(true);
        assert!(model.decode_async_eval, "E1: armed toggle did not take");
        for (index, prompt) in prompts.iter().enumerate() {
            model.test_split_trace.clear();
            let got = run_greedy(&mut model, prompt, 128)?;
            let trace = &model.test_split_trace;
            assert!(
                !trace.is_empty() && trace.len() % 2 == 0,
                "prompt {index}: armed arm never fired the eval/sync split — empty-arm (doc §Gate)"
            );
            for pair in trace.chunks(2) {
                assert_eq!(
                    pair,
                    &["async_eval", "synchronize"][..],
                    "prompt {index}: split call order must be async_eval then synchronize"
                );
            }
            assert_eq!(
                got, expected[index],
                "prompt {index}: async-eval split diverged from baseline — INVALID (doc §Gate.1)"
            );
        }

        model.set_decode_async_eval(false);
        set_decode_async_eval(false);
        set_sync_schedule(false);
        Ok(())
    }

    /// E2 — budget and cancel edges (doc §9 E2) with the split armed: budgets
    /// 1 and 8 and a cancel after the second publish must publish the same
    /// prefix and land on the same position as baseline, and every armed arm
    /// must have fired the seam. Cancel is after the **second** publish, not
    /// the first the pre-registration names: under the sync schedule the first
    /// decode forward — the only call that reaches the split — runs when
    /// producing the second token, so cancel-after-first would publish only
    /// prefill logits and never touch the mechanism.
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn e2_budget_and_cancel_edges_match_with_split() -> Result<()> {
        let prompt = pseudo_prompt(16, 741);
        set_sync_schedule(true);
        set_decode_async_eval(false);
        let mut model = load_checkpoint_model()?;
        model.set_decode_async_eval(false);

        for budget in [1usize, 8] {
            let baseline = run_greedy(&mut model, &prompt, budget)?;
            let baseline_pos = model.position();
            model.set_decode_async_eval(true);
            model.test_split_trace.clear();
            let armed = run_greedy(&mut model, &prompt, budget)?;
            let armed_pos = model.position();
            let fired = !model.test_split_trace.is_empty();
            model.set_decode_async_eval(false);
            if budget > 1 {
                // Budget 1 publishes only prefill logits, so no decode forward
                // runs and the seam cannot fire — token/position equality is
                // the whole assertion for that edge.
                assert!(fired, "E2 budget {budget}: split seam never fired");
            }
            assert_eq!(armed, baseline, "budget {budget}: published tokens");
            assert_eq!(armed_pos, baseline_pos, "budget {budget}: final position");
            assert_eq!(
                armed_pos,
                prompt.len() + armed.len() - 1,
                "budget {budget}: position must be P+N-1"
            );
        }

        // Cancel after the second publish (see doc comment): one decode
        // forward has run, so the seam must have fired, and both arms publish
        // the same prefix at the same position.
        let options = GenerationOptions::greedy(16, None);
        let cancel = GenerationCancellation::default();
        let mut baseline_published = Vec::new();
        let flag = cancel.clone();
        model.generate_streaming_cancellable_dyn(
            crate::llm_trait::GenerationRequest {
                input: LlmInput::text(&prompt),
                options: &options,
            },
            &cancel,
            &mut |token| {
                baseline_published.push(token.token_id);
                if baseline_published.len() >= 2 {
                    flag.cancel();
                }
            },
        )?;
        let baseline_pos = model.position();

        model.set_decode_async_eval(true);
        model.test_split_trace.clear();
        let cancel = GenerationCancellation::default();
        let mut armed_published = Vec::new();
        let flag = cancel.clone();
        model.generate_streaming_cancellable_dyn(
            crate::llm_trait::GenerationRequest {
                input: LlmInput::text(&prompt),
                options: &options,
            },
            &cancel,
            &mut |token| {
                armed_published.push(token.token_id);
                if armed_published.len() >= 2 {
                    flag.cancel();
                }
            },
        )?;
        let armed_pos = model.position();
        let fired = !model.test_split_trace.is_empty();
        model.set_decode_async_eval(false);
        assert!(fired, "E2 cancel: split seam never fired");
        assert_eq!(armed_published, baseline_published, "cancel: published prefix");
        assert_eq!(armed_pos, baseline_pos, "cancel: final position");
        assert_eq!(
            armed_pos,
            prompt.len() + armed_published.len() - 1,
            "cancel: position must be P+N-1"
        );

        set_decode_async_eval(false);
        set_sync_schedule(false);
        Ok(())
    }

    /// E3 — continuation state (doc §9 E3): after an armed sync run, every KV
    /// row `[0, position)` and all conv windows are byte-identical to the
    /// baseline run's, and a 16-token continuation matches token-for-token.
    /// The sync schedule has no lookahead step to roll back, so the state the
    /// split hands to a sequel is the committed request state itself.
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn e3_continuation_state_matches_baseline_with_split() -> Result<()> {
        let prompt = pseudo_prompt(24, 751);
        let take = 12usize;
        set_sync_schedule(true);
        set_decode_async_eval(false);

        let mut model = load_checkpoint_model()?;
        model.set_decode_async_eval(false);
        let baseline_ids = run_greedy(&mut model, &prompt, take)?;
        let baseline_pos = model.position();
        assert_eq!(
            baseline_pos,
            prompt.len() + take - 1,
            "E3: baseline position must be P+N-1"
        );
        let (baseline_kv, baseline_conv) = state_bytes(&model, baseline_pos)?;
        let mut baseline_cont = Vec::with_capacity(16);
        let mut prev = baseline_ids[take - 1];
        for _ in 0..16 {
            let logits = model.forward(&[prev], model.position() as u32)?;
            prev = greedy_from_logits(&model, &logits)?;
            baseline_cont.push(prev);
        }

        model.set_decode_async_eval(true);
        model.test_split_trace.clear();
        let armed_ids = run_greedy(&mut model, &prompt, take)?;
        let armed_pos = model.position();
        let fired = !model.test_split_trace.is_empty();
        assert!(fired, "E3: split seam never fired on the armed arm");
        assert_eq!(armed_ids, baseline_ids, "E3: armed tokens diverged from baseline");
        assert_eq!(armed_pos, baseline_pos, "E3: armed position diverged from baseline");
        let (armed_kv, armed_conv) = state_bytes(&model, armed_pos)?;
        assert_eq!(
            armed_kv, baseline_kv,
            "E3: KV rows [0, position) must match baseline byte-for-byte"
        );
        assert_eq!(
            armed_conv, baseline_conv,
            "E3: conv windows must match baseline byte-for-byte"
        );
        let mut armed_cont = Vec::with_capacity(16);
        let mut prev = armed_ids[take - 1];
        for _ in 0..16 {
            let logits = model.forward(&[prev], model.position() as u32)?;
            prev = greedy_from_logits(&model, &logits)?;
            armed_cont.push(prev);
        }
        assert_eq!(
            armed_cont, baseline_cont,
            "E3: 16-token continuation after the armed run must match baseline"
        );

        model.set_decode_async_eval(false);
        set_decode_async_eval(false);
        set_sync_schedule(false);
        Ok(())
    }

    /// E4 — empty-arm and ordering (doc §9 E4). Unset: the model captured the
    /// split off, `decode_async_eval_enabled()` (banner source) is false, and
    /// a full sync request leaves the seam untouched. Armed: every seam pair
    /// is `async_eval` then `synchronize`, the `eval_calls` delta covers the
    /// seam firings, and `trace_callbacks` is unchanged (same eager graph, no
    /// compile). Finally, under the **pipelined** schedule the seam stays
    /// empty even with the toggle on: the split lives in `execute_inner`,
    /// which `GreedySequence`/`DeviceDecode` never calls — the executable
    /// record of the schedule/arm mismatch noted in STATUS, which the window
    /// must resolve (sync-schedule arms, or move the mechanism) before any
    /// measurement is meaningful.
    #[test]
    #[ignore = "requires shared Metal qualification lock and APXINF_LFM3_TEST_MODEL"]
    fn e4_empty_arm_and_split_ordering() -> Result<()> {
        set_sync_schedule(true);
        set_decode_async_eval(false);
        assert!(
            !decode_async_eval_enabled(),
            "E4: helper must report false when the env is unset"
        );
        let mut model = load_checkpoint_model()?;
        assert!(
            !model.decode_async_eval,
            "E4: model captured the split while the env was unset"
        );
        assert!(model.test_split_trace.is_empty());

        let prompt = pseudo_prompt(24, 731);
        let ids = run_greedy(&mut model, &prompt, 32)?;
        assert_eq!(ids.len(), 32, "E4: budget must be honoured");
        assert!(
            model.test_split_trace.is_empty(),
            "E4: the split branch ran with the switch unset"
        );

        let stream = model.backend.stream().clone();
        let before = stream.counters();
        model.set_decode_async_eval(true);
        model.test_split_trace.clear();
        run_greedy(&mut model, &prompt, 32)?;
        let after = stream.counters();
        let trace = &model.test_split_trace;
        assert!(!trace.is_empty(), "E4: armed sync arm never fired the split");
        for pair in trace.chunks(2) {
            assert_eq!(
                pair,
                &["async_eval", "synchronize"][..],
                "E4: async_eval must be called before synchronize"
            );
        }
        let firings = trace.len() / 2;
        assert!(
            after.eval_calls - before.eval_calls >= firings,
            "E4: eval_calls delta ({}) is below the seam firings ({firings}) — counters and seam disagree",
            after.eval_calls - before.eval_calls
        );
        assert_eq!(
            after.trace_callbacks, before.trace_callbacks,
            "E4: the split must not add trace callbacks (no compile)"
        );
        model.set_decode_async_eval(false);

        // Pipelined schedule: same toggle, but `execute_inner` is never
        // reached, so the seam stays empty — the switch is a no-op there.
        set_sync_schedule(false);
        model.set_decode_async_eval(true);
        model.test_split_trace.clear();
        run_greedy(&mut model, &prompt, 32)?;
        assert!(
            model.test_split_trace.is_empty(),
            "E4: pipelined path fired the execute_inner split — mechanism location changed; revisit the arm schedule"
        );
        model.set_decode_async_eval(false);
        set_decode_async_eval(false);
        Ok(())
    }
}
