//! Native LFM3 forward and request-owned state (prefill, decode and KV/conv caches).
//!
//! The execution recipe is a line-by-line reproduction of the verified reference
//! path, because that path is the acceptance oracle: sandwiched BF16 weighted
//! RMSNorm, explicit FP32 zero-tail rotary tables, FP32 depthwise convolution
//! with a six-column input history, per-head sigmoid output gate, and a dense KV
//! row for full-attention layers with a 256-token read window for sliding ones.
//! No fused variants are selected here yet; every operation names the backend
//! primitive the reference itself calls.

use super::{
    config::{LayerKind, Lfm3Config},
    weights::{LayerWeights, Mixer, Weights},
};
use crate::llm_trait::{LlmInput, LlmTrait, TextPreparationState, TextPreparationStatus};
use apxinf_core::{Backend, Device, Error, Result, Tensor};
use apxinf_loader::ModelConfig;
use apxinf_mlx::{Array, MlxBackend, MlxDType, Stream};
use std::{collections::HashMap, sync::Arc};

fn err(message: impl Into<String>) -> Error {
    Error::Other(message.into())
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

/// `silu(gate) * up` expressed exactly as the reference's `nn.silu(g) * u`.
fn silu_mul(gate: &Array, up: &Array) -> Result<Array> {
    gate.mul(&gate.sigmoid()?)?.mul(up)
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

pub struct Lfm3Model {
    config: Lfm3Config,
    backend: MlxBackend,
    weights: Weights,
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
        let kinds = config.layer_kinds()?;
        let layers = config.num_hidden_layers;
        let mut model = Self {
            config,
            backend,
            weights,
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
        };
        model.prepare(1, 0)?;
        Ok(model)
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
        let Mixer::Conv {
            in_proj,
            out_proj,
            taps,
        } = &layer.mixer
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
        let conv = z_stream
            .cast(MlxDType::F32)?
            .conv1d(&weight, 1, taps_kernel - 1, 1, h)?;
        // conv1d pads both sides; outputs [stream-len, stream) are this chunk.
        let y = gate.mul(
            &conv
                .slice_axis(1, stream_len - len, stream_len)?
                .cast(MlxDType::BF16)?,
        )?;
        let op = out_proj.call(&y)?;
        let first = stream_len.saturating_sub(keep);
        let mut next = z_stream
            .slice_axis(1, first, stream_len)?
            .transpose(&[0, 2, 1])?;
        let width = next.shape()[2];
        if width < keep {
            next = next.pad(&[0, 0, keep - width], &[0, 0, 0])?;
        }
        Ok((op, next))
    }

    /// One-token short-conv step: seven-tap FP32 accumulation over the retained
    /// history plus the new input column.
    fn conv_step(&self, n: &Array, layer: &LayerWeights, state: &Array) -> Result<(Array, Array)> {
        let Mixer::Conv {
            in_proj,
            out_proj,
            taps,
        } = &layer.mixer
        else {
            return Err(err("LFM3 conv layer has an attention weight plan"));
        };
        let c = &self.config;
        let h = c.hidden_size;
        let kernel = c.conv_l_cache;
        let bcx = in_proj.call(n)?;
        let b = bcx.slice_axis(2, 0, h)?;
        let gate = bcx.slice_axis(2, h, 2 * h)?;
        let z = bcx.slice_axis(2, 2 * h, 3 * h)?;
        let z = b.mul(&z)?.transpose(&[0, 2, 1])?;
        let window = Array::concat(&[state, &z], 2)?;
        // Every operand of the accumulation is `[1, hidden, 1]`; adding a
        // `[1, 1, hidden]` accumulator to it would broadcast into `[1, h, h]`.
        let mut conv = Array::zeros(self.backend.stream(), &[1, h, 1], MlxDType::F32)?;
        for tap in 0..kernel {
            let weight = taps.slice_axis(1, tap, tap + 1)?.reshape(&[1, h, 1])?;
            conv = conv.add(&weight.mul(&window.slice_axis(2, tap, tap + 1)?)?)?;
        }
        let conv = conv.transpose(&[0, 2, 1])?;
        let y = gate.mul(&conv.cast(MlxDType::BF16)?)?;
        let op = out_proj.call(&y)?;
        Ok((op, window.slice_axis(2, 1, kernel)?))
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
        let q = q_proj
            .call(n)
            .map_err(|e| err(format!("q_proj: {e}")))?
            .reshape(&[1, 1, heads, dim])
            .map_err(|e| err(format!("q reshape n={:?}: {e}", n.shape())))?;
        let q = norm(&q, q_norm, c.norm_eps)?.transpose(&[0, 2, 1, 3])?;
        let k = k_proj
            .call(n)?
            .reshape(&[1, 1, c.num_key_value_heads, dim])
            .map_err(|e| err(format!("k reshape: {e}")))?;
        let k = norm(&k, k_norm, c.norm_eps)?.transpose(&[0, 2, 1, 3])?;
        let v = v_proj
            .call(n)?
            .reshape(&[1, 1, c.num_key_value_heads, dim])
            .map_err(|e| err(format!("v reshape: {e}")))?
            .transpose(&[0, 2, 1, 3])?;
        let (cos, sin) = self
            .cos_sin(kind, start, 1)
            .map_err(|e| err(format!("rope slice start={start}: {e}")))?;
        let q = apply_rope(&q, &cos, &sin, dim)?;
        let k = apply_rope(&k, &cos, &sin, dim)?;
        let index = Array::from_i32(self.backend.stream(), &[1], &[start as i32])?;
        let keys = kv.keys.slice_update(&k, &index, &[2])?;
        let values = kv.values.slice_update(&v, &index, &[2])?;
        let end = start + 1;
        let low = match kind {
            LayerKind::SlidingAttention => end.saturating_sub(c.window()),
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
    fn mix_and_residual(&self, x: &Array, op: &Array, layer: &LayerWeights) -> Result<Array> {
        let eps = self.config.norm_eps;
        let op = norm(op, &layer.post_operator_norm, eps)?;
        let hidden = x.add(&op)?;
        let ffn = norm(&hidden, &layer.ffn_norm, eps)?;
        let gate = layer.w1.call(&ffn)?;
        let up = layer.w3.call(&ffn)?;
        let activated = silu_mul(&gate, &up)?;
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
        Ok(output)
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

    fn execute_inner(&mut self, token_ids: &[u32], last_only: bool) -> Result<Tensor> {
        let stream = self.backend.stream().clone();
        let len = token_ids.len();
        let start = self.position;
        let end = start + len;
        let ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
        let ids = Array::from_i32(&stream, &[len], &ids)?;
        let mut x = self
            .weights
            .embed(&ids)?
            .reshape(&[1, len, self.config.hidden_size])?;
        // A prompt chunk evaluates every layer boundary (bounded peak memory);
        // a single-token step keeps one graph and evaluates once below.
        let per_layer_eval = len > 1;
        for layer_idx in 0..self.config.num_hidden_layers {
            x = self
                .run_layer(&x, layer_idx, start, len, per_layer_eval)
                .map_err(|e| err(format!("layer {layer_idx}: {e}")))?;
        }
        let hidden = norm(&x, &self.weights.final_norm, self.config.norm_eps)
            .map_err(|e| err(format!("final norm: {e}")))?;
        let rows = if last_only {
            hidden.slice_axis(1, len - 1, len)?
        } else {
            hidden
        };
        let logits = self
            .weights
            .project(&rows)
            .map_err(|e| err(format!("head projection: {e}")))?
            .reshape(&[rows.shape()[1], self.config.vocab_size])
            .map_err(|e| err(format!("head reshape rows={:?}: {e}", rows.shape())))?;
        let mut required: Vec<Array> = vec![logits.clone()];
        if !per_layer_eval {
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
        }
        stream.eval(&required)?;
        self.position = end;
        self.backend.from_array(logits)
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
        self.position = 0;
        self.invalid = None;
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
        self.backend.stream().synchronize()
    }

    /// Keep the model-owned position capacity and rotary tables, drop only the
    /// request's KV/conv content.
    fn release_request_state_retaining_preparation(&mut self) -> Result<()> {
        self.backend.stream().synchronize()?;
        self.kv = self.initial_kv.clone();
        self.conv = self.initial_conv.clone();
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
