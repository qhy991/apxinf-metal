//! Native d1-2B text forward: one packed sequence in, final-normed hidden
//! states out. No token generation, no KV cache — the D1 readout is a single
//! prefill pass per sequence.
//!
//! Provenance: the layer arithmetic is copied from `metal_lfm3`
//! (`origin/feat/lfm3-metal` @ 1ed3f2f3), whose LFM3-1.7B text stack is
//! field-for-field the d1-2B `text_config`. The d1 change is the attention
//! program: the checkpoint's `d1_attention = "set"` makes every
//! full-attention layer bidirectional per the reference's `Sees` mask
//! (a state token sees the whole packed row; a question token sees the state
//! and its own question), while sliding-window and convolution layers stay
//! causal, exactly as the reference keeps them.
//!
//! The execution recipe reproduces the verified reference path: sandwiched
//! BF16 weighted RMSNorm, explicit FP32 zero-tail rotary tables built with
//! the platform libm, FP32 depthwise convolution, per-head sigmoid output
//! gate, and the tied embedding as the readout projection.

use super::{
    config::{D1Config, LayerKind, MAX_SEQUENCE},
    weights::{LayerWeights, Mixer, Weights},
};
use apxinf_core::{Backend, Error, Result, Tensor};
use apxinf_mlx::{Array, MlxBackend, MlxDType};
use std::collections::HashMap;
use std::sync::Arc;

fn err(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

// Platform C math library, used only to generate the fixed rotary tables on
// the host. Declared here instead of relying on Rust's `f32::sin`/`f32::cos`,
// which are correctly rounded and therefore not bit-identical to the
// reference's float32 numpy path (see `rope_table`).
extern "C" {
    #[link_name = "cosf"]
    fn libm_cosf(x: f32) -> f32;
    #[link_name = "sinf"]
    fn libm_sinf(x: f32) -> f32;
}

/// Weighted RMSNorm over the last axis on BF16 activations.
fn norm(x: &Array, weight: &Array, epsilon: f32) -> Result<Array> {
    if weight.dtype() != MlxDType::BF16 || x.dtype() != MlxDType::BF16 {
        return Err(err("d1 RMSNorm requires BF16 activations and weights"));
    }
    x.fast_rms_norm(weight, epsilon)
}

/// HF `rotate_half` over the last axis: `[x1, x2] -> [-x2, x1]`.
fn rotate_half(x: &Array, dim: usize) -> Result<Array> {
    let half = dim / 2;
    Array::concat(
        &[&x.slice_axis(3, half, dim)?.neg()?, &x.slice_axis(3, 0, half)?],
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

/// The set-attention program of one packed sequence (the reference's
/// `pack_set` output): `state_len` tokens of state followed by questions,
/// each token tagged with its question (1..=N) or -1 for the state.
pub struct SetAttention {
    pub state_len: usize,
    pub question: Vec<i32>,
}

pub struct D1Model {
    config: D1Config,
    backend: MlxBackend,
    weights: Weights,
    kinds: Vec<LayerKind>,
}

impl D1Model {
    pub fn new(
        config: D1Config,
        tensors: HashMap<String, Tensor>,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let backend = backend
            .as_any()
            .downcast_ref::<MlxBackend>()
            .ok_or_else(|| err("d1 native implementation requires an MLX backend"))?
            .clone();
        let weights = Weights::load(&config.text_config, tensors, backend.stream())?;
        let kinds = config.text_config.layer_kinds()?;
        Ok(Self {
            config,
            backend,
            weights,
            kinds,
        })
    }

    pub fn config(&self) -> &D1Config {
        &self.config
    }

    pub fn stream(&self) -> &apxinf_mlx::Stream {
        self.backend.stream()
    }

    /// `cos`/`sin` tables `[len, head_dim]` in FP32.
    ///
    /// The angle table is `concat(freqs, freqs)` like the reference, so the
    /// zero-tail frequencies arrive as `cos=1, sin=0` and the same `rotate()`
    /// formula is exact for both rotary schemes.
    fn rope_table(&self, kind: LayerKind, len: usize, cosine: bool) -> Result<Array> {
        let config = &self.config.text_config;
        let scheme = config.rope(kind)?;
        let dim = config.head_dim;
        let half = dim / 2;
        let inv_freq = scheme.inv_freq(dim);
        // The table is built on the host in FP32 with the platform `sinf`/`cosf`.
        // That is what the reference's `np.sin`/`np.cos` on a float32 angle table
        // compiles to on this platform, and matching it bit-for-bit matters:
        // Rust's `f32::sin`/`f32::cos` are correctly rounded and differ from the
        // platform libm by one ULP on ~5-12% of entries, which is enough to flip
        // a BF16 rounding of `q`/`k` and lose parity in the first sliding layer.
        let mut table = vec![0f32; len * dim];
        for position in 0..len {
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
        Array::from_f32(self.backend.stream(), &[len, dim], &table)
    }

    fn cos_sin(&self, kind: LayerKind, len: usize) -> Result<(Array, Array)> {
        let dim = self.config.text_config.head_dim;
        Ok((
            self.rope_table(kind, len, true)?.reshape(&[1, 1, len, dim])?,
            self.rope_table(kind, len, false)?.reshape(&[1, 1, len, dim])?,
        ))
    }

    /// Boolean mask `[1, 1, q_len, k_len]` over absolute positions.
    fn band_mask(
        &self,
        q_len: usize,
        k_len: usize,
        q_start: usize,
        k_start: usize,
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

    /// The d1 set-attention mask `[1, 1, T, T]` for full-attention layers:
    /// a query attends to a key when the query is a state token (`q < s`),
    /// the key is a state token (`k < s`), or both carry the same question
    /// (`question[q] == question[k]`). For a lone question this is all-true,
    /// exactly as the reference reads one question both ways.
    fn set_mask(&self, set: &SetAttention) -> Result<Array> {
        let stream = self.backend.stream();
        let len = set.question.len();
        let g = Array::from_i32(stream, &[len], &set.question)?;
        let s = Array::scalar(stream, set.state_len as f32, MlxDType::I32)?;
        let q_pos = Array::arange(stream, 0., len as f32, 1., MlxDType::I32)?.reshape(&[len, 1])?;
        let k_pos = Array::arange(stream, 0., len as f32, 1., MlxDType::I32)?.reshape(&[1, len])?;
        // `q < s` and `k < s` as integer comparisons: `q + 1 <= s`.
        let one = Array::scalar(stream, 1., MlxDType::I32)?;
        let q_state = q_pos.add(&one)?.less_equal(&s)?;
        let k_state = k_pos.add(&one)?.less_equal(&s)?;
        let same = g
            .reshape(&[len, 1])?
            .equal(&g.reshape(&[1, len])?)?;
        // The union as `1 <= q_state + k_state + same` (integer comparison).
        let one = Array::scalar(stream, 1., MlxDType::I32)?;
        let mask = one.less_equal(
            &q_state
                .cast(MlxDType::I32)?
                .add(&k_state.cast(MlxDType::I32)?)?
                .add(&same.cast(MlxDType::I32)?)?,
        )?
        .reshape(&[1, 1, len, len])?;
        Ok(mask)
    }

    /// Short-conv branch over a fresh sequence (no history): the reference's
    /// `causal_conv1d_fn` pads `kernel - 1` zeros on the left and convolves.
    fn conv_layer(&self, n: &Array, layer: &LayerWeights) -> Result<Array> {
        let Mixer::Conv {
            in_proj,
            out_proj,
            taps,
        } = &layer.mixer
        else {
            return Err(err("d1 conv layer has an attention weight plan"));
        };
        let c = &self.config.text_config;
        let h = c.hidden_size;
        let taps_kernel = c.conv_l_cache;
        let len = n.shape()[1];
        let bcx = in_proj.call(n)?;
        let b = bcx.slice_axis(2, 0, h)?;
        let gate = bcx.slice_axis(2, h, 2 * h)?;
        let z = bcx.slice_axis(2, 2 * h, 3 * h)?;
        let z = b.mul(&z)?;
        let weight = taps.reshape(&[h, taps_kernel, 1])?;
        let conv = z
            .cast(MlxDType::F32)?
            .conv1d(&weight, 1, taps_kernel - 1, 1, h)?;
        // conv1d pads both sides; the first `len` outputs are the causal part.
        let y = gate.mul(
            &conv
                .slice_axis(1, 0, len)?
                .cast(MlxDType::BF16)?,
        )?;
        out_proj.call(&y)
    }

    /// Attention branch over a fresh sequence. Full-attention layers run the
    /// d1 set-attention mask; sliding layers stay causal within their window.
    fn attention_layer(
        &self,
        n: &Array,
        layer: &LayerWeights,
        kind: LayerKind,
        len: usize,
        set_mask: Option<&Array>,
    ) -> Result<Array> {
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
            return Err(err("d1 attention layer has a conv weight plan"));
        };
        let c = &self.config.text_config;
        let heads = c.num_attention_heads;
        let dim = c.head_dim;
        let scale = (dim as f64).powf(-0.5) as f32;
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
        let (cos, sin) = self.cos_sin(kind, len)?;
        let q = apply_rope(&q, &cos, &sin, dim)?;
        let k = apply_rope(&k, &cos, &sin, dim)?;
        let attention = match kind {
            LayerKind::FullAttention => {
                // A lone question reads the whole row both ways: no mask at all.
                q.sdpa(&k, &v, scale, false, set_mask)?
            }
            LayerKind::SlidingAttention => {
                let window = c.window();
                if len <= window {
                    q.sdpa(&k, &v, scale, len > 1, None)?
                } else {
                    self.windowed_sdpa(&q, &k, &v, len, window, scale)?
                }
            }
            LayerKind::Conv => return Err(err("d1 conv layer has no attention")),
        };
        // Per-head sigmoid output gate: `[1, len, heads] -> [1, heads, len, 1]`.
        let gate = gate.call(n)?.sigmoid()?.transpose(&[0, 2, 1])?;
        let gated = attention.mul(&gate.reshape(&[1, heads, len, 1])?)?;
        let merged = gated
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, len, heads * dim])?;
        o_proj.call(&merged)
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
            let mask = self.band_mask(end - start, end - low, start, low, Some(window))?;
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

    /// Sandwich-norm residual composition shared by both branches.
    fn mix_and_residual(&self, x: &Array, op: &Array, layer: &LayerWeights) -> Result<Array> {
        let eps = self.config.text_config.norm_eps;
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

    /// One packed sequence forward. Returns the final-normed hidden states
    /// `[1, T, hidden]` — the D1 readout gathers it at the answer slots.
    pub fn forward_hidden(&mut self, token_ids: &[u32], set: Option<&SetAttention>) -> Result<Array> {
        let config = &self.config.text_config;
        let len = token_ids.len();
        if len == 0 || len > MAX_SEQUENCE {
            return Err(err(format!(
                "d1 forward requires a nonempty sequence of at most {MAX_SEQUENCE} tokens, got {len}"
            )));
        }
        if token_ids.iter().any(|&id| id as usize >= config.vocab_size) {
            return Err(err("d1 forward got an out-of-vocabulary token id"));
        }
        if let Some(set) = set {
            if set.question.len() != len || set.state_len > len {
                return Err(err(
                    "d1 set attention requires one question tag per token and state_len <= len",
                ));
            }
            if set.question.iter().take(set.state_len).any(|&g| g != -1)
                || set.question.iter().skip(set.state_len).any(|&g| g < 1)
            {
                return Err(err(
                    "d1 set attention requires -1 on the state and positive question ids after it",
                ));
            }
        }
        let stream = self.backend.stream().clone();
        let ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
        let ids = Array::from_i32(&stream, &[len], &ids)?;
        let mut x = self
            .weights
            .embed(&ids)?
            .reshape(&[1, len, config.hidden_size])?;
        let set_mask = match set {
            Some(set) => Some(self.set_mask(set)?),
            None => None,
        };
        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let kind = self.kinds[layer_idx];
            let n = norm(&x, &layer.operator_norm, config.norm_eps)
                .map_err(|e| err(format!("layer {layer_idx} operator norm: {e}")))?;
            let op = if kind.is_conv() {
                self.conv_layer(&n, layer)
            } else {
                let mask = match kind {
                    LayerKind::FullAttention => set_mask.as_ref(),
                    _ => None,
                };
                self.attention_layer(&n, layer, kind, len, mask)
            }
            .map_err(|e| err(format!("layer {layer_idx} ({}): {e}", kind.name())))?;
            x = self
                .mix_and_residual(&x, &op, layer)
                .map_err(|e| err(format!("layer {layer_idx} residual: {e}")))?;
        }
        norm(&x, &self.weights.final_norm, config.norm_eps)
    }

    /// Project hidden rows through the tied embedding head: `[rows, vocab]`.
    pub fn project(&self, hidden_rows: &Array) -> Result<Array> {
        self.weights.project(hidden_rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_table_has_the_zero_tail() {
        // The full-attention scheme rotates 64 of 128 dims: the tail is identity.
        let config_json = r##"{
            "model_type": "lfm3_vl",
            "bos_token_id": 124894, "downsample_factor": 2, "eos_token_id": 124900,
            "image_token_id": 125030, "pad_token_id": 124893, "projector_bias": true,
            "projector_hidden_act": "gelu", "projector_hidden_size": 4096,
            "projector_layer_norm_eps": 1e-06, "tie_word_embeddings": true,
            "video_token_id": 125029, "vision_end_token_id": 125010, "vision_start_token_id": 125009,
            "d1_codes": 2, "d1_attention": "set", "d1_numbers": true, "d1_set_size": 16,
            "text_config": {
                "model_type": "lfm3", "hidden_size": 2048, "intermediate_size": 7680,
                "num_hidden_layers": 24, "num_attention_heads": 16, "num_key_value_heads": 4,
                "head_dim": 128, "vocab_size": 128000, "norm_eps": 1e-5, "conv_L_cache": 7,
                "conv_bias": false, "sliding_window": 256,
                "layer_types": ["conv","conv","full_attention","conv","sliding_attention",
                                "conv","full_attention","conv","sliding_attention","conv",
                                "full_attention","conv","sliding_attention","conv",
                                "full_attention","conv","sliding_attention","conv",
                                "full_attention","conv","sliding_attention","conv",
                                "full_attention","conv"],
                "use_sandwich_norm": true, "tie_word_embeddings": true,
                "rope_parameters": {
                    "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"},
                    "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}
                }
            },
            "vision_config": {
                "model_type": "lfm3_vl_vision", "hidden_size": 1024, "intermediate_size": 2560,
                "num_hidden_layers": 14, "num_attention_heads": 16, "num_key_value_heads": 8,
                "conv_kernel_size": 3, "in_channels": 3, "patch_size": 16, "temporal_patch_size": 2,
                "spatial_merge_size": 2, "norm_eps": 1e-5,
                "layer_types": ["conv","conv","full_attention","conv","full_attention","conv",
                                "full_attention","conv","full_attention","conv","full_attention",
                                "conv","full_attention","conv"],
                "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "axial"}
            }
        }"##;
        let config = D1Config::from_json(config_json).unwrap();
        let full = config.text_config.rope(LayerKind::FullAttention).unwrap();
        let inv = full.inv_freq(config.text_config.head_dim);
        assert_eq!(inv.len(), 64);
        assert!(inv[32..].iter().all(|&value| value == 0.0));
        assert!(inv[..32].iter().all(|&value| value > 0.0));
    }
}
