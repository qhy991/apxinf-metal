//! d3-lite text forward on native MLX/Metal: one lazy request graph, compiled
//! per admitted length, evaluated once. No decode and no KV cache — the GDN
//! recurrence state and the attention K/V are created and consumed inside a
//! single request and never survive it.
//!
//! Semantics follow the checkpoint reference (`modeling_d3.py`) and the
//! EngineTailor MLX executor op for op, including every rounding point:
//! `1 + w` norms, the depthwise causal conv over three zero rows, the official
//! GDN q/k l2-norm (sum epsilon `1e-6`, not the mlx-lm mean-epsilon variant),
//! the FP32 gated norm in the GDN block, non-causal SDPA over the full prompt,
//! the output gate on the attention path, and the FP32 255-code readout at the
//! last position.
//!
//! The apxinf kernels are the maintained device implementations of the same
//! primitives: `GatedDeltaRecurrence` (four value rows per SIMD group) and
//! `HalfSplitRope` (FP32 angles, one rounding to the input dtype). The causal
//! convolution is composed from MLX `conv1d` with an explicit zero history;
//! an ApxInf causal-conv kernel exists only in the private EngineTailor
//! reference and is a P3 candidate, not part of this unoptimized baseline.

use std::cell::RefCell;
use std::rc::Rc;

use apxinf_core::{Error, Result};
use apxinf_mlx::fusions::{GatedDeltaRecurrence, HalfSplitRope};
use apxinf_mlx::{Array, Compiled, MlxDType, Stream};

use super::config::{Config, Geometry, MAX_INPUT_TOKENS};
use super::weights::{Attention, GatedDelta, Mlp, Mixer, Weights};

const EPS: f32 = 1e-6;

/// Everything the compiled callable captures: immutable geometry, evaluated
/// weights, registered kernels and the request-independent inverse
/// frequencies. `Rc` lets the trace closure and the runner share one copy
/// without introducing a second owner for device arrays.
struct Parts {
    geometry: Geometry,
    weights: Weights,
    recurrence: GatedDeltaRecurrence,
    rope: HalfSplitRope,
    inv_freq: Array,
    stream: Stream,
}

/// One d3-lite execution over shared weights. `compiled` is a whole-request
/// callable (`mx.compile` over the forward, inputs `ids [1, L]` and
/// `count [1]`, output `probabilities [1, 255]`); `prepared` records the exact
/// `(length, count)` profiles that have been warmed with `Compiled::prepare`,
/// which is the only operation that may trace.
pub struct Model {
    parts: Rc<Parts>,
    compiled: Compiled,
    prepared: RefCell<Vec<(usize, usize)>>,
}

fn scalar(like: &Array, value: f64) -> Result<Array> {
    Array::scalar(like.stream(), value as f32, like.dtype())
}

/// `x * sigmoid(x)`.
fn silu(x: &Array) -> Result<Array> {
    x.mul(&x.sigmoid()?)
}

/// `nn.Linear` without bias: `x @ W.T` (`W` is `[out, in]` as stored).
fn project(x: &Array, weight: &Array) -> Result<Array> {
    x.matmul(&weight.transpose(&[1, 0])?)
}

/// `compute_g`: `exp(-exp(A_log_f32) * softplus(a + dt_bias))`; beta:
/// `sigmoid(b)`. `A_log` is promoted to F32 before `exp`, exactly as the
/// vendored reference does; the softplus stays in the operand dtype (BF16).
fn decay_beta(a_log: &Array, a: &Array, b: &Array, dt_bias: &Array) -> Result<(Array, Array)> {
    let shifted = a.add(dt_bias)?;
    let softplus = shifted.logaddexp(&scalar(&shifted, 0.0)?)?;
    let decay = a_log
        .cast(MlxDType::F32)?
        .exp()?
        .neg()?
        .mul(&softplus)?
        .exp()?;
    Ok((decay, b.sigmoid()?))
}

impl Model {
    pub fn new(config: &Config, stream: Stream, weights: Weights) -> Result<Self> {
        let geometry = config.geometry;
        let parts = Rc::new(Parts {
            geometry,
            weights,
            recurrence: GatedDeltaRecurrence::new(&stream)?,
            rope: HalfSplitRope::new(&stream, geometry.rope_dims)?,
            inv_freq: inverse_frequencies(&stream, geometry)?,
            stream: stream.clone(),
        });
        let trace_parts = parts.clone();
        let compiled = Compiled::with_shapeless(&stream, 1, false, move |inputs| {
            let [ids, count] = inputs else {
                return Err(Error::Contract(
                    "d3-lite compiled forward takes ids and candidate count",
                ));
            };
            forward(&trace_parts, ids, count)
        })?;
        Ok(Self {
            parts,
            compiled,
            prepared: RefCell::new(Vec::new()),
        })
    }

    fn stream(&self) -> &Stream {
        &self.parts.stream
    }

    fn validate(&self, token_ids: &[u32], count: usize) -> Result<()> {
        let geometry = self.parts.geometry;
        if token_ids.is_empty() {
            return Err(Error::Other(
                "d3-lite decision needs at least one token".into(),
            ));
        }
        if token_ids.len() > geometry.max_positions || token_ids.len() > MAX_INPUT_TOKENS {
            return Err(Error::Other(format!(
                "d3-lite decision length {} exceeds the admitted maximum {MAX_INPUT_TOKENS}",
                token_ids.len()
            )));
        }
        if count == 0 || count > super::config::CODES {
            return Err(Error::Other(format!(
                "d3-lite candidate count {count} is outside 1..={}",
                super::config::CODES
            )));
        }
        if let Some((position, token)) = token_ids
            .iter()
            .enumerate()
            .find(|(_, token)| **token as usize >= geometry.vocab)
        {
            return Err(Error::Other(format!(
                "d3-lite token ID {token} at position {position} is outside vocabulary {}",
                geometry.vocab
            )));
        }
        Ok(())
    }

    fn inputs(&self, token_ids: &[u32], count: usize) -> Result<(Array, Array)> {
        let length = token_ids.len();
        let ids = token_ids
            .iter()
            .map(|&token| {
                i32::try_from(token).map_err(|_| {
                    Error::Other(format!("d3-lite token ID {token} exceeds the embedding index range"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let ids = Array::from_i32(self.stream(), &[1, length], &ids)?;
        let count = Array::from_i32(self.stream(), &[1], &[count as i32])?;
        Ok((ids, count))
    }

    fn is_prepared(&self, length: usize, count: usize) -> bool {
        self.prepared
            .borrow()
            .iter()
            .any(|&(known, known_count)| known == length && known_count == count)
    }

    /// Trace and warm one exact profile. This is the only call that may
    /// compile; `Compiled::prepare` evaluates the profile twice and fails if
    /// the second invocation traces again.
    pub fn prepare_profile(&self, length: usize, count: usize) -> Result<()> {
        let geometry = self.parts.geometry;
        if length == 0 || length > geometry.max_positions || length > MAX_INPUT_TOKENS {
            return Err(Error::Other(format!(
                "d3-lite prepared length {length} is outside 1..={MAX_INPUT_TOKENS}"
            )));
        }
        if count == 0 || count > super::config::CODES {
            return Err(Error::Other(format!(
                "d3-lite prepared candidate count {count} is outside 1..={}",
                super::config::CODES
            )));
        }
        // Zero token IDs keep the warm-up input independent of any request;
        // the traced graph depends on shapes, not values.
        let ids = Array::zeros(self.stream(), &[1, length], MlxDType::I32)?;
        let count_array = Array::from_i32(self.stream(), &[1], &[count as i32])?;
        self.compiled.prepare(&[ids, count_array])?;
        if !self.is_prepared(length, count) {
            self.prepared.borrow_mut().push((length, count));
        }
        Ok(())
    }

    fn prepare_for(&self, token_ids: &[u32], count: usize) -> Result<()> {
        if self.is_prepared(token_ids.len(), count) {
            return Ok(());
        }
        self.prepare_profile(token_ids.len(), count)
    }

    /// One request: prepared compiled execution. The returned vector holds the
    /// first `count` option probabilities, in package code order.
    pub fn probabilities(&self, token_ids: &[u32], count: usize) -> Result<Vec<f32>> {
        self.validate(token_ids, count)?;
        self.prepare_for(token_ids, count)?;
        let (ids, candidate_count) = self.inputs(token_ids, count)?;
        let outputs = self.compiled.call_and_eval(&[ids, candidate_count])?;
        read_probabilities(&outputs[0], count)
    }

    /// Eager lazy-graph execution of the same `forward`, used to compare the
    /// compiled path against its own body and to diagnose a divergence.
    pub fn eager_probabilities(&self, token_ids: &[u32], count: usize) -> Result<Vec<f32>> {
        self.validate(token_ids, count)?;
        let (ids, candidate_count) = self.inputs(token_ids, count)?;
        let outputs = forward(&self.parts, &ids, &candidate_count)?;
        self.stream().eval(&outputs)?;
        read_probabilities(&outputs[0], count)
    }

    /// `(length, count)` profiles warmed so far, in preparation order.
    pub fn prepared_profiles(&self) -> Vec<(usize, usize)> {
        self.prepared.borrow().clone()
    }
}

fn read_probabilities(output: &Array, count: usize) -> Result<Vec<f32>> {
    let values = output.to_f32_vec()?;
    if values.len() != super::config::CODES {
        return Err(Error::Other(format!(
            "d3-lite forward returned {} probabilities, expected {}",
            values.len(),
            super::config::CODES
        )));
    }
    Ok(values[..count].to_vec())
}

/// `inv_freq[i] = exp(-(i / rope_dims) * log(rope_base))` in F32, computed as
/// the reference does (`exp(-exponents * log(base))`) and published as one
/// evaluated constant. The host evaluates this once at load; the compiled
/// graph then captures the array.
fn inverse_frequencies(stream: &Stream, geometry: Geometry) -> Result<Array> {
    let values = (0..geometry.rope_dims)
        .step_by(2)
        .map(|index| {
            let exponent = (index as f32) / (geometry.rope_dims as f32);
            (-(exponent * (geometry.rope_base as f32).ln())).exp()
        })
        .collect::<Vec<_>>();
    let inv_freq = Array::from_f32(stream, &[geometry.rope_dims / 2], &values)?;
    stream.eval(&[inv_freq.clone()])?;
    Ok(inv_freq)
}

/// The whole request graph. `ids` is `[1, L]` I32, `candidate_count` is `[1]`
/// I32; the returned vector is `[probabilities [1, 255]]` in F32, with every
/// code at or past `candidate_count` set to `-inf` before the softmax so the
/// result is exactly the masked softmax the reference produces.
fn forward(parts: &Parts, ids: &Array, candidate_count: &Array) -> Result<Vec<Array>> {
    let geometry = parts.geometry;
    let [1, length] = *ids.shape() else {
        return Err(Error::Contract("d3-lite ids must be [1, L]"));
    };
    if length == 0 {
        return Err(Error::Contract("d3-lite ids must be nonempty"));
    }
    let mut hidden = parts.weights.embed.take(ids, 0)?;
    let positions = Array::arange(
        &parts.stream,
        0.0,
        length as f32,
        1.0,
        MlxDType::I32,
    )?
    .reshape(&[1, length])?;
    for (index, layer) in parts.weights.layers.iter().enumerate() {
        let normalized = hidden.fast_rms_norm(&layer.input_norm, geometry.eps)?;
        let mixed = if geometry.is_attention(index) {
            let Mixer::Attention(attention) = &layer.mixer else {
                unreachable!("the validated schedule selects attention here")
            };
            attention_forward(parts, &normalized, attention, &positions)?
        } else {
            let Mixer::GatedDelta(gated_delta) = &layer.mixer else {
                unreachable!("the validated schedule selects gated delta here")
            };
            gated_delta_forward(parts, &normalized, gated_delta, length)?
        };
        let residual = hidden.add(&mixed)?;
        let mlp_input = residual.fast_rms_norm(&layer.post_norm, geometry.eps)?;
        let mlp_output = mlp_forward(&mlp_input, &layer.mlp)?;
        hidden = residual.add(&mlp_output)?;
    }
    let hidden = hidden.fast_rms_norm(&parts.weights.norm, geometry.eps)?;
    let last = hidden
        .slice_axis(1, length - 1, length)?
        .reshape(&[1, geometry.hidden])?
        .cast(MlxDType::F32)?;
    let logits = last.matmul(&parts.weights.readout.transpose(&[1, 0])?)?;
    let indices = Array::arange(
        &parts.stream,
        0.0,
        super::config::CODES as f32,
        1.0,
        MlxDType::I32,
    )?;
    // `candidate_count <= index` is the "code is not an option" mask.
    let masked = candidate_count.less_equal(&indices)?;
    let negative_infinity = Array::scalar(&parts.stream, f32::NEG_INFINITY, MlxDType::F32)?;
    let logits = masked.where_select(&negative_infinity, &logits)?;
    let probabilities = logits.softmax_with(-1, false)?;
    Ok(vec![probabilities])
}

fn mlp_forward(input: &Array, mlp: &Mlp) -> Result<Array> {
    let gate = project(input, &mlp.gate.weight)?;
    let up = project(input, &mlp.up.weight)?;
    let activated = silu(&gate)?.mul(&up)?;
    project(&activated, &mlp.down.weight)
}

fn gated_delta_forward(
    parts: &Parts,
    input: &Array,
    gated_delta: &GatedDelta,
    length: usize,
) -> Result<Array> {
    let geometry = parts.geometry;
    let key_dim = geometry.key_dim();
    let value_dim = geometry.value_dim();
    let conv_dim = geometry.conv_dim();
    let qkv = project(input, &gated_delta.qkv.weight)?;
    let z = project(input, &gated_delta.z.weight)?
        .reshape(&[1, length, geometry.v_heads, geometry.v_dim])?;
    let b = project(input, &gated_delta.b.weight)?;
    let a = project(input, &gated_delta.a.weight)?;
    // Stock reference: a depthwise causal convolution over `S - 1` zero rows.
    let history = Array::zeros(
        &parts.stream,
        &[1, geometry.conv_kernel - 1, conv_dim],
        qkv.dtype(),
    )?;
    let convolved = Array::concat(&[&history, &qkv], 1)?
        .conv1d(&gated_delta.conv, 1, 0, 1, conv_dim)?;
    let convolved = silu(&convolved)?;
    let queries = convolved
        .slice_axis(2, 0, key_dim)?
        .reshape(&[1, length, geometry.k_heads, geometry.k_dim])?;
    let keys = convolved
        .slice_axis(2, key_dim, 2 * key_dim)?
        .reshape(&[1, length, geometry.k_heads, geometry.k_dim])?;
    let values = convolved
        .slice_axis(2, 2 * key_dim, conv_dim)?
        .reshape(&[1, length, geometry.v_heads, geometry.v_dim])?;
    // Official GDN q/k l2-norm: `x * rsqrt(sum(x^2) + 1e-6)`, written as an
    // RMSNorm whose epsilon is on the mean (`1e-6 / k_dim`), then scaled by
    // `k_dim^-1` for q and `k_dim^-0.5` for k. The mlx-lm variant uses a mean
    // epsilon of `1e-6` and differs enough to move the probabilities; the
    // checkpoint's own model uses this form.
    let mean_epsilon = 1e-6 / geometry.k_dim as f32;
    let queries = queries
        .fast_rms_norm_unweighted(mean_epsilon)?
        .mul(&scalar(&queries, (geometry.k_dim as f64).powf(-1.0))?)?;
    let keys = keys
        .fast_rms_norm_unweighted(mean_epsilon)?
        .mul(&scalar(&keys, (geometry.k_dim as f64).powf(-0.5))?)?;
    let (decay, beta) = decay_beta(&gated_delta.a_log, &a, &b, &gated_delta.dt_bias)?;
    let state = Array::zeros(
        &parts.stream,
        &[1, geometry.v_heads, geometry.v_dim, geometry.k_dim],
        MlxDType::F32,
    )?;
    let (recurrent, _next_state) =
        parts
            .recurrence
            .call(&queries, &keys, &values, &decay, &beta, &state)?;
    let normalized = recurrent.fast_rms_norm(&gated_delta.norm, geometry.eps)?;
    // Gated norm in FP32, rounded once to BF16 before the output projection.
    let gated = silu(&z.cast(MlxDType::F32)?)?
        .mul(&normalized.cast(MlxDType::F32)?)?
        .cast(recurrent.dtype())?;
    project(
        &gated.reshape(&[1, length, value_dim])?,
        &gated_delta.out.weight,
    )
}

fn attention_forward(
    parts: &Parts,
    input: &Array,
    attention: &Attention,
    positions: &Array,
) -> Result<Array> {
    let geometry = parts.geometry;
    let [1, length, _] = *input.shape() else {
        return Err(Error::Contract("d3-lite attention input must be [1, L, H]"));
    };
    // One packed projection yields queries and the output gate: `[B, L, 2*heads*head_dim]`.
    let projected = project(input, &attention.q.weight)?
        .reshape(&[1, length, geometry.q_heads, 2 * geometry.head_dim])?;
    let queries = projected.slice_axis(3, 0, geometry.head_dim)?;
    let gate = projected
        .slice_axis(3, geometry.head_dim, 2 * geometry.head_dim)?
        .reshape(&[1, length, geometry.q_heads * geometry.head_dim])?;
    let keys = project(input, &attention.k.weight)?;
    let values = project(input, &attention.v.weight)?;
    let queries = queries
        .fast_rms_norm(&attention.q_norm, geometry.eps)?
        .transpose(&[0, 2, 1, 3])?;
    let keys = keys
        .reshape(&[1, length, geometry.kv_heads, geometry.head_dim])?
        .fast_rms_norm(&attention.k_norm, geometry.eps)?
        .transpose(&[0, 2, 1, 3])?;
    let values = values
        .reshape(&[1, length, geometry.kv_heads, geometry.head_dim])?
        .transpose(&[0, 2, 1, 3])?;
    let queries = parts.rope.call(&queries, positions, &parts.inv_freq)?;
    let keys = parts.rope.call(&keys, positions, &parts.inv_freq)?;
    // d3's full-attention layers are explicitly non-causal over the prompt.
    let output = queries.sdpa(
        &keys,
        &values,
        (geometry.head_dim as f32).powf(-0.5),
        false,
        None,
    )?;
    let output = output
        .transpose(&[0, 2, 1, 3])?
        .reshape(&[1, length, geometry.q_heads * geometry.head_dim])?;
    project(&output.mul(&gate.sigmoid()?)?, &attention.o.weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream() -> Stream {
        Stream::cpu().expect("CPU stream for eager semantics")
    }

    #[test]
    fn inverse_frequencies_match_the_reference_formula() {
        let geometry = Geometry {
            hidden: 1_024,
            intermediate: 3_584,
            layers: 24,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            k_heads: 16,
            v_heads: 16,
            k_dim: 128,
            v_dim: 128,
            conv_kernel: 4,
            full_attention_interval: 4,
            eps: 1e-6,
            rope_dims: 64,
            rope_base: 10_000_000.0,
            vocab: 248_320,
            max_positions: 262_144,
        };
        let expected = (0..64)
            .step_by(2)
            .map(|index| (-(index as f32 / 64.0 * 10_000_000f32.ln())).exp())
            .collect::<Vec<_>>();
        let inv_freq = inverse_frequencies(&stream(), geometry).unwrap();
        assert_eq!(inv_freq.dtype(), MlxDType::F32);
        assert_eq!(inv_freq.shape(), &[32]);
        let actual = inv_freq.to_f32_vec().unwrap();
        assert_eq!(actual, expected);
        assert!(actual[0] > actual[31]);
        assert!((actual[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decay_beta_follows_compute_g_and_sigmoid() {
        let stream = stream();
        let a_log = Array::from_f32(&stream, &[2], &[0.0, 1.0]).unwrap();
        let a = Array::from_f32(&stream, &[1, 3, 2], &[0.0, 0.5, -0.5, 1.0, -1.0, 2.0])
            .unwrap()
            .cast(MlxDType::BF16)
            .unwrap();
        let b = Array::from_f32(&stream, &[1, 3, 2], &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
            .unwrap()
            .cast(MlxDType::BF16)
            .unwrap();
        let dt_bias = Array::from_f32(&stream, &[2], &[0.1, -0.1])
            .unwrap()
            .cast(MlxDType::BF16)
            .unwrap();
        let (decay, beta) = decay_beta(&a_log, &a, &b, &dt_bias).unwrap();
        stream.eval(&[decay.clone(), beta.clone()]).unwrap();
        assert_eq!(decay.dtype(), MlxDType::F32);
        assert_eq!(decay.shape(), &[1, 3, 2]);
        assert_eq!(beta.dtype(), MlxDType::BF16);
        let decay = decay.to_f32_vec().unwrap();
        let beta = beta.cast(MlxDType::F32).unwrap().to_f32_vec().unwrap();
        // `exp(-exp(A_log) * softplus(a + dt_bias))` in (0, 1].
        assert!(decay.iter().all(|value| value.is_finite() && *value > 0.0 && *value <= 1.0));
        assert!(decay.iter().zip([0.0f32, 1.0]).all(|(value, log)| value <= (-exp_f32(log)).exp() + 1e-6));
        assert!(beta.iter().all(|value| (0.0..1.0).contains(value)));
        // b = -2 → sigmoid ≈ 0.119; b = 3 → ≈ 0.953.
        assert!((beta[0] - (-2.0f32).exp() / (1.0 + (-2.0f32).exp())).abs() < 3e-3);
    }

    #[test]
    fn project_is_linear_without_bias() {
        let stream = stream();
        let weight = Array::from_f32(&stream, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let input = Array::from_f32(&stream, &[1, 2, 3], &[1.0, 0.0, -1.0, 0.5, 2.0, -0.5])
            .unwrap();
        let output = project(&input, &weight).unwrap();
        stream.eval(&[output.clone()]).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2]);
        assert_eq!(
            output.to_f32_vec().unwrap(),
            [-2.0, -2.0, 8.0, 8.5],
        );
    }

    fn exp_f32(value: f32) -> f32 {
        value.exp()
    }
}
