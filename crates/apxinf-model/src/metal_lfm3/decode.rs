//! Compiled, bucketed decode step for the native LFM3 family.
//!
//! The eager decode step builds ~200 device ops per token in Rust, deriving
//! every shape from the request state. This module ports the mechanism the
//! MiniCPM5 family already ships (`metal_minicpm5/model.rs::compile_step` and
//! `prepare`): one `mx.compile`d graph per request **capacity bucket** whose
//! only changing inputs are explicit compiled arguments — the token, the
//! position index, the two RoPE rows, and every KV/conv state array — so a
//! per-token step is a graph replay instead of host-side graph construction.
//!
//! Bucket geometry matches `Lfm3Config::request_capacity`: capacity is already a
//! 256-grain value, so the capacity *is* the bucket and the KV rows keep the
//! eager layout `[1, kv_heads, capacity, head_dim]`. Inside one capacity bucket
//! the attention width follows MiniCPM5's rule (`attention_width`): at
//! `capacity <= FULL_WIDTH_CAPACITY` one graph attends the whole capacity; above
//! it, each 256-grain width bucket that the request will cross is pre-captured
//! as its own graph, and `prepare()` warms every one of them so no decode step
//! pays a trace.
//!
//! The eager step stays the reference implementation (and the default): every
//! arithmetic operation below names the same primitive the eager path names, so
//! a parity failure is a scheduling difference, never a different recipe.

use super::{
    config::{LayerKind, Lfm3Config},
    model::{apply_rope, err, fused_conv_window, norm, silu_mul},
    weights::{Mixer, Weights},
};
use apxinf_core::Result;
use apxinf_mlx::{fusions::CausalDepthwiseConv, Array, Compiled, MlxDType, Stream};
use std::{collections::BTreeMap, rc::Rc};

/// Selects the compiled decode step: `1` compiles and uses it, anything else
/// (including unset) keeps the eager step. Opt-in until the performance gate in
/// `devlocal/lfm3-bucketed-decode/LEDGER.md` says otherwise.
pub(crate) const BUCKETED_DECODE_ENV: &str = "APXINF_LFM3_BUCKETED_DECODE";

/// Capacity grain of `Lfm3Config::request_capacity`; one bucket is one capacity.
pub(crate) const BUCKET_GRAIN: usize = 256;

/// At or below this capacity a single graph attends the whole capacity, exactly
/// like MiniCPM5's compiled variant (`attention_width` there). Above it the KV
/// columns grow with the position, so each 256-grain width gets one graph.
pub(crate) const FULL_WIDTH_CAPACITY: usize = 4096;

/// Which per-token step the family runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecodePolicy {
    /// Eager Rust graph construction per token (the default, the parity oracle).
    Eager,
    /// Pre-captured compiled step per capacity bucket.
    Bucketed,
}

impl DecodePolicy {
    pub(crate) fn from_override(value: Option<&std::ffi::OsStr>) -> Self {
        match value {
            Some(value) if value == "1" => Self::Bucketed,
            _ => Self::Eager,
        }
    }

    pub(crate) fn is_bucketed(self) -> bool {
        matches!(self, Self::Bucketed)
    }
}

/// KV columns one compiled step reads for a query at `end` (exclusive).
///
/// MiniCPM5's rule, kept verbatim so the two families bucket the same way: a
/// capacity this short admits one graph over the whole capacity; a longer one
/// reads the 256-grain bucket containing `end`.
pub(crate) fn attention_width(end: usize, capacity: usize) -> usize {
    if capacity <= FULL_WIDTH_CAPACITY {
        return capacity;
    }
    end.div_ceil(BUCKET_GRAIN)
        .saturating_mul(BUCKET_GRAIN)
        .min(capacity)
}

/// Every attention width a request that ends at `prompt_end` and lives in
/// `capacity` will hit, in ascending order. `prepare()` captures exactly these.
pub(crate) fn decode_widths(prompt_end: usize, capacity: usize) -> Vec<usize> {
    if capacity <= FULL_WIDTH_CAPACITY {
        return vec![capacity];
    }
    let mut width = attention_width(prompt_end.max(1), capacity);
    let mut widths = Vec::new();
    loop {
        widths.push(width);
        if width == capacity {
            return widths;
        }
        width = (width + BUCKET_GRAIN).min(capacity);
    }
}

/// `(start, span)` of the static KV columns a sliding-window layer reads.
///
/// A sliding layer may only attend the last `window` positions, so the static
/// slice must contain `[end - window, end]` for every `end` the graph serves:
/// `[0, width)` while the graph serves the whole capacity, otherwise the two
/// trailing width buckets of `width` (a position `end` in `(width - grain,
/// width]` can only reach back into `[width - grain - window, width)`).
pub(crate) fn sliding_span(width: usize, capacity: usize, window: usize) -> (usize, usize) {
    if width <= 2 * window || (width == capacity && capacity <= FULL_WIDTH_CAPACITY) {
        (0, width)
    } else {
        (width - 2 * window, 2 * window)
    }
}

/// Number of state arguments (and outputs) one compiled step carries.
fn state_arity(kinds: &[LayerKind]) -> usize {
    kinds
        .iter()
        .map(|kind| if kind.is_conv() { 1 } else { 2 })
        .sum()
}

/// Boolean `[1, 1, 1, span]` mask over the absolute key positions in
/// `positions` for the query position `index`; `window` additionally restricts
/// the sliding layers to the checkpoint's attention window.
fn key_mask(positions: &Array, index: &Array, window: Option<usize>) -> Result<Array> {
    let span = positions.shape()[0];
    let causal = positions.less_equal(index)?.reshape(&[1, 1, 1, span])?;
    let Some(window) = window else {
        return Ok(causal);
    };
    let stream = positions.stream();
    let distance = index.reshape(&[1])?.sub(positions)?;
    let inside =
        distance.less_equal(&Array::scalar(stream, (window - 1) as f32, MlxDType::I32)?)?;
    Ok(causal
        .cast(MlxDType::I32)?
        .mul(&inside.reshape(&[1, 1, 1, span])?.cast(MlxDType::I32)?)?
        .cast(MlxDType::Bool)?)
}

/// Builds the compiled one-token decode step for `capacity` and attention
/// `width`.
///
/// Arguments, in order: `tokens [1,1] i32`, `index [1] i32`, the full-attention
/// `cos`/`sin` rows, the sliding-attention `cos`/`sin` rows (each
/// `[1, 1, 1, head_dim] f32`), then one state per layer in layer order — a conv
/// history `[1, hidden, conv_L_cache - 1]` or a KV pair
/// `[1, kv_heads, capacity, head_dim]`. Outputs: `[1, vocab]` logits followed by
/// the updated states in the same order.
#[allow(clippy::too_many_arguments)]
pub(super) fn compile_step(
    stream: &Stream,
    weights: Rc<Weights>,
    config: Lfm3Config,
    kinds: Vec<LayerKind>,
    causal_conv: Rc<CausalDepthwiseConv>,
    capacity: usize,
    width: usize,
) -> Result<Compiled> {
    let eps = config.norm_eps;
    let hidden = config.hidden_size;
    let heads = config.num_attention_heads;
    let kv_heads = config.num_key_value_heads;
    let dim = config.head_dim;
    let vocab = config.vocab_size;
    let window = config.window();
    let scale = (dim as f64).powf(-0.5) as f32;
    let outputs = 1 + state_arity(&kinds);
    // Layer -> index of its first state argument.
    let mut state_at = Vec::with_capacity(kinds.len());
    let mut next = 6;
    for kind in &kinds {
        state_at.push(next);
        next += if kind.is_conv() { 1 } else { 2 };
    }
    let (slide_start, slide_span) = sliding_span(width, capacity, window);
    let full_positions = Array::arange(stream, 0., width as f32, 1., MlxDType::I32)?;
    let slide_positions = Array::arange(
        stream,
        slide_start as f32,
        (slide_start + slide_span) as f32,
        1.,
        MlxDType::I32,
    )?;
    let kinds_for_body = kinds.clone();
    Compiled::new(stream, outputs, move |a| {
        let index = a[1].clone();
        let cos_full = a[2].clone();
        let sin_full = a[3].clone();
        let cos_sliding = a[4].clone();
        let sin_sliding = a[5].clone();
        let mut x = weights.embed(&a[0])?.reshape(&[1, 1, hidden])?;
        let mut states = Vec::with_capacity(outputs - 1);
        for (layer_index, layer) in weights.layers.iter().enumerate() {
            let n = norm(&x, &layer.operator_norm, eps)?;
            let op = match &layer.mixer {
                Mixer::Conv {
                    in_proj,
                    out_proj,
                    taps_bf16,
                    ..
                } => {
                    let state = &a[state_at[layer_index]];
                    let bcx = in_proj.call(&n)?;
                    let b = bcx.slice_axis(2, 0, hidden)?;
                    let gate = bcx.slice_axis(2, hidden, 2 * hidden)?;
                    let z = bcx.slice_axis(2, 2 * hidden, 3 * hidden)?;
                    let z = b.mul(&z)?;
                    let window_in = Array::concat(&[&state.transpose(&[0, 2, 1])?, &z], 1)?;
                    let (conv, next) = fused_conv_window(&causal_conv, &window_in, taps_bf16)?;
                    let y = gate.mul(&conv)?;
                    states.push(next);
                    out_proj.call(&y)?
                }
                Mixer::Attention {
                    q: q_proj,
                    k: k_proj,
                    v: v_proj,
                    o: o_proj,
                    q_norm,
                    k_norm,
                    gate,
                } => {
                    let kind = kinds_for_body[layer_index];
                    let slot = state_at[layer_index];
                    let q = norm(&q_proj.call(&n)?.reshape(&[1, 1, heads, dim])?, q_norm, eps)?
                        .transpose(&[0, 2, 1, 3])?;
                    let k = norm(
                        &k_proj.call(&n)?.reshape(&[1, 1, kv_heads, dim])?,
                        k_norm,
                        eps,
                    )?
                    .transpose(&[0, 2, 1, 3])?;
                    let v = v_proj
                        .call(&n)?
                        .reshape(&[1, 1, kv_heads, dim])?
                        .transpose(&[0, 2, 1, 3])?;
                    let (cos, sin) = match kind {
                        LayerKind::FullAttention => (&cos_full, &sin_full),
                        LayerKind::SlidingAttention => (&cos_sliding, &sin_sliding),
                        LayerKind::Conv => {
                            return Err(err("LFM3 conv layer has an attention weight plan"))
                        }
                    };
                    let q = apply_rope(&q, cos, sin, dim)?;
                    let k = apply_rope(&k, cos, sin, dim)?;
                    let keys = a[slot].slice_update(&k, &index, &[2])?;
                    let values = a[slot + 1].slice_update(&v, &index, &[2])?;
                    let (start, span, positions, window_mask) = match kind {
                        LayerKind::FullAttention => (0, width, &full_positions, None),
                        LayerKind::SlidingAttention => {
                            (slide_start, slide_span, &slide_positions, Some(window))
                        }
                        LayerKind::Conv => {
                            return Err(err("LFM3 conv layer has an attention weight plan"))
                        }
                    };
                    let mask = key_mask(positions, &index, window_mask)?;
                    let attention = q.sdpa(
                        &keys.slice_axis(2, start, start + span)?,
                        &values.slice_axis(2, start, start + span)?,
                        scale,
                        false,
                        Some(&mask),
                    )?;
                    // Per-head sigmoid output gate, `[1, 1, heads] -> [1, heads, 1, 1]`.
                    let gate = gate.call(&n)?.sigmoid()?.transpose(&[0, 2, 1])?;
                    let gated = attention.mul(&gate.reshape(&[1, heads, 1, 1])?)?;
                    let merged = gated
                        .transpose(&[0, 2, 1, 3])?
                        .reshape(&[1, 1, heads * dim])?;
                    states.push(keys);
                    states.push(values);
                    o_proj.call(&merged)?
                }
            };
            // Sandwich-norm residual composition, identical to `mix_and_residual`.
            let op = norm(&op, &layer.post_operator_norm, eps)?;
            let residual = x.add(&op)?;
            let ffn = norm(&residual, &layer.ffn_norm, eps)?;
            let gate = layer.w1.call(&ffn)?;
            let up = layer.w3.call(&ffn)?;
            let activated = silu_mul(&gate, &up)?;
            let down = norm(&layer.w2.call(&activated)?, &layer.post_ffn_norm, eps)?;
            x = residual.add(&down)?;
        }
        let hidden_state = norm(&x, &weights.final_norm, eps)?;
        let logits = weights.project(&hidden_state)?.reshape(&[1, vocab])?;
        let mut output = vec![logits];
        output.extend(states);
        Ok(output)
    })
}

/// A request's prepared decode graphs: one compiled step per attention width in
/// `capacity`, reused across requests that land in the same capacity bucket.
pub(crate) struct PreparedDecode {
    capacity: usize,
    steps: BTreeMap<usize, Compiled>,
}

impl PreparedDecode {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            steps: BTreeMap::new(),
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn widths(&self) -> Vec<usize> {
        self.steps.keys().copied().collect()
    }

    pub(crate) fn contains(&self, width: usize) -> bool {
        self.steps.contains_key(&width)
    }

    pub(crate) fn insert(&mut self, width: usize, step: Compiled) {
        self.steps.insert(width, step);
    }

    pub(crate) fn step(&self, width: usize) -> Option<&Compiled> {
        self.steps.get(&width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_trait::{LlmInput, LlmTrait};
    use crate::metal_lfm3::model::Lfm3Model;
    use apxinf_core::Tensor;
    use apxinf_mlx::MlxBackend;
    use half::bf16;
    use std::{ffi::OsStr, sync::Arc};

    /// A reduced LFM3 geometry with every one of the three operator types.
    ///
    /// The production constructor still enforces the pinned LFM3-1.7B scope;
    /// this fixture exists so the bucket parity tests can run on a model small
    /// enough to build twice in one process. Arithmetic, layer plan, caches,
    /// kernels and both rotary programs are the production ones.
    const TEST_CONFIG: &str = r#"{
        "model_type": "lfm3",
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "vocab_size": 512,
        "max_position_embeddings": 8192,
        "norm_eps": 1e-5,
        "conv_L_cache": 7,
        "conv_bias": false,
        "sliding_window": 256,
        "layer_types": [
            "conv", "full_attention", "sliding_attention",
            "conv", "full_attention", "sliding_attention"
        ],
        "use_sandwich_norm": true,
        "tie_word_embeddings": true,
        "eos_token_id": 1,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"
            },
            "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}
        }
    }"#;

    fn values(len: usize, seed: u64, scale: f32, offset: f32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                (unit * 2.0 - 1.0) * scale + offset
            })
            .collect()
    }

    fn put(
        map: &mut std::collections::HashMap<String, Tensor>,
        name: &str,
        shape: &[usize],
        seed: u64,
        scale: f32,
        offset: f32,
    ) {
        let count = shape.iter().product::<usize>();
        let data: Vec<bf16> = values(count, seed, scale, offset)
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        map.insert(
            name.to_string(),
            Tensor::from_bf16(shape.to_vec(), &data).expect("synthetic LFM3 tensor"),
        );
    }

    fn norms(shape: usize, seed: u64) -> u64 {
        // Norm weights stay near 1 so the reduced model is not degenerate.
        seed.wrapping_add(shape as u64)
    }

    fn synthetic_tensors() -> std::collections::HashMap<String, Tensor> {
        let (h, i, heads, dim, kv, vocab) = (256usize, 512usize, 4usize, 64usize, 2usize, 512usize);
        let mut map = std::collections::HashMap::new();
        put(
            &mut map,
            "model.embed_tokens.weight",
            &[vocab, h],
            11,
            0.02,
            0.0,
        );
        put(&mut map, "model.embedding_norm.weight", &[h], 12, 0.1, 1.0);
        let kinds = [
            "conv",
            "full_attention",
            "sliding_attention",
            "conv",
            "full_attention",
            "sliding_attention",
        ];
        for (layer, kind) in kinds.iter().enumerate() {
            let p = format!("model.layers.{layer}");
            let seed = 100 + 17 * layer as u64;
            for suffix in [
                "operator_norm.weight",
                "ffn_norm.weight",
                "post_operator_norm.weight",
                "post_ffn_norm.weight",
            ] {
                put(
                    &mut map,
                    &format!("{p}.{suffix}"),
                    &[h],
                    norms(h, seed),
                    0.1,
                    1.0,
                );
            }
            put(
                &mut map,
                &format!("{p}.feed_forward.w1.weight"),
                &[i, h],
                seed,
                0.02,
                0.0,
            );
            put(
                &mut map,
                &format!("{p}.feed_forward.w3.weight"),
                &[i, h],
                seed + 1,
                0.02,
                0.0,
            );
            put(
                &mut map,
                &format!("{p}.feed_forward.w2.weight"),
                &[h, i],
                seed + 2,
                0.02,
                0.0,
            );
            if *kind == "conv" {
                put(
                    &mut map,
                    &format!("{p}.conv.conv.weight"),
                    &[h, 1, 7],
                    seed + 3,
                    0.2,
                    0.0,
                );
                put(
                    &mut map,
                    &format!("{p}.conv.in_proj.weight"),
                    &[3 * h, h],
                    seed + 4,
                    0.02,
                    0.0,
                );
                put(
                    &mut map,
                    &format!("{p}.conv.out_proj.weight"),
                    &[h, h],
                    seed + 5,
                    0.02,
                    0.0,
                );
            } else {
                put(
                    &mut map,
                    &format!("{p}.self_attn.q_proj.weight"),
                    &[heads * dim, h],
                    seed + 6,
                    0.02,
                    0.0,
                );
                put(
                    &mut map,
                    &format!("{p}.self_attn.k_proj.weight"),
                    &[kv * dim, h],
                    seed + 7,
                    0.02,
                    0.0,
                );
                put(
                    &mut map,
                    &format!("{p}.self_attn.v_proj.weight"),
                    &[kv * dim, h],
                    seed + 8,
                    0.02,
                    0.0,
                );
                put(
                    &mut map,
                    &format!("{p}.self_attn.o_proj.weight"),
                    &[heads * dim, h],
                    seed + 9,
                    0.02,
                    0.0,
                );
                put(
                    &mut map,
                    &format!("{p}.self_attn.q_norm.weight"),
                    &[dim],
                    seed + 10,
                    0.1,
                    1.0,
                );
                put(
                    &mut map,
                    &format!("{p}.self_attn.k_norm.weight"),
                    &[dim],
                    seed + 11,
                    0.1,
                    1.0,
                );
                put(
                    &mut map,
                    &format!("{p}.self_attn.gate_proj.weight"),
                    &[heads, h],
                    seed + 12,
                    0.02,
                    0.0,
                );
            }
        }
        map
    }

    fn synthetic_model(policy: DecodePolicy) -> Result<(Lfm3Model, Arc<MlxBackend>)> {
        let config = Lfm3Config::from_json(TEST_CONFIG)?;
        let backend = Arc::new(MlxBackend::new(0)?);
        let model =
            Lfm3Model::build_for_test(config, synthetic_tensors(), backend.clone(), policy)?;
        Ok((model, backend))
    }

    fn argmax(row: &[f32]) -> usize {
        let mut best = 0usize;
        for (index, value) in row.iter().enumerate() {
            if *value > row[best] {
                best = index;
            }
        }
        best
    }

    /// Runs the shared greedy generation loop and returns its token ids.
    fn generate_tokens(model: &mut Lfm3Model, prompt: &[u32], steps: usize) -> Result<Vec<u32>> {
        let mut options = crate::GenerationOptions::greedy(steps, None);
        options.eos_token_ids = None;
        let output = model.generate_streaming_controlled_dyn(
            crate::GenerationRequest {
                input: LlmInput::text(prompt),
                options: &options,
            },
            &mut |_| Ok(true),
        )?;
        Ok(output.token_ids())
    }

    /// Logits rows as host f32. The tensor a Metal family returns carries a
    /// Metal device, so it is read through the backend's array view rather than
    /// `Tensor::to_f32_vec` (which accepts CPU storage only).
    fn logits_of(backend: &MlxBackend, tensor: &Tensor) -> Result<Vec<f32>> {
        backend.array(tensor)?.to_f32_vec()
    }

    /// Greedy decode: the model samples its own argmax, so the token sequences
    /// are comparable end to end.
    fn greedy(
        model: &mut Lfm3Model,
        backend: &MlxBackend,
        prompt: &[u32],
        steps: usize,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
        let mut logits = logits_of(backend, &model.prefill(LlmInput::text(prompt))?)?;
        let mut tokens = Vec::with_capacity(steps);
        let mut rows = Vec::with_capacity(steps);
        for _ in 0..steps {
            let token = argmax(&logits) as u32;
            tokens.push(token);
            rows.push(logits);
            if tokens.len() == steps {
                break;
            }
            // Token `k` (1-based) is consumed at position prompt + k - 1.
            let position = (prompt.len() + tokens.len() - 1) as u32;
            logits = logits_of(backend, &model.forward(&[token], position)?)?;
        }
        Ok((tokens, rows))
    }

    /// Teacher-forced decode: both arms consume the *same* token ids, so their
    /// per-step logits are directly comparable even after a divergence.
    fn teacher_forced(
        model: &mut Lfm3Model,
        backend: &MlxBackend,
        prompt: &[u32],
        forced: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        let mut logits = logits_of(backend, &model.prefill(LlmInput::text(prompt))?)?;
        let mut rows = Vec::with_capacity(forced.len());
        for (step, token) in forced.iter().enumerate() {
            rows.push(logits);
            if step + 1 == forced.len() {
                break;
            }
            let position = (prompt.len() + step) as u32;
            logits = logits_of(backend, &model.forward(&[*token], position)?)?;
        }
        Ok(rows)
    }

    fn max_abs_diff(lhs: &[Vec<f32>], rhs: &[Vec<f32>]) -> f32 {
        let mut worst = 0f32;
        for (left, right) in lhs.iter().zip(rhs) {
            assert_eq!(left.len(), right.len(), "logit row widths differ");
            for (a, b) in left.iter().zip(right) {
                worst = worst.max((a - b).abs());
            }
        }
        worst
    }

    fn first_token_difference(lhs: &[u32], rhs: &[u32]) -> Option<usize> {
        lhs.iter().zip(rhs).position(|(a, b)| a != b)
    }

    #[test]
    fn policy_defaults_to_eager_and_selects_bucketed_only_for_one() {
        assert_eq!(DecodePolicy::from_override(None), DecodePolicy::Eager);
        assert_eq!(
            DecodePolicy::from_override(Some(OsStr::new("1"))),
            DecodePolicy::Bucketed
        );
        for value in ["0", "", "true", "yes"] {
            assert_eq!(
                DecodePolicy::from_override(Some(OsStr::new(value))),
                DecodePolicy::Eager,
                "value {value:?}"
            );
        }
    }

    #[test]
    fn short_capacities_compile_one_graph_at_the_whole_capacity() {
        for capacity in [256usize, 512, 1024, 2048, 4096] {
            for end in [1usize, 127, 256, capacity] {
                assert_eq!(
                    attention_width(end, capacity),
                    capacity,
                    "capacity={capacity} end={end}"
                );
            }
            assert_eq!(decode_widths(1, capacity), vec![capacity]);
        }
    }

    #[test]
    fn long_capacities_bucket_the_attention_width_by_the_next_position() {
        // 4608 is the first capacity above the full-width envelope.
        assert_eq!(attention_width(1, 4608), 256);
        assert_eq!(attention_width(256, 4608), 256);
        assert_eq!(attention_width(257, 4608), 512);
        assert_eq!(attention_width(4351, 4608), 4352);
        assert_eq!(attention_width(4352, 4608), 4352);
        assert_eq!(attention_width(4353, 4608), 4608);
        assert_eq!(attention_width(4608, 4608), 4608);
        // A 4301-token prompt crossing two buckets.
        assert_eq!(decode_widths(4301, 4608), vec![4352, 4608]);
        // The full ladder from the first bucket to the capacity.
        let ladder = decode_widths(1, 4608);
        assert_eq!(ladder.len(), 4608 / BUCKET_GRAIN);
        assert_eq!(ladder[0], 256);
        assert_eq!(*ladder.last().unwrap(), 4608);
    }

    #[test]
    fn sliding_span_covers_every_window_a_graph_serves() {
        let window = 256usize;
        // (width, capacity) -> the whole width is the static slice when the
        // graph serves positions from the start of the request.
        assert_eq!(sliding_span(256, 256, window), (0, 256));
        assert_eq!(sliding_span(512, 512, window), (0, 512));
        assert_eq!(sliding_span(1024, 1024, window), (0, 1024));
        assert_eq!(sliding_span(2048, 2048, window), (0, 2048));
        // Above the full-width envelope the width moves with the position, so
        // the two trailing buckets are enough and the slice does not grow.
        assert_eq!(sliding_span(4352, 4608, window), (3840, 512));
        assert_eq!(sliding_span(4608, 4608, window), (4096, 512));
        // Coverage: for every end in (width - grain, width] the admissible
        // window [end - window, end) lies inside the static slice.
        for (start, span) in [
            sliding_span(256, 256, window),
            sliding_span(1024, 1024, window),
            sliding_span(4352, 4608, window),
            sliding_span(4608, 4608, window),
        ] {
            let width = start + span;
            let first = width.saturating_sub(BUCKET_GRAIN) + 1;
            for end in first..=width {
                let low = end.saturating_sub(window);
                assert!(
                    low >= start && end <= start + span,
                    "window [{low}, {end}) escapes [{start}, {}) at width {width}",
                    start + span
                );
            }
        }
    }

    #[test]
    #[ignore = "requires shared Metal qualification lock"]
    fn compiled_step_matches_the_eager_step_in_every_bucket() -> Result<()> {
        let steps = 64usize;
        for bucket in [256usize, 512, 1024, 2048] {
            // prompt + output lands exactly on the bucket: the boundary case.
            let prompt: Vec<u32> = (0..128u32).map(|i| 8 + i % 500).collect();
            let max_new = bucket - prompt.len();
            let (mut model, backend) = synthetic_model(DecodePolicy::Eager)?;
            model.prepare(prompt.len(), max_new)?;
            assert_eq!(model.capacity(), bucket, "capacity bucket");
            let (eager_tokens, eager_rows) = greedy(&mut model, &backend, &prompt, steps)?;
            assert_eq!(eager_tokens.len(), steps);

            model.reset();
            let forced = eager_tokens.clone();
            let eager_forced = teacher_forced(&mut model, &backend, &prompt, &forced)?;

            model.reset();
            model.select_decode_policy(DecodePolicy::Bucketed);
            model.prepare(prompt.len(), max_new)?;
            assert_eq!(
                model.decode_buckets(),
                vec![bucket],
                "bucket {bucket} must be pre-captured"
            );
            let traces = backend.stream().counters().trace_callbacks;
            let (bucketed_tokens, bucketed_rows) = greedy(&mut model, &backend, &prompt, steps)?;
            assert_eq!(
                backend.stream().counters().trace_callbacks,
                traces,
                "bucket {bucket} retraced during decode"
            );
            assert_eq!(
                first_token_difference(&eager_tokens, &bucketed_tokens),
                None,
                "bucket {bucket} greedy tokens diverged"
            );

            model.reset();
            let bucketed_forced = teacher_forced(&mut model, &backend, &prompt, &forced)?;
            let diff = max_abs_diff(&eager_forced, &bucketed_forced);
            eprintln!(
                "lfm3 bucket={bucket} parity: steps={steps} tokens_identical=true logits_max_abs_diff={diff:.3e}"
            );
            assert!(
                diff <= 2e-3,
                "bucket {bucket} logits residual {diff:.3e} exceeds the 2e-3 gate"
            );
            // The self-sampled rows must agree too: identical tokens plus
            // identical inputs means the same comparison.
            assert!(max_abs_diff(&eager_rows, &bucketed_rows) <= 2e-3);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires shared Metal qualification lock"]
    fn bucket_boundaries_grow_reuse_and_reject_the_next_position() -> Result<()> {
        let prompt: Vec<u32> = (0..128u32).map(|i| 8 + i % 500).collect();
        let (mut model, backend) = synthetic_model(DecodePolicy::Bucketed)?;
        // Exact boundary: 128 + 128 == 256.
        model.prepare(prompt.len(), 128)?;
        assert_eq!(model.capacity(), 256);
        assert_eq!(model.decode_buckets(), vec![256]);
        let mut logits = logits_of(&backend, &model.prefill(LlmInput::text(&prompt))?)?;
        let mut position = prompt.len();
        while position < 256 {
            let token = argmax(&logits) as u32;
            logits = logits_of(&backend, &model.forward(&[token], position as u32)?)?;
            position += 1;
        }
        assert_eq!(model.position(), 256);
        // One past the last admissible position of the bucket is rejected, and
        // it is rejected rather than silently stepping outside the graph.
        assert!(model.forward(&[argmax(&logits) as u32], 256).is_err());

        // Growth: 200 + 200 == 400 rounds up to 512, a larger bucket.
        model.reset();
        model.prepare(200, 200)?;
        assert_eq!(model.capacity(), 512);
        assert_eq!(model.decode_buckets(), vec![512]);
        let (tokens, _) = greedy(
            &mut model,
            &backend,
            &(0..200u32).map(|i| 8 + i % 500).collect::<Vec<_>>(),
            16,
        )?;
        assert_eq!(tokens.len(), 16);

        // A smaller re-prepare keeps the capacity and reuses the compiled graph:
        // no trace and no new bucket.
        model.reset();
        let traces = backend.stream().counters().trace_callbacks;
        model.prepare(10, 10)?;
        assert_eq!(model.capacity(), 512, "capacity never shrinks in a session");
        assert_eq!(model.decode_buckets(), vec![512]);
        assert_eq!(
            backend.stream().counters().trace_callbacks,
            traces,
            "re-prepare with a smaller request must reuse the compiled step"
        );
        let (tokens, _) = greedy(&mut model, &backend, &[9, 10, 11], 16)?;
        assert_eq!(tokens.len(), 16);

        // Outside the admitted envelope: no halo bucket, no chunked KV.
        model.reset();
        assert!(model.prepare(8192, 1).is_err());
        assert!(model.prepare(0, 8).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires shared Metal qualification lock"]
    fn long_context_crosses_width_buckets_without_retracing() -> Result<()> {
        let prompt: Vec<u32> = (0..4300u32).map(|i| 3 + i % 500).collect();
        let steps = 64usize;
        let (mut model, backend) = synthetic_model(DecodePolicy::Eager)?;
        model.prepare(prompt.len(), 308)?;
        assert_eq!(model.capacity(), 4608);
        let (eager_tokens, _) = greedy(&mut model, &backend, &prompt, steps)?;
        model.reset();
        let eager_forced = teacher_forced(&mut model, &backend, &prompt, &eager_tokens)?;

        model.reset();
        model.select_decode_policy(DecodePolicy::Bucketed);
        model.prepare(prompt.len(), 308)?;
        assert_eq!(model.decode_buckets(), vec![4352, 4608]);
        let traces = backend.stream().counters().trace_callbacks;
        let (bucketed_tokens, _) = greedy(&mut model, &backend, &prompt, steps)?;
        assert_eq!(backend.stream().counters().trace_callbacks, traces);
        assert_eq!(
            first_token_difference(&eager_tokens, &bucketed_tokens),
            None
        );
        model.reset();
        let bucketed_forced = teacher_forced(&mut model, &backend, &prompt, &eager_tokens)?;
        let diff = max_abs_diff(&eager_forced, &bucketed_forced);
        eprintln!("lfm3 long-context parity: steps={steps} logits_max_abs_diff={diff:.3e}");
        assert!(diff <= 2e-3);
        Ok(())
    }

    /// The public generation loop (reset -> prewarm_decode -> prefill ->
    /// per-token forward) must produce the same greedy tokens on both arms, and
    /// the bucketed arm must run prepared graphs rather than retracing.
    #[test]
    #[ignore = "requires shared Metal qualification lock"]
    fn generation_loop_matches_between_arms_and_engages_the_bucket() -> Result<()> {
        let prompt: Vec<u32> = (0..96u32).map(|i| 8 + i % 500).collect();
        let steps = 48usize;
        let (mut model, backend) = synthetic_model(DecodePolicy::Eager)?;
        let eager = generate_tokens(&mut model, &prompt, steps)?;
        assert_eq!(eager.len(), steps);

        model.select_decode_policy(DecodePolicy::Bucketed);
        // Prepare first: the bucket compile is preparation, not decode. The
        // measured window starts after it, exactly like the perf gate does.
        model.prepare(prompt.len(), steps)?;
        let traces = backend.stream().counters().trace_callbacks;
        let bucketed = generate_tokens(&mut model, &prompt, steps)?;
        assert_eq!(
            backend.stream().counters().trace_callbacks,
            traces,
            "generation retraced on the bucketed arm"
        );
        assert_eq!(model.decode_buckets(), vec![256]);
        assert_eq!(eager, bucketed);
        Ok(())
    }

    /// Real-checkpoint parity, run only when the BF16 checkpoint is complete:
    /// `APXINF_LFM3_CHECKPOINT=~/lfm3bench/models/bf16 cargo test -p apxinf-model
    /// metal_lfm3::decode::tests::real_checkpoint -- --ignored --nocapture`.
    ///
    /// Every admitted capacity bucket gets its own comparison: prompt length plus
    /// the 64 decode steps is chosen to land exactly on 256 / 512 / 1024 / 2048.
    #[test]
    #[ignore = "requires the complete LFM3-1.7B BF16 checkpoint, isolated process and shared Metal lock"]
    fn real_checkpoint_bucketed_parity() -> Result<()> {
        let checkpoint = std::env::var_os("APXINF_LFM3_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                    .join("lfm3bench/models/bf16")
            });
        let weights_path = checkpoint.join("model.safetensors");
        let size = std::fs::metadata(&weights_path)
            .map_err(|e| err(format!("LFM3 checkpoint missing at {weights_path:?}: {e}")))?
            .len();
        let json = std::fs::read_to_string(checkpoint.join("config.json"))
            .map_err(|e| err(format!("LFM3 checkpoint config missing: {e}")))?;
        let config = Lfm3Config::from_json(&json)?;
        // Same service allowance the native text probe uses, so this test can
        // run beside an ordinary desktop session without an unbounded allocator.
        apxinf_mlx::set_memory_limit(10 * 1024 * 1024 * 1024)?;
        apxinf_mlx::set_cache_limit(2 * 1024 * 1024 * 1024)?;
        let backend = Arc::new(MlxBackend::new(0)?);
        let (tensors, _) =
            apxinf_loader::safetensors::load_native_path(&checkpoint).map_err(|e| {
                err(format!(
                    "LFM3 checkpoint load failed (size {size} bytes): {e}"
                ))
            })?;
        let mut model = Lfm3Model::new(config, tensors, backend.clone())?;
        let steps = 64usize;
        for (prompt_len, bucket) in [(64usize, 256usize), (448, 512), (960, 1024), (1984, 2048)] {
            let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| 8 + i % 500).collect();
            model.reset();
            model.prepare(prompt.len(), steps)?;
            assert_eq!(model.capacity(), bucket, "prompt {prompt_len}");
            let (eager_tokens, _) = greedy(&mut model, &backend, &prompt, steps)?;
            model.reset();
            let eager_forced = teacher_forced(&mut model, &backend, &prompt, &eager_tokens)?;

            model.reset();
            model.select_decode_policy(DecodePolicy::Bucketed);
            model.prepare(prompt.len(), steps)?;
            assert_eq!(model.decode_buckets(), vec![bucket]);
            let traces = backend.stream().counters().trace_callbacks;
            let (bucketed_tokens, _) = greedy(&mut model, &backend, &prompt, steps)?;
            assert_eq!(backend.stream().counters().trace_callbacks, traces);
            model.reset();
            let bucketed_forced = teacher_forced(&mut model, &backend, &prompt, &eager_tokens)?;
            let diff = max_abs_diff(&eager_forced, &bucketed_forced);
            eprintln!(
                "lfm3 real-geometry parity: checkpoint_bytes={size} bucket={bucket} \
                 prompt={prompt_len} steps={steps} greedy_tokens_identical={} \
                 logits_max_abs_diff={diff:.3e}",
                eager_tokens == bucketed_tokens
            );
            assert!(diff <= 2e-3, "bucket {bucket} logits residual {diff:.3e}");
        }
        Ok(())
    }
}
