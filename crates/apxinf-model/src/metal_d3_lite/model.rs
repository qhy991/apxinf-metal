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
//! primitives: `GatedDeltaRecurrence` (four value rows per SIMD group, or
//! eight behind `APXINF_D3_GDN_CORE`, which now defaults to `eight_lanes`
//! after this leg's KEEP; the image callable keeps its own
//! `APXINF_D3_GDN_CORE_IMAGE` defaulting to `four_values` because its parity
//! budget has no headroom) and `HalfSplitRope` (FP32 angles, one rounding to
//! the input dtype). The causal convolution defaults to
//! `APXINF_D3_CONV1D=apxinf` (the ApxInf one-thread-per-output kernel: FP32
//! tap order, its own zero history, one launch, bit-identical to stock);
//! `APXINF_D3_CONV1D=stock` restores the composed MLX `conv1d` over an
//! explicit zero history. See `doc/d3-lite-metal-gdn.md` (P3, D3-A) and
//! `doc/notworth-audit.md` §4.2.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use apxinf_core::{Error, Result};
use apxinf_mlx::fusions::{
    CausalDepthwiseConv, GatedDeltaLayout, GatedDeltaRecurrence, HalfSplitRope,
};
use apxinf_mlx::{Array, Compiled, MlxDType, Stream};

use super::config::{Config, Geometry, MAX_INPUT_TOKENS};
use super::signpost;
use super::vision::Vision;
use super::weights::{Attention, GatedDelta, Mixer, Mlp, VisionWeights, Weights};
use crate::decision::DecisionStageProfile;

/// EngineTailor's `conv1d` option over the GDN causal convolution. `stock`
/// keeps the composed MLX `conv1d` over an explicit zero history; `apxinf`
/// runs the ApxInf one-thread-per-output kernel (FP32 tap order, own zero
/// history, one launch). The list is ordered **default first**: this leg's
/// KEEP configuration is `apxinf` (bit-identical to `stock`, +0.53 % on its
/// own and part of the +4.27 % combined result — see
/// `doc/d3-lite-metal-gdn.md`). To reproduce the P1a/P1b anchor set
/// `APXINF_D3_CONV1D=stock`.
const CONV1D_VALUES: &[&str] = &["apxinf", "stock"];
/// The recurrence layout. The native port's base is EngineTailor's
/// `apxinf_four_values`; there is no vendored mlx-lm stock kernel in this
/// port, so the values name the two ApxInf layouts.
///
/// [`GDN_CORE`] selects the **text** callable; [`GDN_CORE_IMAGE`] selects the
/// **image** callable with its own list, because the two paths have different
/// parity headroom: the image budget (0.06) has none (P1b's worst case was
/// 0.0554), so a layout change the text gate welcomes can still fail it. For
/// each gate the **first value is that gate's default** (the reference
/// executor's "absent takes the first value" convention), so the text default
/// can be `eight_lanes` (this leg's KEEP) while the image default stays
/// `four_values` (the P1a/P1b anchor).
const GDN_CORE_VALUES: &[&str] = &["eight_lanes", "four_values"];
const GDN_CORE_IMAGE_VALUES: &[&str] = &["four_values", "eight_lanes"];
const GDN_CORE: &str = "APXINF_D3_GDN_CORE";
const GDN_CORE_IMAGE: &str = "APXINF_D3_GDN_CORE_IMAGE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Conv1d {
    Stock,
    ApxInf,
}

/// Resolve one env gate against its value list: `None` (unset) selects index
/// 0, `Some` must name a value exactly. Kept pure so the gate contract is
/// unit-testable without a device; [`env_gate`] wraps it with the real env.
fn parse_gate(given: Option<&str>, values: &[&str], name: &str) -> Result<usize> {
    match given {
        None => Ok(0),
        Some(value) => values
            .iter()
            .position(|known| *known == value)
            .ok_or_else(|| {
                Error::Other(format!(
                    "{name}={value:?} is not one of {}",
                    values.join(", ")
                ))
            }),
    }
}

fn env_gate(name: &str, values: &[&str]) -> Result<usize> {
    let given = match std::env::var(name) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(Error::Other(format!(
                "{name} must be one of {} (the value is not valid UTF-8)",
                values.join(", ")
            )))
        }
    };
    parse_gate(given.as_deref(), values, name)
}

/// Everything the compiled callables capture: immutable geometry, evaluated
/// weights, registered kernels, the eager vision tower and the
/// request-independent inverse frequencies. `Rc` lets the trace closures and
/// the runner share one copy without introducing a second owner for device
/// arrays.
struct Parts {
    geometry: Geometry,
    weights: Weights,
    vision: Vision,
    image_token_id: u32,
    video_token_id: u32,
    vision_merge: usize,
    /// The text callable's recurrence layout (`APXINF_D3_GDN_CORE`).
    recurrence: GatedDeltaRecurrence,
    /// The image callable's recurrence layout (`APXINF_D3_GDN_CORE_IMAGE`).
    /// Kept separate so an image request can stay on the parity-safe anchor
    /// while the text path takes the faster layout.
    recurrence_image: GatedDeltaRecurrence,
    /// The gate-selected causal-conv implementation; `None` when `stock`
    /// composes it from MLX primitives inside `gated_delta_forward`.
    conv1d: Conv1d,
    causal_conv: Option<CausalDepthwiseConv>,
    rope: HalfSplitRope,
    inv_freq: Array,
    stream: Stream,
}

/// One d3-lite execution over shared weights. `compiled` is the text
/// whole-request callable (`ids [1, L]`, `count [1]`); `image_compiled`
/// adds the multimodal inputs (`positions [3, L]`, `features [1, M, D]`,
/// `run_start [1]`). `prepared` / `prepared_images` record the exact profiles
/// that have been warmed with `Compiled::prepare`, which is the only
/// operation that may trace.
pub struct Model {
    parts: Rc<Parts>,
    compiled: Compiled,
    image_compiled: Compiled,
    prepared: RefCell<Vec<(usize, usize)>>,
    prepared_images: RefCell<Vec<(usize, usize, usize)>>,
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
    pub fn new(
        config: &Config,
        stream: Stream,
        weights: Weights,
        vision_weights: VisionWeights,
    ) -> Result<Self> {
        let geometry = config.geometry;
        let vision = Vision::new(config, stream.clone(), vision_weights)?;
        // All gates are read once, here: the compiled trace closures must see
        // a fixed configuration, and a campaign runs its arms in separate
        // processes with different env.
        let conv1d = match CONV1D_VALUES[env_gate("APXINF_D3_CONV1D", CONV1D_VALUES)?] {
            "stock" => Conv1d::Stock,
            "apxinf" => Conv1d::ApxInf,
            other => unreachable!("the conv1d list only names known kernels: {other}"),
        };
        // Resolve by the selected *name*, not the list index, so a gate's
        // default can move by reordering its value list without silently
        // selecting the wrong kernel.
        let make_recurrence = |gate: &str, values: &[&str]| -> Result<GatedDeltaRecurrence> {
            match values[env_gate(gate, values)?] {
                "four_values" => GatedDeltaRecurrence::new(&stream),
                "eight_lanes" => {
                    GatedDeltaRecurrence::with_layout(&stream, GatedDeltaLayout::EightLanes)
                }
                other => {
                    unreachable!("the layout list only names known layouts: {other}")
                }
            }
        };
        let recurrence = make_recurrence(GDN_CORE, GDN_CORE_VALUES)?;
        let recurrence_image = make_recurrence(GDN_CORE_IMAGE, GDN_CORE_IMAGE_VALUES)?;
        let causal_conv = match conv1d {
            Conv1d::Stock => None,
            Conv1d::ApxInf => Some(CausalDepthwiseConv::new(&stream)?),
        };
        let parts = Rc::new(Parts {
            geometry,
            weights,
            vision,
            image_token_id: config.image_token_id,
            video_token_id: config.video_token_id,
            vision_merge: config.vision.spatial_merge,
            recurrence,
            recurrence_image,
            conv1d,
            causal_conv,
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
        let trace_parts = parts.clone();
        let image_compiled = Compiled::with_shapeless(&stream, 1, false, move |inputs| {
            let [ids, positions, count, features, run_start] = inputs else {
                return Err(Error::Contract(
                    "d3-lite compiled image forward takes ids, mrope positions, candidate \
                     count, merged image features and the placeholder run start",
                ));
            };
            image_forward(&trace_parts, ids, positions, count, features, run_start)
        })?;
        Ok(Self {
            parts,
            compiled,
            image_compiled,
            prepared: RefCell::new(Vec::new()),
            prepared_images: RefCell::new(Vec::new()),
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
                    Error::Other(format!(
                        "d3-lite token ID {token} exceeds the embedding index range"
                    ))
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
        self.probabilities_profiled(token_ids, count)
            .map(|(values, _)| values)
    }

    /// [`probabilities`](Self::probabilities) plus the host wall-clock
    /// attribution of this call. `layout`/`vision` are zero and `forward +
    /// sync` close against the call's own wall time.
    pub fn probabilities_profiled(
        &self,
        token_ids: &[u32],
        count: usize,
    ) -> Result<(Vec<f32>, DecisionStageProfile)> {
        self.validate(token_ids, count)?;
        let request = Instant::now();
        self.prepare_for(token_ids, count)?;
        let (ids, candidate_count) = self.inputs(token_ids, count)?;
        let _fwd = signpost_interval!(signpost::FORWARD, token_ids.len(), 0usize);
        let outputs = self.compiled.call_and_eval(&[ids, candidate_count])?;
        drop(_fwd);
        let forward_ms = request.elapsed().as_secs_f64() * 1e3;
        let sync = Instant::now();
        let _sync = signpost_interval!(signpost::SYNC, count);
        let probabilities = read_probabilities(&outputs[0], count)?;
        drop(_sync);
        let sync_ms = sync.elapsed().as_secs_f64() * 1e3;
        Ok((
            probabilities,
            DecisionStageProfile {
                layout_ms: 0.0,
                vision_ms: 0.0,
                forward_ms,
                sync_ms,
            },
        ))
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

    fn is_image_prepared(&self, length: usize, count: usize, image_tokens: usize) -> bool {
        self.prepared_images
            .borrow()
            .iter()
            .any(|&(known, known_count, known_tokens)| {
                known == length && known_count == count && known_tokens == image_tokens
            })
    }

    /// Trace and warm one exact image profile. Zero token IDs, positions,
    /// features and run start keep the warm-up input independent of any
    /// request; the traced graph depends on shapes, not values.
    fn prepare_image_profile(
        &self,
        length: usize,
        count: usize,
        image_tokens: usize,
    ) -> Result<()> {
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
        if image_tokens == 0 || image_tokens > length {
            return Err(Error::Other(format!(
                "d3-lite prepared image token count {image_tokens} is outside 1..={length}"
            )));
        }
        let ids = Array::zeros(self.stream(), &[1, length], MlxDType::I32)?;
        let positions = Array::zeros(self.stream(), &[3, length], MlxDType::I32)?;
        let count_array = Array::from_i32(self.stream(), &[1], &[count as i32])?;
        let features = Array::zeros(
            self.stream(),
            &[1, image_tokens, geometry.hidden],
            MlxDType::BF16,
        )?;
        // Start 0 writes the feature run over the head of the warm-up row;
        // the values do not affect the traced graph.
        let run_start = Array::zeros(self.stream(), &[1], MlxDType::I32)?;
        self.image_compiled
            .prepare(&[ids, positions, count_array, features, run_start])?;
        if !self.is_image_prepared(length, count, image_tokens) {
            self.prepared_images
                .borrow_mut()
                .push((length, count, image_tokens));
        }
        Ok(())
    }

    fn prepare_image_for(&self, length: usize, count: usize, image_tokens: usize) -> Result<()> {
        if self.is_image_prepared(length, count, image_tokens) {
            return Ok(());
        }
        self.prepare_image_profile(length, count, image_tokens)
    }

    /// One image request: the eager vision tower produces merged image
    /// tokens, the compiled image callable scatters them into the prompt and
    /// runs the text layers plus FP32 readout in one forward. Returns the
    /// first `count` option probabilities, in package code order.
    pub fn image_probabilities(
        &self,
        pixel_values: &[f32],
        grids: &[[u32; 3]],
        token_ids: &[u32],
        count: usize,
    ) -> Result<Vec<f32>> {
        self.image_probabilities_profiled(pixel_values, grids, token_ids, count)
            .map(|(values, _)| values)
    }

    /// [`image_probabilities`](Self::image_probabilities) plus the host
    /// wall-clock attribution of this call: `layout` is request validation
    /// plus the multimodal layout walk, `vision` the eager tower's graph build
    /// and blocking eval, `forward` the one-time profile trace, input uploads
    /// and compiled call, and `sync` the probability readback. The four close
    /// against the call's own wall time.
    pub fn image_probabilities_profiled(
        &self,
        pixel_values: &[f32],
        grids: &[[u32; 3]],
        token_ids: &[u32],
        count: usize,
    ) -> Result<(Vec<f32>, DecisionStageProfile)> {
        let layout = Instant::now();
        self.validate(token_ids, count)?;
        let merge = self.parts.vision_merge;
        if merge == 0 {
            return Err(Error::Contract(
                "d3-lite vision merge size must be positive",
            ));
        }
        let (positions, slots) = multimodal_layout(
            token_ids,
            grids,
            self.parts.image_token_id,
            self.parts.video_token_id,
            merge,
        )?;
        let length = token_ids.len();
        let image_tokens = slots.len();
        // The processor expands one image into one contiguous placeholder run;
        // MLX's dynamic slice_update takes one start index, so that run is the
        // only scatter this path supports.
        let run_start = slots[0];
        if slots.windows(2).any(|window| window[1] != window[0] + 1) {
            return Err(Error::Other(
                "d3-lite Metal P1b writes one contiguous image run per request; \
                 multiple images are not wired"
                    .into(),
            ));
        }
        let layout_ms = layout.elapsed().as_secs_f64() * 1e3;
        // `vision` spans the tower's host graph build AND its blocking eval:
        // the lazy graph contributes no device time until it is evaluated, so
        // an eval left outside the span would leave the stage sum unable to
        // close.
        let vision = Instant::now();
        let _vision = signpost_interval!(signpost::VISION, image_tokens);
        let features = self.parts.vision.forward(pixel_values, grids)?;
        let features = features.reshape(&[1, image_tokens, self.parts.geometry.hidden])?;
        // Evaluate before the compiled callable sees it: preparation passes
        // concrete zero arrays, so a concrete input keeps the profile
        // replaying without re-tracing. Placing the eval directly after the
        // tower lets the span own the tower's full wall time; it still
        // precedes `prepare_image_for` and `call_and_eval`, which consume no
        // real feature value.
        self.stream().eval(&[features.clone()])?;
        drop(_vision);
        let vision_ms = vision.elapsed().as_secs_f64() * 1e3;

        let forward = Instant::now();
        self.prepare_image_for(length, count, image_tokens)?;

        let ids = token_ids
            .iter()
            .map(|&token| {
                i32::try_from(token).map_err(|_| {
                    Error::Other(format!(
                        "d3-lite token ID {token} exceeds the embedding index range"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let ids = Array::from_i32(self.stream(), &[1, length], &ids)?;
        let positions = Array::from_i32(self.stream(), &[3, length], &positions)?;
        let count_array = Array::from_i32(self.stream(), &[1], &[count as i32])?;
        let run_start = Array::from_i32(self.stream(), &[1], &[run_start])?;
        let _fwd = signpost_interval!(signpost::FORWARD, length, image_tokens);
        let outputs = self.image_compiled.call_and_eval(&[
            ids,
            positions,
            count_array,
            features,
            run_start,
        ])?;
        drop(_fwd);
        let forward_ms = forward.elapsed().as_secs_f64() * 1e3;

        let sync = Instant::now();
        let _sync = signpost_interval!(signpost::SYNC, count);
        let probabilities = read_probabilities(&outputs[0], count)?;
        drop(_sync);
        let sync_ms = sync.elapsed().as_secs_f64() * 1e3;
        Ok((
            probabilities,
            DecisionStageProfile {
                layout_ms,
                vision_ms,
                forward_ms,
                sync_ms,
            },
        ))
    }

    /// `(length, count, image_tokens)` image profiles warmed so far.
    pub fn prepared_image_profiles(&self) -> Vec<(usize, usize, usize)> {
        self.prepared_images.borrow().clone()
    }
}

/// The multimodal request layout the reference builds from `mm_token_type_ids`
/// and `image_grid_thw`: mRoPE positions `[3, L]` (text positions advance by
/// one per token; an image run takes its grid's H/W coordinates and advances
/// by `max(h, w) / merge`) and the placeholder slot indices `[M]` the merged
/// image tokens scatter into. A run of `image_token_id` is one image, in
/// request order, and must match the next grid exactly.
fn multimodal_layout(
    token_ids: &[u32],
    grids: &[[u32; 3]],
    image_token_id: u32,
    video_token_id: u32,
    merge: usize,
) -> Result<(Vec<i32>, Vec<i32>)> {
    if grids.is_empty() {
        return Err(Error::Other(
            "d3-lite image request needs at least one grid".into(),
        ));
    }
    let length = token_ids.len();
    let mut positions = vec![vec![0i32; length]; 3];
    let mut slots = Vec::new();
    let mut current = 0i32;
    let mut index = 0usize;
    let mut grid_index = 0usize;
    while index < length {
        if token_ids[index] == video_token_id {
            return Err(Error::Other(
                "d3-lite Metal image path does not read video tokens; video is not wired".into(),
            ));
        }
        let image_run = token_ids[index] == image_token_id;
        let mut end = index + 1;
        while end < length && (token_ids[end] == image_token_id) == image_run {
            end += 1;
        }
        if image_run {
            let [temporal, grid_h, grid_w] = *grids.get(grid_index).ok_or_else(|| {
                Error::Other(format!(
                    "d3-lite prompt has more image runs than grids ({grid_index} grids consumed)"
                ))
            })?;
            grid_index += 1;
            if temporal != 1 {
                return Err(Error::Other(
                    "d3-lite Metal vision runs still images (grid t = 1); video is not wired"
                        .into(),
                ));
            }
            let merged_h = grid_h as usize / merge;
            let merged_w = grid_w as usize / merge;
            let expected = merged_h * merged_w;
            if merged_h == 0 || merged_w == 0 {
                return Err(Error::Other(format!(
                    "d3-lite image grid [{temporal}, {grid_h}, {grid_w}] is smaller than one merge block"
                )));
            }
            if end - index != expected {
                return Err(Error::Other(format!(
                    "d3-lite image run at position {index} holds {} placeholders, expected {expected} \
                     for grid [{temporal}, {grid_h}, {grid_w}]",
                    end - index
                )));
            }
            for block_row in 0..merged_h {
                for block_column in 0..merged_w {
                    let slot = index + block_row * merged_w + block_column;
                    positions[0][slot] = current;
                    positions[1][slot] = current + block_row as i32;
                    positions[2][slot] = current + block_column as i32;
                    slots.push(slot as i32);
                }
            }
            current += merged_h.max(merged_w) as i32;
        } else {
            for offset in index..end {
                let value = current + (offset - index) as i32;
                positions[0][offset] = value;
                positions[1][offset] = value;
                positions[2][offset] = value;
            }
            current += (end - index) as i32;
        }
        index = end;
    }
    if grid_index != grids.len() {
        return Err(Error::Other(format!(
            "d3-lite prompt consumed {grid_index} image grids but the request passed {}",
            grids.len()
        )));
    }
    if slots.is_empty() {
        return Err(Error::Other(
            "d3-lite image request found no image_token_id placeholders".into(),
        ));
    }
    let flat = positions.into_iter().flatten().collect();
    Ok((flat, slots))
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

/// How the attention layers receive rotary positions. The text path uses the
/// fused half-split kernel with 1-D positions (unchanged from P1a); an image
/// request uses mRoPE cos/sin recomposed from the `[3, L]` multimodal
/// positions, exactly as the reference's `Qwen3_5TextRotaryEmbedding` does.
enum RopeMode<'a> {
    HalfSplit(&'a Array),
    Mrope { cos: &'a Array, sin: &'a Array },
}

/// The whole text request graph. `ids` is `[1, L]` I32, `candidate_count` is
/// `[1]` I32; the returned vector is `[probabilities [1, 255]]` in F32, with
/// every code at or past `candidate_count` set to `-inf` before the softmax so
/// the result is exactly the masked softmax the reference produces.
fn forward(parts: &Parts, ids: &Array, candidate_count: &Array) -> Result<Vec<Array>> {
    let [1, length] = *ids.shape() else {
        return Err(Error::Contract("d3-lite ids must be [1, L]"));
    };
    if length == 0 {
        return Err(Error::Contract("d3-lite ids must be nonempty"));
    }
    let hidden = parts.weights.embed.take(ids, 0)?;
    let positions = Array::arange(&parts.stream, 0.0, length as f32, 1.0, MlxDType::I32)?
        .reshape(&[1, length])?;
    let hidden = stack(
        parts,
        hidden,
        &RopeMode::HalfSplit(&positions),
        &parts.recurrence,
    )?;
    readout_tail(parts, &hidden, candidate_count)
}

/// The whole image request graph: embedding lookup, the eager vision tower's
/// merged tokens written over the single contiguous `image_token_id` run, the
/// same 24-layer stack under mRoPE, and the FP32 255-code readout. `positions`
/// is `[3, L]` I32, `features` `[1, M, hidden]` BF16 and `run_start` `[1]` I32
/// — MLX's dynamic `slice_update` takes one start index per axis, so the
/// placeholders must form one contiguous run (the processor's expansion of a
/// single image).
fn image_forward(
    parts: &Parts,
    ids: &Array,
    positions: &Array,
    candidate_count: &Array,
    features: &Array,
    run_start: &Array,
) -> Result<Vec<Array>> {
    let [1, length] = *ids.shape() else {
        return Err(Error::Contract("d3-lite ids must be [1, L]"));
    };
    let [3, position_length] = *positions.shape() else {
        return Err(Error::Contract("d3-lite mrope positions must be [3, L]"));
    };
    if position_length != length {
        return Err(Error::Other(format!(
            "d3-lite mrope positions cover {position_length} tokens, ids cover {length}"
        )));
    }
    let [1, image_tokens, hidden_width] = *features.shape() else {
        return Err(Error::Contract(
            "d3-lite merged image features must be [1, M, hidden]",
        ));
    };
    if hidden_width != parts.geometry.hidden {
        return Err(Error::Other(format!(
            "d3-lite merged image features have width {hidden_width}, expected {}",
            parts.geometry.hidden
        )));
    }
    if image_tokens == 0 || run_start.shape() != [1] {
        return Err(Error::Other(format!(
            "d3-lite image scatter needs a nonempty feature run and a [1] start, got {:?} and {:?}",
            features.shape(),
            run_start.shape()
        )));
    }
    let mut embeds = parts.weights.embed.take(ids, 0)?;
    embeds = embeds.slice_update(features, run_start, &[1])?;
    let (cos, sin) = mrope_rotary(parts, positions, length)?;
    let hidden = stack(
        parts,
        embeds,
        &RopeMode::Mrope {
            cos: &cos,
            sin: &sin,
        },
        &parts.recurrence_image,
    )?;
    readout_tail(parts, &hidden, candidate_count)
}

/// The 24-layer stack under one rotary contract and one recurrence layout,
/// ending at the final RMSNorm. The op order is shared by the text and image
/// callables; they differ in the rotary arguments and, independently, in the
/// GDN recurrence layout each path is allowed to use.
fn stack(
    parts: &Parts,
    mut hidden: Array,
    rope: &RopeMode<'_>,
    recurrence: &GatedDeltaRecurrence,
) -> Result<Array> {
    let geometry = parts.geometry;
    let length = hidden.shape()[1];
    for (index, layer) in parts.weights.layers.iter().enumerate() {
        let normalized = hidden.fast_rms_norm(&layer.input_norm, geometry.eps)?;
        let mixed = if geometry.is_attention(index) {
            let Mixer::Attention(attention) = &layer.mixer else {
                unreachable!("the validated schedule selects attention here")
            };
            attention_forward(parts, &normalized, attention, rope)?
        } else {
            let Mixer::GatedDelta(gated_delta) = &layer.mixer else {
                unreachable!("the validated schedule selects gated delta here")
            };
            gated_delta_forward(parts, recurrence, &normalized, gated_delta, length)?
        };
        let residual = hidden.add(&mixed)?;
        let mlp_input = residual.fast_rms_norm(&layer.post_norm, geometry.eps)?;
        let mlp_output = mlp_forward(&mlp_input, &layer.mlp)?;
        hidden = residual.add(&mlp_output)?;
    }
    hidden.fast_rms_norm(&parts.weights.norm, geometry.eps)
}

/// Last position → FP32 → the 255-code readout → masked FP32 softmax.
fn readout_tail(parts: &Parts, hidden: &Array, candidate_count: &Array) -> Result<Vec<Array>> {
    let geometry = parts.geometry;
    let length = hidden.shape()[1];
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

/// mRoPE cos/sin for one image request: `[3, L]` positions × the 32 inverse
/// frequencies, recomposed with the interleaved `[T | H | W]` sections
/// (`i % 3`) and duplicated over the 64-wide rotary block, all in F32 then
/// rounded once to the embedding dtype, exactly as the reference does.
fn mrope_rotary(parts: &Parts, positions: &Array, length: usize) -> Result<(Array, Array)> {
    let width = parts.geometry.rope_dims;
    let frequency_count = parts.inv_freq.shape()[0];
    let frequencies = positions
        .cast(MlxDType::F32)?
        .reshape(&[3, length, 1])?
        .mul(&parts.inv_freq.reshape(&[1, 1, frequency_count])?)?;
    let text = frequencies
        .slice_axis(0, 0, 1)?
        .reshape(&[length, frequency_count])?;
    let height = frequencies
        .slice_axis(0, 1, 2)?
        .reshape(&[length, frequency_count])?;
    let column = frequencies
        .slice_axis(0, 2, 3)?
        .reshape(&[length, frequency_count])?;
    // mrope_section [11, 11, 10] over 32 frequencies is the interleaved
    // `i % 3`: T, H, W, T, H, W, ... .
    let partition_ids = (0..frequency_count as i32)
        .map(|index| index % 3)
        .collect::<Vec<_>>();
    let partition = Array::from_i32(&parts.stream, &[frequency_count], &partition_ids)?;
    let zero = Array::scalar(&parts.stream, 0.0, MlxDType::I32)?;
    let one = Array::scalar(&parts.stream, 1.0, MlxDType::I32)?;
    let is_text = partition.equal(&zero)?;
    let is_height = partition.equal(&one)?;
    let spatial = is_height.where_select(&height, &column)?;
    let recomposed = is_text.where_select(&text, &spatial)?;
    let duplicate = |values: Array| -> Result<Array> { Array::concat(&[&values, &values], 1) };
    let cos = duplicate(recomposed.cos()?.cast(MlxDType::BF16)?)?;
    let sin = duplicate(recomposed.sin()?.cast(MlxDType::BF16)?)?;
    debug_assert_eq!(cos.shape(), &[length, width]);
    Ok((cos, sin))
}

/// Apply mRoPE to one `[B, H, L, D]` tensor: the first `rotary` features are
/// paired half-and-half and rotated by the recomposed cos/sin (BF16 ops, one
/// rounding per operation, as the reference's `apply_rotary_pos_emb` does);
/// the remaining features pass through untouched.
fn mrope_apply(tensor: &Array, cos: &Array, sin: &Array, rotary: usize) -> Result<Array> {
    let [batch, heads, length, width] = *tensor.shape() else {
        return Err(Error::Contract("d3-lite rotary input must be [B, H, L, D]"));
    };
    if rotary == 0 || rotary > width || rotary % 2 != 0 || cos.shape() != [length, rotary] {
        return Err(Error::Other(format!(
            "d3-lite mrope needs even rotary width {rotary} ≤ {width} and cos/sin [{length}, {rotary}]"
        )));
    }
    let active = tensor.slice_axis(3, 0, rotary)?;
    let first = active.slice_axis(3, rotary / 2, rotary)?;
    let second = active.slice_axis(3, 0, rotary / 2)?;
    let rotated = Array::concat(&[&first.neg()?, &second], 3)?;
    let cos = cos.reshape(&[1, 1, length, rotary])?;
    let sin = sin.reshape(&[1, 1, length, rotary])?;
    let active = active.mul(&cos)?.add(&rotated.mul(&sin)?)?;
    if rotary == width {
        debug_assert_eq!(active.shape(), &[batch, heads, length, width]);
        return Ok(active);
    }
    Array::concat(&[&active, &tensor.slice_axis(3, rotary, width)?], 3)
}

fn mlp_forward(input: &Array, mlp: &Mlp) -> Result<Array> {
    let gate = project(input, &mlp.gate.weight)?;
    let up = project(input, &mlp.up.weight)?;
    let activated = silu(&gate)?.mul(&up)?;
    project(&activated, &mlp.down.weight)
}

fn gated_delta_forward(
    parts: &Parts,
    recurrence: &GatedDeltaRecurrence,
    input: &Array,
    gated_delta: &GatedDelta,
    length: usize,
) -> Result<Array> {
    let geometry = parts.geometry;
    let key_dim = geometry.key_dim();
    let value_dim = geometry.value_dim();
    let conv_dim = geometry.conv_dim();
    let qkv = project(input, &gated_delta.qkv.weight)?;
    let z = project(input, &gated_delta.z.weight)?.reshape(&[
        1,
        length,
        geometry.v_heads,
        geometry.v_dim,
    ])?;
    let b = project(input, &gated_delta.b.weight)?;
    let a = project(input, &gated_delta.a.weight)?;
    let convolved = match parts.conv1d {
        Conv1d::Stock => {
            // Stock reference: a depthwise causal convolution over `S - 1`
            // zero rows, composed from MLX primitives.
            let history = Array::zeros(
                &parts.stream,
                &[1, geometry.conv_kernel - 1, conv_dim],
                qkv.dtype(),
            )?;
            let convolved = Array::concat(&[&history, &qkv], 1)?.conv1d(
                &gated_delta.conv,
                1,
                0,
                1,
                conv_dim,
            )?;
            silu(&convolved)?
        }
        Conv1d::ApxInf => {
            let kernel = parts.causal_conv.as_ref().ok_or_else(|| {
                Error::Contract("APXINF_D3_CONV1D=apxinf was selected without its kernel")
            })?;
            // Same arithmetic contract — causal depthwise conv with zero
            // history, FP32 accumulation in tap order, one BF16 rounding —
            // but one launch with no zero-history concat and no im2col-style
            // padding materialization.
            silu(&kernel.call(&qkv, &gated_delta.conv)?)?
        }
    };
    let queries = convolved.slice_axis(2, 0, key_dim)?.reshape(&[
        1,
        length,
        geometry.k_heads,
        geometry.k_dim,
    ])?;
    let keys = convolved.slice_axis(2, key_dim, 2 * key_dim)?.reshape(&[
        1,
        length,
        geometry.k_heads,
        geometry.k_dim,
    ])?;
    let values = convolved.slice_axis(2, 2 * key_dim, conv_dim)?.reshape(&[
        1,
        length,
        geometry.v_heads,
        geometry.v_dim,
    ])?;
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
        recurrence.call(&queries, &keys, &values, &decay, &beta, &state)?;
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
    rope: &RopeMode<'_>,
) -> Result<Array> {
    let geometry = parts.geometry;
    let [1, length, _] = *input.shape() else {
        return Err(Error::Contract("d3-lite attention input must be [1, L, H]"));
    };
    // One packed projection yields queries and the output gate: `[B, L, 2*heads*head_dim]`.
    let projected = project(input, &attention.q.weight)?.reshape(&[
        1,
        length,
        geometry.q_heads,
        2 * geometry.head_dim,
    ])?;
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
    let (queries, keys) = match rope {
        RopeMode::HalfSplit(positions) => (
            parts.rope.call(&queries, positions, &parts.inv_freq)?,
            parts.rope.call(&keys, positions, &parts.inv_freq)?,
        ),
        RopeMode::Mrope { cos, sin } => (
            mrope_apply(&queries, cos, sin, geometry.rope_dims)?,
            mrope_apply(&keys, cos, sin, geometry.rope_dims)?,
        ),
    };
    // d3's full-attention layers are explicitly non-causal over the prompt.
    let output = queries.sdpa(
        &keys,
        &values,
        (geometry.head_dim as f32).powf(-0.5),
        false,
        None,
    )?;
    let output = output.transpose(&[0, 2, 1, 3])?.reshape(&[
        1,
        length,
        geometry.q_heads * geometry.head_dim,
    ])?;
    project(&output.mul(&gate.sigmoid()?)?, &attention.o.weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream() -> Stream {
        Stream::cpu().expect("CPU stream for eager semantics")
    }

    #[test]
    fn engine_gates_default_to_the_first_value() {
        for (name, values) in [
            ("APXINF_D3_CONV1D", CONV1D_VALUES),
            (GDN_CORE, GDN_CORE_VALUES),
            (GDN_CORE_IMAGE, GDN_CORE_IMAGE_VALUES),
        ] {
            assert_eq!(parse_gate(None, values, name).unwrap(), 0, "{name} unset");
            for (index, value) in values.iter().enumerate() {
                assert_eq!(
                    parse_gate(Some(value), values, name).unwrap(),
                    index,
                    "{name}={value}"
                );
            }
            let error = parse_gate(Some("true"), values, name)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(name) && error.contains(&values.join(", ")),
                "{name} rejected value must name itself and the list: {error}"
            );
        }
        // The two D3 gates have distinct value lists; the parse is generic.
        assert_ne!(CONV1D_VALUES, GDN_CORE_VALUES);
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
        // Equal inputs across the two heads make `A_log` the only variable:
        // softplus(0) = ln 2, so decay is 0.5 vs exp(-e * ln 2) ≈ 0.152.
        let a = Array::zeros(&stream, &[1, 3, 2], MlxDType::BF16).unwrap();
        let dt_bias = Array::zeros(&stream, &[2], MlxDType::BF16).unwrap();
        let b = Array::from_f32(&stream, &[1, 3, 2], &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
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
        assert!(decay
            .iter()
            .all(|value| value.is_finite() && *value > 0.0 && *value <= 1.0));
        let ln_two = 2f32.ln();
        let expected = [(-ln_two).exp(), (-(std::f32::consts::E * ln_two)).exp()];
        assert!(
            (decay[0] - expected[0]).abs() < 2e-3,
            "{} vs {}",
            decay[0],
            expected[0]
        );
        assert!(
            (decay[1] - expected[1]).abs() < 2e-3,
            "{} vs {}",
            decay[1],
            expected[1]
        );
        assert!(decay[0] > decay[1], "larger A_log must decay faster");
        assert!(beta.iter().all(|value| (0.0..1.0).contains(value)));
        // b = -2 → sigmoid ≈ 0.119; b = 3 → ≈ 0.953.
        assert!((beta[0] - (-2.0f32).exp() / (1.0 + (-2.0f32).exp())).abs() < 3e-3);
        assert!((beta[5] - 3.0f32.exp() / (1.0 + 3.0f32.exp())).abs() < 3e-3);
    }

    #[test]
    fn project_is_linear_without_bias() {
        let stream = stream();
        let weight = Array::from_f32(&stream, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let input =
            Array::from_f32(&stream, &[1, 2, 3], &[1.0, 0.0, -1.0, 0.5, 2.0, -0.5]).unwrap();
        let output = project(&input, &weight).unwrap();
        stream.eval(&[output.clone()]).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2]);
        assert_eq!(output.to_f32_vec().unwrap(), [-2.0, -2.0, 3.0, 9.0],);
    }

    /// End-to-end device acceptance for the prepared whole-request callable:
    /// compiled output matches the eager body, replay traces nothing new,
    /// changed inputs reach the output, and a new length traces exactly once.
    #[test]
    #[ignore = "requires APXINF_D3_CHECKPOINT and an Apple Metal GPU"]
    fn compiled_forward_matches_eager_replays_and_binds_changed_inputs() -> Result<()> {
        let directory = std::env::var_os("APXINF_D3_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_CHECKPOINT to the d3-lite package directory");
        let config = Config::from_dir(&directory)?;
        let stream = Stream::metal(0)?;
        let weights = Weights::load(config.geometry, &stream, &directory)?;
        let vision_weights = VisionWeights::load(config.vision, &stream, &directory)?;
        let model = Model::new(&config, stream.clone(), weights, vision_weights)?;
        let count = config.codes.len();
        let tokens = 37usize;
        let first = vec![1u32; tokens];
        let eager = model.eager_probabilities(&first, count)?;
        let compiled = model.probabilities(&first, count)?;
        let max = eager
            .iter()
            .zip(&compiled)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-4, "compiled vs eager max|Δp| = {max}");

        // Replay: the trace callback must not run again for a known profile.
        let before = stream.counters();
        let replay = model.probabilities(&first, count)?;
        let after = stream.counters();
        assert_eq!(before.trace_callbacks, after.trace_callbacks);
        assert!(
            replay
                .iter()
                .zip(&compiled)
                .all(|(left, right)| left.to_bits() == right.to_bits()),
            "prepared replay changed values"
        );

        // Changed token IDs must reach the published probabilities.
        let changed = model.probabilities(&vec![2u32; tokens], count)?;
        let changed_delta = compiled
            .iter()
            .zip(&changed)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(
            changed_delta > 0.0,
            "changed token IDs did not change probabilities"
        );

        // A second length traces once; repeating it traces nothing.
        let longer = vec![1u32; 64];
        model.probabilities(&longer, count)?;
        let before = stream.counters();
        model.probabilities(&longer, count)?;
        let after = stream.counters();
        assert_eq!(before.trace_callbacks, after.trace_callbacks);
        assert_eq!(
            model.prepared_profiles(),
            vec![(tokens, count), (64, count)]
        );
        Ok(())
    }

    /// Vision-tower acceptance: the native `visual.*` forward reproduces the
    /// checkpoint's PyTorch pooler output (`image_features`) on the same
    /// preprocessed pixels. This localizes image-path drift between the
    /// vision tower and the text layers + mRoPE + readout.
    #[test]
    #[ignore = "requires APXINF_D3_CHECKPOINT, APXINF_D3_VISION_REF and an Apple Metal GPU"]
    fn vision_tower_matches_pytorch_reference() -> Result<()> {
        use serde::Deserialize;
        use std::collections::HashMap;

        #[derive(Deserialize)]
        struct Receipt {
            cases: Vec<Case>,
        }
        #[derive(Deserialize)]
        struct Case {
            id: String,
            grid_thw: Vec<u32>,
            arrays: Arrays,
        }
        #[derive(Deserialize)]
        struct Arrays {
            pixel_values: Option<ArrayFile>,
            image_features: Option<ArrayFile>,
        }
        #[derive(Deserialize)]
        struct ArrayFile {
            file: String,
        }
        #[derive(Deserialize)]
        struct StageCase {
            stages: HashMap<String, ArrayFile>,
        }

        fn read_f32(base: &std::path::Path, file: &str) -> Result<Vec<f32>> {
            let path = base.join(file);
            let bytes = std::fs::read(&path)
                .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect())
        }

        let checkpoint = std::env::var_os("APXINF_D3_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_CHECKPOINT to the d3-lite package directory");
        let reference = std::env::var_os("APXINF_D3_VISION_REF")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_VISION_REF to the reference cases.json");
        let base = reference
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let receipt: Receipt = serde_json::from_str(
            &std::fs::read_to_string(&reference)
                .map_err(|error| Error::Other(format!("read {}: {error}", reference.display())))?,
        )
        .map_err(|error| Error::Other(format!("parse {}: {error}", reference.display())))?;

        let config = Config::from_dir(&checkpoint)?;
        let stream = Stream::metal(0)?;
        let weights = Weights::load(config.geometry, &stream, &checkpoint)?;
        let vision_weights = VisionWeights::load(config.vision, &stream, &checkpoint)?;
        let model = Model::new(&config, stream.clone(), weights, vision_weights)?;

        // Optional per-stage receipt (dump_reference_vision_stages.py), which
        // localizes a divergence to patch embed, position add, one block or
        // the merger.
        let stage_index: HashMap<String, StageCase> =
            match std::env::var_os("APXINF_D3_VISION_STAGES") {
                Some(path) => {
                    let text = std::fs::read_to_string(&path).map_err(|error| {
                        Error::Other(format!("read {}: {error}", path.to_string_lossy()))
                    })?;
                    serde_json::from_str(&text).map_err(|error| {
                        Error::Other(format!("parse {}: {error}", path.to_string_lossy()))
                    })?
                }
                None => HashMap::new(),
            };

        let mut worst = 0.0f32;
        let mut worst_relative = 0.0f32;
        let mut worst_case = String::new();
        for case in &receipt.cases {
            let pixels =
                case.arrays.pixel_values.as_ref().ok_or_else(|| {
                    Error::Other(format!("{} has no pixel_values array", case.id))
                })?;
            let features =
                case.arrays.image_features.as_ref().ok_or_else(|| {
                    Error::Other(format!("{} has no image_features array", case.id))
                })?;
            let pixels = read_f32(base, &pixels.file)?;
            let expected_final = read_f32(base, &features.file)?;
            if case.grid_thw.len() != 3 {
                return Err(Error::Other(format!(
                    "{} grid_thw must have 3 entries",
                    case.id
                )));
            }
            let grids = [[case.grid_thw[0], case.grid_thw[1], case.grid_thw[2]]];
            let native_stages = model.parts.vision.forward_stages(&pixels, &grids)?;
            let stages = stage_index.get(&case.id);
            let mut case_worst = 0.0f32;
            println!("{}: grid={grids:?}", case.id);
            for (name, native_values) in &native_stages {
                let expected = match *name {
                    "merged" => Some(expected_final.clone()),
                    _ => stages
                        .and_then(|case| case.stages.get(*name))
                        .map(|file| read_f32(base, &file.file))
                        .transpose()?,
                };
                let Some(expected) = expected else { continue };
                if native_values.len() != expected.len() {
                    return Err(Error::Other(format!(
                        "{} {name}: native has {} values, reference has {}",
                        case.id,
                        native_values.len(),
                        expected.len()
                    )));
                }
                let stage_worst = native_values
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| (*actual - *expected).abs())
                    .fold(0.0f32, f32::max);
                let scale = expected
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-6);
                println!(
                    "  {name:<13} max|Δ|={stage_worst:<10.6} rel={}",
                    stage_worst / scale
                );
                if *name == "merged" {
                    case_worst = stage_worst;
                    worst_relative = worst_relative.max(stage_worst / scale);
                }
            }
            if case_worst > worst {
                worst = case_worst;
                worst_case = case.id.clone();
            }
        }
        // The tower's late residual stream is ill-conditioned: the
        // checkpoint's own reference moves by 4–46 % across the eight cases
        // between MPS and CPU, while native Metal moves 8–28 %. Bound native
        // inside the reference's own device-to-device envelope (46 % → 50 %)
        // rather than at a rounding-scale delta.
        assert!(
            worst_relative < 0.50,
            "native vision features differ from the reference by {worst} ({worst_relative:.3} relative) in {worst_case}"
        );
        Ok(())
    }

    /// Image-path acceptance through the **public** runtime: registry load →
    /// `D3LiteRunner::decide` → `Model::image_probabilities`, compared with
    /// the checkpoint's PyTorch reference. The declared probability budget is
    /// 3× the reference's own MPS↔CPU noise (0.0201) because the vision
    /// tower's residual stream amplifies BF16 kernel-order differences.
    #[test]
    #[ignore = "requires APXINF_D3_CHECKPOINT, APXINF_D3_VISION_REF and an Apple Metal GPU"]
    fn image_decision_matches_pytorch_reference_through_the_public_path() -> Result<()> {
        use crate::{AutoModel, DecisionInput, DecisionModel, ImageInput, LoadOptions};

        const MAX_IMAGE_ABS_DELTA: f32 = 0.06;

        #[derive(serde::Deserialize)]
        struct Receipt {
            cases: Vec<Case>,
        }
        #[derive(serde::Deserialize)]
        struct Case {
            id: String,
            token_ids: Vec<u32>,
            option_count: usize,
            grid_thw: Vec<u32>,
            probabilities: Vec<f32>,
            arrays: Arrays,
        }
        #[derive(serde::Deserialize)]
        struct Arrays {
            pixel_values: ArrayFile,
        }
        #[derive(serde::Deserialize)]
        struct ArrayFile {
            file: String,
        }

        let checkpoint = std::env::var_os("APXINF_D3_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_CHECKPOINT to the d3-lite package directory");
        let reference = std::env::var_os("APXINF_D3_VISION_REF")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_VISION_REF to the reference cases.json");
        let base = reference
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let receipt: Receipt = serde_json::from_str(
            &std::fs::read_to_string(&reference)
                .map_err(|error| Error::Other(format!("read {}: {error}", reference.display())))?,
        )
        .map_err(|error| Error::Other(format!("parse {}: {error}", reference.display())))?;

        let mut loaded = AutoModel::load_model(
            apxinf_core::Device::Metal(0),
            &checkpoint,
            &LoadOptions::default(),
        )?;
        assert_eq!(
            loaded
                .decision_capabilities()
                .map_err(|error| Error::Other(error.to_string()))?,
            crate::llm_trait::LlmCapabilities::VISION
        );
        let DecisionModel::Code(runtime) = loaded
            .decision_model_mut()
            .map_err(|error| Error::Other(error.to_string()))?
        else {
            return Err(Error::Contract("d3-lite loads as a code decision model"));
        };

        let mut worst = 0.0f32;
        let mut worst_case = String::new();
        let mut flips = Vec::new();
        for case in &receipt.cases {
            if case.grid_thw.len() != 3 {
                return Err(Error::Other(format!(
                    "{} grid_thw must have 3 entries",
                    case.id
                )));
            }
            let path = base.join(&case.arrays.pixel_values.file);
            let bytes = std::fs::read(&path)
                .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
            let pixels = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            let rows = case.grid_thw[1] as usize * case.grid_thw[2] as usize;
            if pixels.len() != rows * 1_536 {
                return Err(Error::Other(format!(
                    "{}: {} pixel values for {rows} patches",
                    case.id,
                    pixels.len()
                )));
            }
            let pixel_tensor = apxinf_core::Tensor::from_f32(vec![rows, 1_536], &pixels)?;
            let grids = vec![[case.grid_thw[0], case.grid_thw[1], case.grid_thw[2]]];
            let output = runtime.decide(DecisionInput::with_vision(
                &case.token_ids,
                ImageInput::new(&pixel_tensor, &grids),
                case.option_count,
            ))?;
            let probabilities = output
                .scores
                .iter()
                .map(|score| score.probability)
                .collect::<Vec<_>>();
            let expected = &case.probabilities[..case.option_count.min(case.probabilities.len())];
            let decision = {
                let mut best = 0usize;
                for (index, value) in probabilities.iter().enumerate() {
                    if *value > probabilities[best] {
                        best = index;
                    }
                }
                best
            };
            let reference_decision = {
                let mut best = 0usize;
                for (index, value) in expected.iter().enumerate() {
                    if *value > expected[best] {
                        best = index;
                    }
                }
                best
            };
            let delta = probabilities
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (*actual - *expected).abs())
                .fold(0.0f32, f32::max);
            println!(
                "{}: decisions {decision}/{reference_decision} max|Δp|={delta:.6}",
                case.id
            );
            if decision != reference_decision {
                flips.push(case.id.clone());
            }
            if delta > worst {
                worst = delta;
                worst_case = case.id.clone();
            }
        }
        assert!(flips.is_empty(), "image decisions flipped: {flips:?}");
        assert!(
            worst <= MAX_IMAGE_ABS_DELTA,
            "image max|Δp| {worst} in {worst_case} exceeds {MAX_IMAGE_ABS_DELTA}"
        );
        Ok(())
    }
}
