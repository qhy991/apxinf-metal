//! Native MiniCPM5 forward and request state. No Python model provider.
//! Arithmetic/schedule adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin.
use super::{
    config::{attention_width, Config, PrefillChunk, PrefillPlan, Variant},
    math,
    weights::Weights,
};
#[cfg(feature = "coreml")]
use super::config::{SerialReason, WavefrontPlan, WavefrontSegment};
use crate::{
    GenerationBoundary, LlmInput, LlmTrait, TextCompilationScope, TextPreparationState, TextPreparationStatus,
};
use apxinf_core::{Backend, Device, Error, Result, Tensor};
use apxinf_loader::ModelConfig;
use apxinf_mlx::fusions::PackedResidualRmsNorm;
use apxinf_mlx::{Array, Compiled, MlxBackend, MlxDType};
use std::{collections::HashMap, rc::Rc};

struct LocalMath {
    norm: Compiled,
    rope: Compiled,
}

impl LocalMath {
    fn new(variant: Variant, epsilon: &Array, rms_norm_eps: f32) -> Result<Option<Self>> {
        if variant == Variant::Public {
            return Ok(None);
        }
        let stream = epsilon.stream();
        let eps = epsilon.clone();
        let native_norm = variant == Variant::Compiled;
        Ok(Some(Self {
            norm: Compiled::with_shapeless(stream, 1, true, move |a| {
                let normalized = if native_norm {
                    a[0].fast_rms_norm(&a[1], rms_norm_eps)?
                } else {
                    math::norm(&a[0], &a[1], &eps)?
                };
                Ok(vec![normalized])
            })?,
            rope: Compiled::new(stream, 1, |a| Ok(vec![math::rotate(&a[0], &a[1], &a[2])?]))?,
        }))
    }
}

enum RequiredOutput {
    CacheOnly,
    Logits { last_only: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeAttention {
    Bucketed,
    Exact,
}

enum DecodeAttentionPolicy {
    Bucketed,
    Exact,
}

impl DecodeAttentionPolicy {
    fn from_override(value: Option<&std::ffi::OsStr>) -> Self {
        match value {
            Some(value) if value == "1" => Self::Exact,
            _ => Self::Bucketed,
        }
    }

    fn resolve(&self, variant: Variant) -> DecodeAttention {
        if variant != Variant::Compiled {
            return DecodeAttention::Bucketed;
        }
        match self {
            Self::Exact => DecodeAttention::Exact,
            _ => DecodeAttention::Bucketed,
        }
    }
}

#[cfg(all(test, feature = "coreml"))]
mod wavefront_mask_tests {
    use super::*;

    /// Reads a `[1, 1, rows, read]` boolean mask on the CPU stream.
    fn mask(stream: &apxinf_mlx::Stream, position: usize, rows: usize, read: usize) -> Vec<bool> {
        let mask = segment_mask(stream, position, rows, read)
            .unwrap()
            .to_bytes()
            .unwrap();
        mask.into_iter().map(|byte| byte != 0).collect()
    }

    #[test]
    fn first_chunk_mask_is_causal_over_its_own_rows() {
        let stream = apxinf_mlx::Stream::cpu().unwrap();
        let (rows, read) = (5, 5);
        let visible = mask(&stream, 0, rows, read);
        for row in 0..rows {
            for column in 0..read {
                assert_eq!(
                    visible[row * read + column],
                    column <= row,
                    "first chunk row {row} column {column}"
                );
            }
        }
    }

    /// The property the whole schedule rests on: every second-chunk query sees
    /// all first-chunk keys, plus its own causal prefix. If this mask were
    /// causal over the full key window instead, the wavefront would silently
    /// drop the first half of the prompt from the second half's attention.
    #[test]
    fn second_chunk_mask_sees_every_first_chunk_key() {
        let stream = apxinf_mlx::Stream::cpu().unwrap();
        let (half0, half1) = (4, 3);
        let prompt = half0 + half1;
        let visible = mask(&stream, half0, half1, prompt);
        for row in 0..half1 {
            let query = half0 + row;
            for column in 0..prompt {
                assert_eq!(
                    visible[row * prompt + column],
                    column <= query,
                    "second chunk row {row} column {column}"
                );
            }
            // The first chunk's rows are all strictly before this query.
            assert!(visible[row * prompt..row * prompt + half0]
                .iter()
                .all(|&v| v));
        }
    }

    /// The second chunk's first query can only read its own row, not the rest
    /// of the second chunk, so no future token leaks in either.
    #[test]
    fn second_chunk_mask_hides_its_own_future_rows() {
        let stream = apxinf_mlx::Stream::cpu().unwrap();
        let (half0, half1) = (2, 6);
        let prompt = half0 + half1;
        let visible = mask(&stream, half0, half1, prompt);
        for row in 0..half1 {
            for column in half0..prompt {
                assert_eq!(
                    visible[row * prompt + column],
                    column - half0 <= row,
                    "second chunk local row {row} column {column}"
                );
            }
        }
    }

    #[test]
    fn mask_rejects_queries_outside_the_read_window() {
        let stream = apxinf_mlx::Stream::cpu().unwrap();
        assert!(segment_mask(&stream, 4, 3, 4).is_err());
        assert!(segment_mask(&stream, 0, 0, 4).is_err());
        assert!(segment_mask(&stream, 0, 4, 4).is_ok());
    }
}

#[cfg(test)]
mod attention_policy_tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn default_attention_remains_bucketed_for_every_variant() {
        let policy = DecodeAttentionPolicy::from_override(None);
        for variant in [Variant::Compiled, Variant::Public, Variant::DSpark] {
            assert_eq!(policy.resolve(variant), DecodeAttention::Bucketed);
        }
    }

    #[test]
    fn explicit_attention_override_preserves_existing_semantics() {
        let exact = DecodeAttentionPolicy::from_override(Some(OsStr::new("1")));
        assert_eq!(
            exact.resolve(Variant::Compiled),
            DecodeAttention::Exact
        );
        for variant in [Variant::Public, Variant::DSpark] {
            assert_eq!(exact.resolve(variant), DecodeAttention::Bucketed);
        }
        for value in ["0", "", "true", "invalid"] {
            let policy = DecodeAttentionPolicy::from_override(Some(OsStr::new(value)));
            assert_eq!(
                policy.resolve(Variant::Compiled),
                DecodeAttention::Bucketed
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_override_selects_bucketed_attention() {
        use std::os::unix::ffi::OsStrExt;
        let policy = DecodeAttentionPolicy::from_override(Some(OsStr::from_bytes(&[0xff])));
        assert_eq!(
            policy.resolve(Variant::Compiled),
            DecodeAttention::Bucketed
        );
    }
}

struct PreparedRequest {
    prefill: PrefillPlan,
    output: usize,
    attention: DecodeAttention,
    positions: HashMap<usize, Array>,
    /// Two-chunk overlap schedule for this request, or `None` when the prepared
    /// request must keep the serial prefill path.
    #[cfg(feature = "coreml")]
    wavefront: Option<WavefrontPlan>,
}

pub struct MiniCpm5 {
    pub(crate) backend: MlxBackend,
    pub(crate) weights: Rc<Weights>,
    config: Config,
    variant: Variant,
    prefill_chunk: Option<PrefillChunk>,
    epsilon: Array,
    cos: Array,
    sin: Array,
    local: Option<LocalMath>,
    swiglu: Compiled,
    step: Option<Compiled>,
    attention_policy: DecodeAttentionPolicy,
    prepared: Option<PreparedRequest>,
    retained_preparation: bool,
    states: Vec<Array>,
    capacity: usize,
    pub(crate) offset: usize,
    pub(crate) last_taps: Option<Array>,
    failure: Option<String>,
    /// Layers whose MLP runs as one FP16 Core ML program per layer on the
    /// Neural Engine. Empty unless `APXINF_MINICPM5_ANE_MLP` selects layers.
    #[cfg(feature = "coreml")]
    ane: Vec<Option<super::ane::AneMlp>>,
    /// `APXINF_MINICPM5_ANE_OVERLAP`: run the two-chunk wavefront prefill on
    /// requests that admit it instead of the serial chunked schedule.
    #[cfg(feature = "coreml")]
    overlap: bool,
}

/// Builds the per-layer Neural Engine programs named by
/// `APXINF_MINICPM5_ANE_MLP`, or an empty vector when the knob is absent.
///
/// The canonical weights are `[in, out]`; the program wants the transposed
/// `[out, in]` FP16 copy, so the host conversion transposes while narrowing
/// BF16 to FP16. Programs are compiled and cached once, warmed once, and the
/// set is fail-closed: any load failure aborts model construction rather than
/// silently mixing a partially ANE schedule.
#[cfg(feature = "coreml")]
fn ane_programs(
    stream: &apxinf_mlx::Stream,
    layers: &[super::weights::LayerWeights],
) -> Result<Vec<Option<super::ane::AneMlp>>> {
    let Some(plan) = super::config::AneMlpPlan::from_env()? else {
        return Ok((0..layers.len()).map(|_| None).collect());
    };
    let selected: Vec<usize> = plan
        .layers()
        .iter()
        .copied()
        .filter(|&layer| layer < layers.len())
        .collect();
    if selected.len() != plan.layers().len() {
        return Err(Error::Contract(
            "APXINF_MINICPM5_ANE_MLP names a layer this model does not have",
        ));
    }
    let cache = apxinf_coreml::ProgramCache::new(ane_cache_dir());
    let mut programs: Vec<Option<super::ane::AneMlp>> = (0..layers.len()).map(|_| None).collect();
    for &index in &selected {
        let w = &layers[index];
        let gate = transposed_f16(&w.gate, 2048, super::ane::INTERMEDIATE)?;
        let up = transposed_f16(&w.up, 2048, super::ane::INTERMEDIATE)?;
        let down = transposed_f16(&w.down, super::ane::INTERMEDIATE, 2048)?;
        let package = super::ane::program(&gate, &up, &down)?;
        let ane = super::ane::AneMlp::load_package(&cache, &package)?;
        ane.warm()?;
        programs[index] = Some(ane);
    }
    let _ = stream;
    Ok(programs)
}

/// `array` is a canonical `[rows, columns]` BF16 weight; returns the
/// `[columns, rows]` row-major FP16 copy the program's 1x1 convolution wants.
#[cfg(feature = "coreml")]
pub(super) fn transposed_f16(array: &apxinf_mlx::Array, rows: usize, columns: usize) -> Result<Vec<half::f16>> {
    let values = array.to_f32_vec()?;
    if values.len() != rows * columns {
        return Err(Error::Contract(
            "MiniCPM5 ANE weight does not match its canonical shape",
        ));
    }
    let mut out = vec![half::f16::ZERO; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            out[column * rows + row] = half::f16::from_f32(values[row * columns + column]);
        }
    }
    Ok(out)
}

#[cfg(feature = "coreml")]
pub(super) fn ane_cache_dir() -> std::path::PathBuf {
    if let Some(path) = std::env::var_os("APXINF_MINICPM5_ANE_CACHE") {
        return std::path::PathBuf::from(path);
    }
    std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
        .join("Library/Caches/apxinf/coreml/minicpm5")
}

/// The explicit causal mask of one wavefront token chunk: query row `r` (at
/// absolute position `position + r`) may read key rows `0..=position + r`, and
/// the visible window is the first `read` rows of the layer's cache.
///
/// For the first chunk `read == rows`, so this is a plain causal mask. For the
/// second chunk `read` is the whole prompt, so every one of its queries sees
/// all first-chunk keys; that is what keeps the wavefront equal to a
/// sequential prefill of `[c0 | c1]`.
#[cfg(feature = "coreml")]
fn segment_mask(
    stream: &apxinf_mlx::Stream,
    position: usize,
    rows: usize,
    read: usize,
) -> Result<Array> {
    if rows == 0 || read < position + rows {
        return Err(Error::Contract(
            "MiniCPM5 wavefront mask needs every query row inside the read window",
        ));
    }
    let keys = Array::arange(stream, 0., read as f32, 1., MlxDType::I32)?.reshape(&[1, read])?;
    let queries = Array::arange(
        stream,
        position as f32,
        (position + rows) as f32,
        1.,
        MlxDType::I32,
    )?
    .reshape(&[rows, 1])?;
    keys.less_equal(&queries)?.reshape(&[1, 1, rows, read])
}

impl MiniCpm5 {
    pub(crate) fn supports_dspark(&self) -> bool {
        self.variant == Variant::DSpark
    }

    pub fn new(
        config: Config,
        weights: Weights,
        backend: MlxBackend,
        variant: Variant,
    ) -> Result<Self> {
        config.validate()?;
        let prefill_chunk = PrefillChunk::from_env()?;
        let stream = backend.stream();
        let epsilon = Array::scalar(stream, config.rms_norm_eps, MlxDType::F32)?;
        let (cos, sin) = math::tables(stream, 256)?;
        let local = LocalMath::new(variant, &epsilon, config.rms_norm_eps)?;
        let swiglu =
            Compiled::with_shapeless(stream, 1, true, |a| Ok(vec![math::swiglu(&a[0], &a[1])?]))?;
        let attention_policy = DecodeAttentionPolicy::from_override(
            std::env::var_os("APXINF_MINICPM5_EXACT_ATTN").as_deref(),
        );
        #[cfg(feature = "coreml")]
        let ane = ane_programs(stream, &weights.layers)?;
        #[cfg(feature = "coreml")]
        let overlap = super::config::OverlapKnob::from_env().is_on();
        Ok(Self {
            backend,
            weights: Rc::new(weights),
            config,
            variant,
            prefill_chunk,
            epsilon,
            cos,
            sin,
            local,
            swiglu,
            step: None,
            attention_policy,
            prepared: None,
            retained_preparation: false,
            states: Vec::new(),
            capacity: 0,
            offset: 0,
            last_taps: None,
            failure: None,
            #[cfg(feature = "coreml")]
            ane,
            #[cfg(feature = "coreml")]
            overlap,
        })
    }

    /// Prepare bounded KV capacity and all admitted decode graph profiles.
    /// Exact attention evaluates endpoint profiles and traces interior profiles.
    /// Bucketed attention evaluates every prepared profile.
    /// All changing tokens, positions and K/V are explicit compiled arguments.
    pub fn prepare(&mut self, prompt: usize, output: usize) -> Result<()> {
        match self.prepare_inner(prompt, output) {
            Ok(()) => Ok(()),
            Err(error) => {
                // No partially built or unsuccessfully warmed plan is ready.
                // The next request must reset and complete preparation again.
                self.prepared = None;
                self.step = None;
                self.failure = Some(error.to_string());
                Err(error)
            }
        }
    }

    fn prepare_inner(&mut self, prompt: usize, output: usize) -> Result<()> {
        let capacity = self.config.request_capacity(self.variant, prompt, output)?;
        let attention = self.attention_policy.resolve(self.variant);
        let prefill = PrefillPlan::new(
            prompt,
            self.prefill_chunk
                .unwrap_or(if self.variant == Variant::Compiled {
                    PrefillChunk::Tokens1024
                } else {
                    PrefillChunk::Tokens512
                }),
        )?;
        #[cfg(feature = "coreml")]
        let decision = self.wavefront_decision(prefill);
        #[cfg(feature = "coreml")]
        let wavefront = decision.ok();
        let same_profile = self.capacity == capacity
            && self.prepared.as_ref().is_some_and(|p| p.prefill.prompt == prompt && p.output == output && p.attention == attention)
            && self.failure.is_none();
        #[cfg(feature = "coreml")]
        let same_profile =
            same_profile && self.prepared.as_ref().is_some_and(|p| p.wavefront == wavefront);
        if same_profile && !self.states.is_empty() { return Ok(()); }
        if self.offset != 0 { return Err(Error::Contract("reset before changing MiniCPM request preparation")); }
        let reuse_bucket = attention == DecodeAttention::Bucketed && self.capacity == capacity
            && !self.states.is_empty() && self.prepared.as_ref().is_some_and(|p| p.attention == DecodeAttention::Bucketed);
        if self.retained_preparation && !same_profile {
            self.release_request_state()?;
        } else if !same_profile && !reuse_bucket {
            self.step = None;
            self.prepared = None;
        }
        if self.capacity != capacity || self.states.is_empty() {
            self.states.clear();
            if !same_profile {
                self.step = None;
                self.prepared = None;
                self.capacity = 0;
            }
            let stream = self.backend.stream();
            if self.capacity != capacity {
                let (cos, sin) = math::tables(stream, capacity)?;
                self.cos = cos;
                self.sin = sin;
            }
            let zero = Array::zeros(stream, &[1, 2, capacity, 128], MlxDType::BF16)?;
            let states = vec![zero; 84];
            stream.eval(&states)?;
            self.states = states;
            self.capacity = capacity;
        }
        if same_profile { return Ok(()); }
        if self.step.is_none() && self.variant != Variant::Public {
            self.step = Some(self.compile_step(attention)?);
        }
        // Warm shape-dependent RoPE traces used by prefill. Norm and SwiGLU
        // are shapeless; changed request values are never captured. The
        // wavefront prefill runs each token chunk through the same RoPE seam,
        // so its two row counts are traced here as well.
        #[cfg_attr(not(feature = "coreml"), allow(unused_mut))]
        let mut lengths = prefill.required_rows();
        #[cfg(feature = "coreml")]
        if let Some(wavefront) = wavefront {
            lengths.extend([wavefront.half0, wavefront.half1]);
            lengths.sort_unstable();
            lengths.dedup();
        }
        for &rows in &lengths {
            if let Some(local) = &self.local {
                let stream = self.backend.stream();
                let zeros = Array::zeros(stream, &[1, rows, 2048], MlxDType::BF16)?;
                local.norm.prepare(&[zeros, self.weights.norm.clone()])?;
                for heads in [2, 16] {
                    let zeros = Array::zeros(stream, &[1, heads, rows, 128], MlxDType::BF16)?;
                    local.rope.prepare(&[
                        zeros,
                        self.cos.slice_axis(2, 0, rows)?,
                        self.sin.slice_axis(2, 0, rows)?,
                    ])?;
                }
            }
        }
        // MLX-LM's public activation is compiled too. Its warmup is independent
        // of the optional norm/RoPE seams and covers eager prompt/decode shapes.
        for rows in lengths.into_iter().chain(std::iter::once(1)) {
            let activation = Array::zeros(self.backend.stream(), &[1, rows, 6144], MlxDType::BF16)?;
            self.swiglu.prepare(&[activation.clone(), activation])?;
        }
        let mut positions = self
            .prepared
            .take()
            .map(|p| p.positions)
            .unwrap_or_default();
        if let Some(step) = &self.step {
            // Shape-specialized DSpark verification warms all admitted row counts.
            let max_rows = if self.variant == Variant::DSpark {
                8
            } else {
                1
            };
            let (first, last, stride) = match attention {
                DecodeAttention::Exact => (prompt, prompt + output.saturating_sub(1), 1),
                DecodeAttention::Bucketed => (
                    attention_width(self.variant, prompt, capacity),
                    capacity,
                    256,
                ),
            };
            let mut width = first;
            loop {
                if !positions.contains_key(&width) {
                    let key_positions =
                        Array::arange(self.backend.stream(), 0., width as f32, 1., MlxDType::I32)?;
                    for rows in 1..=max_rows {
                        let tokens =
                            Array::from_i32(self.backend.stream(), &[1, rows], &vec![0; rows])?;
                        let position = if attention == DecodeAttention::Exact {
                            width - 1
                        } else {
                            0
                        };
                        let args =
                            self.step_inputs(tokens, position, rows, &self.states, &key_positions)?;
                        if attention == DecodeAttention::Exact && width != first && width != last {
                            step.prepare_trace(&args)?;
                        } else {
                            step.prepare(&args)?;
                        }
                    }
                    positions.insert(width, key_positions);
                }
                if width == last {
                    break;
                }
                width = (width + stride).min(last);
            }
        }
        self.failure = None;
        // One line per published profile: the serving/measurement side has no
        // other way to tell which prefill schedule this request will run, and a
        // silent fallback would make an overlap measurement meaningless.
        #[cfg(feature = "coreml")]
        match decision {
            Ok(plan) => eprintln!(
                "apxinf: MiniCPM5 ANE overlap prefill engaged: {} + {} tokens per layer, {} MLP layer(s) on the Neural Engine",
                plan.half0,
                plan.half1,
                self.ane.iter().filter(|program| program.is_some()).count()
            ),
            Err(reason) if reason != SerialReason::KnobOff => eprintln!(
                "apxinf: MiniCPM5 ANE overlap keeps the serial prefill: {}",
                reason.describe()
            ),
            Err(_) => {}
        }
        self.prepared = Some(PreparedRequest {
            prefill,
            output,
            attention,
            positions,
            #[cfg(feature = "coreml")]
            wavefront,
        });
        Ok(())
    }

    /// Resolves this request's wavefront schedule, or names why the serial
    /// schedule is kept. A request the wavefront cannot serve falls back whole:
    /// the two schedules are never mixed inside one prefill.
    #[cfg(feature = "coreml")]
    fn wavefront_decision(
        &self,
        prefill: PrefillPlan,
    ) -> std::result::Result<WavefrontPlan, SerialReason> {
        let mut chunks = prefill.chunks();
        let chunk = match (chunks.next(), chunks.next()) {
            (Some(only), None) => Some(only),
            _ => None,
        };
        let ane_layers = self.ane.iter().filter(|program| program.is_some()).count();
        WavefrontPlan::for_request(self.overlap, self.variant, ane_layers, chunk)
    }

    fn norm(&self, x: &Array, weight: &Array) -> Result<Array> {
        match &self.local {
            Some(local) => Ok(local.norm.call(&[x.clone(), weight.clone()])?.remove(0)),
            None => math::norm(x, weight, &self.epsilon),
        }
    }

    fn rotate(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        match &self.local {
            Some(local) => Ok(local
                .rope
                .call(&[x.clone(), cos.clone(), sin.clone()])?
                .remove(0)),
            None => math::rotate(x, cos, sin),
        }
    }

    fn step_inputs(
        &self,
        tokens: Array,
        position: usize,
        rows: usize,
        states: &[Array],
        positions: &Array,
    ) -> Result<Vec<Array>> {
        let mut args = vec![
            tokens,
            Array::from_i32(self.backend.stream(), &[1], &[position as i32])?,
            self.cos.slice_axis(2, position, position + rows)?,
            self.sin.slice_axis(2, position, position + rows)?,
            positions.clone(),
        ];
        args.extend(states.iter().cloned());
        Ok(args)
    }

    fn compile_step(&self, attention: DecodeAttention) -> Result<Compiled> {
        let stream = self.backend.stream();
        let packed = Rc::new(PackedResidualRmsNorm::new(
            stream,
            self.config.rms_norm_eps,
        )?);
        let weights = self.weights.clone();
        let eps = self.epsilon.clone();
        let tapped = self.variant == Variant::DSpark;
        let exact = attention == DecodeAttention::Exact;
        Compiled::new(stream, 85 + usize::from(tapped), move |a| {
            let rows = a[0].shape()[1];
            let width = a[4].shape()[0];
            let mask = if exact && rows == 1 {
                None
            } else {
                Some(math::causal_mask(&a[4], &a[1], rows)?)
            };
            let mut x = weights.embedding.take(&a[0], 0)?;
            let mut delta: Option<Array> = None;
            let mut states = Vec::with_capacity(84);
            let mut taps = Vec::new();
            let pack_rows = |x: &Array, delta: &Array, weight: &Array| -> Result<(Array, Array)> {
                let mut residuals = Vec::with_capacity(rows);
                let mut norms = Vec::with_capacity(rows);
                for row in 0..rows {
                    let (r, n) = packed.call(
                        &x.slice_axis(1, row, row + 1)?,
                        &delta.slice_axis(1, row, row + 1)?,
                        weight,
                    )?;
                    residuals.push(r);
                    norms.push(n);
                }
                Ok((
                    Array::concat(&residuals.iter().collect::<Vec<_>>(), 1)?,
                    Array::concat(&norms.iter().collect::<Vec<_>>(), 1)?,
                ))
            };
            for (i, w) in weights.layers.iter().enumerate() {
                let n = if let Some(d) = &delta {
                    let (residual, normalized) = pack_rows(&x, d, &w.input_norm)?;
                    x = residual;
                    normalized
                } else {
                    math::norm(&x, &w.input_norm, &eps)?
                };
                let project = |v: &Array, w: &Array| math::project(v, w, tapped);
                let q = project(&n, &w.q)?
                    .reshape(&[1, rows, 16, 128])?
                    .transpose(&[0, 2, 1, 3])?;
                let k = project(&n, &w.k)?
                    .reshape(&[1, rows, 2, 128])?
                    .transpose(&[0, 2, 1, 3])?;
                let v = project(&n, &w.v)?
                    .reshape(&[1, rows, 2, 128])?
                    .transpose(&[0, 2, 1, 3])?;
                let q = math::rotate(&q, &a[2], &a[3])?;
                let k = math::rotate(&k, &a[2], &a[3])?;
                let keys = a[5 + 2 * i].slice_update(&k, &a[1], &[2])?;
                let values = a[6 + 2 * i].slice_update(&v, &a[1], &[2])?;
                let attention = q
                    .sdpa(
                        &keys.slice_axis(2, 0, width)?,
                        &values.slice_axis(2, 0, width)?,
                        128f32.powf(-0.5),
                        false,
                        mask.as_ref(),
                    )?
                    .transpose(&[0, 2, 1, 3])?
                    .reshape(&[1, rows, 2048])?;
                let attention_delta = project(&attention, &w.o)?;
                let (h, n) = pack_rows(&x, &attention_delta, &w.post_norm)?;
                x = h;
                delta = Some(project(
                    &math::swiglu(&project(&n, &w.gate)?, &project(&n, &w.up)?)?,
                    &w.down,
                )?);
                states.extend([keys, values]);
                if tapped && [1, 10, 20, 30, 39].contains(&i) {
                    taps.push(pack_rows(&x, delta.as_ref().unwrap(), &w.input_norm)?.0);
                }
            }
            let (_, normalized) = pack_rows(&x, delta.as_ref().unwrap(), &weights.norm)?;
            let logits = math::project(&normalized, &weights.head, tapped)?;
            let mut output = vec![logits];
            output.extend(states);
            if tapped {
                output.push(Array::concat(&taps.iter().collect::<Vec<_>>(), 2)?);
            }
            Ok(output)
        })
    }

    fn eager(
        &self,
        tokens: &Array,
        position: usize,
        last_only: bool,
        previous: &[Array],
    ) -> Result<Vec<Array>> {
        let rows = tokens.shape()[1];
        let end = position + rows;
        let index = Array::from_i32(self.backend.stream(), &[1], &[position as i32])?;
        let cos = self.cos.slice_axis(2, position, end)?;
        let sin = self.sin.slice_axis(2, position, end)?;
        let mask = if self.variant == Variant::Compiled {
            None
        } else {
            let k_positions =
                Array::arange(self.backend.stream(), 0., end as f32, 1., MlxDType::I32)?
                    .reshape(&[1, end])?;
            let q_positions = Array::arange(
                self.backend.stream(),
                position as f32,
                end as f32,
                1.,
                MlxDType::I32,
            )?
            .reshape(&[rows, 1])?;
            let mask = k_positions
                .less_equal(&q_positions)?
                .reshape(&[1, 1, rows, end])?;
            Some(mask)
        };
        let mut x = self.weights.embedding.take(tokens, 0)?;
        let mut states = Vec::with_capacity(84);
        let mut taps = Vec::new();
        for (i, w) in self.weights.layers.iter().enumerate() {
            let n = self.norm(&x, &w.input_norm)?;
            let q = n
                .matmul(&w.q)?
                .reshape(&[1, rows, 16, 128])?
                .transpose(&[0, 2, 1, 3])?;
            let k = n
                .matmul(&w.k)?
                .reshape(&[1, rows, 2, 128])?
                .transpose(&[0, 2, 1, 3])?;
            let v = n
                .matmul(&w.v)?
                .reshape(&[1, rows, 2, 128])?
                .transpose(&[0, 2, 1, 3])?;
            let q = self.rotate(&q, &cos, &sin)?;
            let k = self.rotate(&k, &cos, &sin)?;
            let keys = previous[2 * i].slice_update(&k, &index, &[2])?;
            let values = previous[2 * i + 1].slice_update(&v, &index, &[2])?;
            let attention = q
                .sdpa(
                    &keys.slice_axis(2, 0, end)?,
                    &values.slice_axis(2, 0, end)?,
                    128f32.powf(-0.5),
                    self.variant == Variant::Compiled && rows > 1,
                    mask.as_ref(),
                )?
                .transpose(&[0, 2, 1, 3])?
                .reshape(&[1, rows, 2048])?;
            x = x.add(&attention.matmul(&w.o)?)?;
            let n = self.norm(&x, &w.post_norm)?;
            // Prefill only: a single-token decode stays on the GPU because the
            // Neural Engine is a batch device and the host round trip dominates.
            #[cfg(feature = "coreml")]
            let mlp = match (rows > 1, self.ane.get(i).and_then(Option::as_ref)) {
                (true, Some(ane)) => {
                    let packed = super::ane::AneMlp::tiles(&n)?;
                    ane.predict(self.backend.stream(), &packed, rows)?
                }
                _ => {
                    let gate = n.matmul(&w.gate)?;
                    let up = n.matmul(&w.up)?;
                    let act = self.swiglu.call(&[gate, up])?.remove(0);
                    act.matmul(&w.down)?
                }
            };
            #[cfg(not(feature = "coreml"))]
            let mlp = {
                let gate = n.matmul(&w.gate)?;
                let up = n.matmul(&w.up)?;
                let act = self.swiglu.call(&[gate, up])?.remove(0);
                act.matmul(&w.down)?
            };
            x = x.add(&mlp)?;
            states.extend([keys, values]);
            if self.variant == Variant::DSpark && [1, 10, 20, 30, 39].contains(&i) {
                taps.push(x.clone());
            }
        }
        let normalized = self.norm(&x, &self.weights.norm)?;
        let head_input = if last_only {
            normalized.slice_axis(1, rows - 1, rows)?
        } else {
            normalized
        };
        let mut out = vec![head_input.matmul(&self.weights.head)?];
        out.extend(states);
        if self.variant == Variant::DSpark {
            out.push(Array::concat(&taps.iter().collect::<Vec<_>>(), 2)?);
        }
        Ok(out)
    }

    /// Runs the reference K=2 wavefront over the whole prepared prompt
    /// (`spec-et.md` §1): per layer the GPU attention of one token chunk
    /// retires under the Neural Engine MLP of the other, and every layer's KV
    /// rows are written first-chunk-then-second-chunk so the cache grows
    /// exactly as a sequential prefill's. Returns the final logits and installs
    /// the caches at `offset = prompt`.
    #[cfg(feature = "coreml")]
    fn prefill_wavefront(&mut self, ids: &[u32], plan: WavefrontPlan) -> Result<Array> {
        if let Some(failure) = &self.failure {
            return Err(Error::Other(format!(
                "MiniCPM state invalid: {failure}; reset and prepare again"
            )));
        }
        // Take the caches like `run_graph` does: a failed wavefront must not
        // leave half-updated state installed.
        let previous = std::mem::take(&mut self.states);
        let result = (|| {
            let (logits, states) = self.wavefront_graph(ids, plan, previous)?;
            let mut output = vec![logits];
            output.extend(states);
            self.backend.stream().eval(&output)?;
            Ok::<_, Error>(self.install_output(output, plan.tokens()))
        })();
        if let Err(error) = &result {
            self.failure = Some(error.to_string());
        }
        result
    }

    /// Builds the wavefront graph. Device work is submitted, not waited for,
    /// except at the two points the schedule cannot avoid: the Neural Engine
    /// input download and the prediction itself. Everything after the
    /// prediction keeps building lazily until the next download.
    #[cfg(feature = "coreml")]
    fn wavefront_graph(
        &self,
        ids: &[u32],
        plan: WavefrontPlan,
        previous: Vec<Array>,
    ) -> Result<(Array, Vec<Array>)> {
        let stream = self.backend.stream();
        let prompt = plan.tokens();
        let layers = self.weights.layers.len();
        if ids.len() != prompt || prompt != plan.half0 + plan.half1 {
            return Err(Error::Contract(
                "MiniCPM5 wavefront covers exactly the prepared prompt",
            ));
        }
        // Per segment: first token position, token count, KV row it writes, and
        // how many key rows its attention reads. The second chunk reads all
        // `prompt` rows, which is what makes its attention equal to the
        // sequential prefill's.
        let start = [0usize, plan.half0];
        let rows = [plan.half0, plan.half1];
        let write = [0usize, plan.half0];
        let read = [plan.half0, prompt];
        let mut residual = Vec::with_capacity(2);
        for segment in 0..2 {
            let end = start[segment] + rows[segment];
            let tokens: Vec<i32> = ids[start[segment]..end].iter().map(|&id| id as i32).collect();
            let tokens = Array::from_i32(stream, &[1, rows[segment]], &tokens)?;
            residual.push(self.weights.embedding.take(&tokens, 0)?);
        }
        let cos = [
            self.cos.slice_axis(2, 0, plan.half0)?,
            self.cos.slice_axis(2, plan.half0, prompt)?,
        ];
        let sin = [
            self.sin.slice_axis(2, 0, plan.half0)?,
            self.sin.slice_axis(2, plan.half0, prompt)?,
        ];
        let mask = [
            segment_mask(stream, 0, plan.half0, plan.half0)?,
            segment_mask(stream, plan.half0, plan.half1, prompt)?,
        ];
        let mut states = previous;
        if states.len() != 2 * layers {
            return Err(Error::Contract(
                "MiniCPM5 wavefront needs one K/V array per layer",
            ));
        }
        // Attention output of each chunk at the layer its MLP is waiting for.
        let mut attention: [Option<Array>; 2] = [None, None];
        let index = |segment: WavefrontSegment| match segment {
            WavefrontSegment::First => 0,
            WavefrontSegment::Second => 1,
        };
        for step in plan.steps(layers) {
            // 1. Read the MLP input before this step submits anything. The
            //    order is load-bearing: the device queue is in-order, so a
            //    download placed after the attention would make the prediction
            //    start after the attention had already finished, and there would
            //    be nothing left to overlap. The input itself depends only on
            //    the previous step, so this wait is short.
            let feed = match step.mlp {
                Some((segment, layer)) if self.ane_program(layer).is_some() => {
                    let h = attention[index(segment)]
                        .as_ref()
                        .ok_or(Error::Contract("MiniCPM5 wavefront step has no MLP input"))?;
                    Some(self.ane_feed(h, layer)?)
                }
                _ => None,
            };
            // 2. Build this step's GPU attention. It stays lazy until later
            //    unless the MLP leg is on the Neural Engine, where it is
            //    submitted so the device retires it under the prediction.
            let submitted = match step.attention {
                Some((segment, layer)) => {
                    let (h, keys, values) = self.wavefront_attention(
                        &residual[index(segment)],
                        layer,
                        &cos[index(segment)],
                        &sin[index(segment)],
                        &mask[index(segment)],
                        &states[2 * layer],
                        &states[2 * layer + 1],
                        write[index(segment)],
                        read[index(segment)],
                        rows[index(segment)],
                    )?;
                    states[2 * layer] = keys;
                    states[2 * layer + 1] = values;
                    attention[index(segment)] = Some(h.clone());
                    Some(h)
                }
                None => None,
            };
            // 3. Issue the MLP leg. An ANE layer pays the blocking prediction
            //    here; a GPU layer just extends the lazy graph, and the next
            //    download or the final evaluation retires it.
            if let Some((segment, layer)) = step.mlp {
                let h = attention[index(segment)]
                    .as_ref()
                    .ok_or(Error::Contract("MiniCPM5 wavefront step has no MLP input"))?;
                residual[index(segment)] = match feed {
                    Some(feed) => {
                        if let Some(pending) = &submitted {
                            stream.async_eval(std::slice::from_ref(pending))?;
                        }
                        let program = self
                            .ane_program(layer)
                            .ok_or(Error::Contract("MiniCPM5 wavefront lost its ANE program"))?;
                        h.add(&program.predict_feed(stream, &feed)?)?
                    }
                    None => self.layer_mlp(h, layer)?,
                };
            }
        }
        // The last token is the last row of the second chunk.
        let normalized = self.norm(&residual[1], &self.weights.norm)?;
        let head_input = normalized.slice_axis(1, plan.half1 - 1, plan.half1)?;
        Ok((head_input.matmul(&self.weights.head)?, states))
    }

    /// One attention sub-layer of one wavefront chunk: residual, norm,
    /// projections, RoPE, the chunk's KV rows and causal attention over
    /// `0..read`. `write` is the chunk's KV row offset, so each layer's rows
    /// land first-chunk-then-second-chunk.
    #[cfg(feature = "coreml")]
    #[allow(clippy::too_many_arguments)]
    fn wavefront_attention(
        &self,
        x: &Array,
        layer: usize,
        cos: &Array,
        sin: &Array,
        mask: &Array,
        keys: &Array,
        values: &Array,
        write: usize,
        read: usize,
        rows: usize,
    ) -> Result<(Array, Array, Array)> {
        let w = &self.weights.layers[layer];
        let n = self.norm(x, &w.input_norm)?;
        let q = n
            .matmul(&w.q)?
            .reshape(&[1, rows, 16, 128])?
            .transpose(&[0, 2, 1, 3])?;
        let k = n
            .matmul(&w.k)?
            .reshape(&[1, rows, 2, 128])?
            .transpose(&[0, 2, 1, 3])?;
        let v = n
            .matmul(&w.v)?
            .reshape(&[1, rows, 2, 128])?
            .transpose(&[0, 2, 1, 3])?;
        let q = self.rotate(&q, cos, sin)?;
        let k = self.rotate(&k, cos, sin)?;
        let index = Array::from_i32(self.backend.stream(), &[1], &[write as i32])?;
        let keys = keys.slice_update(&k, &index, &[2])?;
        let values = values.slice_update(&v, &index, &[2])?;
        let attention = q
            .sdpa(
                &keys.slice_axis(2, 0, read)?,
                &values.slice_axis(2, 0, read)?,
                128f32.powf(-0.5),
                false,
                Some(mask),
            )?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, rows, 2048])?;
        Ok((x.add(&attention.matmul(&w.o)?)?, keys, values))
    }

    /// The whole MLP of one layer on the GPU, as `eager` computes it. Used for
    /// the layers `APXINF_MINICPM5_ANE_MLP` does not select.
    #[cfg(feature = "coreml")]
    fn layer_mlp(&self, h: &Array, layer: usize) -> Result<Array> {
        let w = &self.weights.layers[layer];
        let n = self.norm(h, &w.post_norm)?;
        let gate = n.matmul(&w.gate)?;
        let up = n.matmul(&w.up)?;
        let act = self.swiglu.call(&[gate, up])?.remove(0);
        h.add(&act.matmul(&w.down)?)
    }

    /// The Neural Engine program of one layer, when it runs there.
    #[cfg(feature = "coreml")]
    fn ane_program(&self, layer: usize) -> Option<&super::ane::AneMlp> {
        self.ane.get(layer).and_then(Option::as_ref)
    }

    /// The Neural Engine MLP's host input for one chunk: the chunk's post
    /// attention norm, tiled FP16 channel-first and downloaded. Split from the
    /// prediction so the other chunk's attention can be queued in between.
    #[cfg(feature = "coreml")]
    fn ane_feed(&self, h: &Array, layer: usize) -> Result<super::ane::AneFeed> {
        let w = &self.weights.layers[layer];
        super::ane::AneFeed::of(&self.norm(h, &w.post_norm)?)
    }

    pub(crate) fn run_tokens(
        &mut self,
        ids: &[u32],
        position: usize,
        last_only: bool,
    ) -> Result<Array> {
        Ok(self
            .execute_tokens(ids, position, RequiredOutput::Logits { last_only })?
            .expect("logits requested"))
    }

    fn execute_tokens(
        &mut self,
        ids: &[u32],
        position: usize,
        required: RequiredOutput,
    ) -> Result<Option<Array>> {
        if let Some(failure) = &self.failure {
            return Err(Error::Other(format!(
                "MiniCPM state invalid: {failure}; reset and prepare again"
            )));
        }
        if ids.is_empty()
            || ids.iter().any(|&t| t >= self.config.vocab_size as u32)
            || position != self.offset
            || position
                .checked_add(ids.len())
                .is_none_or(|n| n > self.capacity)
        {
            return Err(Error::Contract(
                "invalid MiniCPM tokens, position, or context",
            ));
        }
        if self.states.is_empty() || self.prepared.is_none() {
            return Err(Error::Contract("prepare MiniCPM before inference"));
        }
        let prefill = self.prepared.as_ref().unwrap().prefill;
        let admitted = if position < prefill.prompt {
            prefill.admits(position, ids.len())
        } else {
            ids.len() == 1 || (self.variant == Variant::DSpark && ids.len() <= 8)
        };
        if !admitted {
            return Err(Error::Contract(
                "MiniCPM input shape is outside its prepared prompt/decode profile",
            ));
        }
        let end = position + ids.len();
        let cache_only = matches!(required, RequiredOutput::CacheOnly);
        if cache_only && (self.variant != Variant::Compiled || end >= prefill.prompt) {
            return Err(Error::Contract(
                "MiniCPM cache-only evaluation requires a compiled non-final prefill chunk",
            ));
        }
        let last_only = match required {
            RequiredOutput::CacheOnly => true,
            RequiredOutput::Logits { last_only } => last_only,
        };
        let result = (|| {
            let tokens = Array::from_i32(
                self.backend.stream(),
                &[1, ids.len()],
                &ids.iter().map(|&v| v as i32).collect::<Vec<_>>(),
            )?;
            let mut output = self.run_graph(tokens, position, last_only)?;
            if cache_only {
                drop(output.remove(0));
            }
            self.backend.stream().eval(&output)?;
            if cache_only {
                self.install_cache_output(output, end);
                Ok(None)
            } else {
                Ok::<_, Error>(Some(self.install_output(output, end)))
            }
        })();
        if let Err(error) = &result {
            self.failure = Some(error.to_string());
        }
        result
    }

    fn run_graph(&mut self, tokens: Array, position: usize, last_only: bool) -> Result<Vec<Array>> {
        let rows = tokens.numel();
        let compiled =
            position > 0 && (rows == 1 || (self.variant == Variant::DSpark && rows <= 8));
        let positions = if self.step.is_some() && compiled {
            let prepared = self
                .prepared
                .as_ref()
                .ok_or(Error::Contract("MiniCPM request was not prepared"))?;
            let width = match prepared.attention {
                DecodeAttention::Exact => position + rows,
                DecodeAttention::Bucketed => {
                    attention_width(self.variant, position + rows, self.capacity)
                }
            };
            Some(
                prepared
                    .positions
                    .get(&width)
                    .ok_or(Error::Contract("MiniCPM attention width was not prepared"))?
                    .clone(),
            )
        } else {
            None
        };
        let previous = std::mem::take(&mut self.states);
        let output = match (&self.step, compiled) {
            (Some(step), true) => step.call(&self.step_inputs(
                tokens,
                position,
                rows,
                &previous,
                positions.as_ref().unwrap(),
            )?)?,
            _ => self.eager(&tokens, position, last_only, &previous)?,
        };
        drop(previous);
        Ok(output)
    }

    fn install_output(&mut self, mut output: Vec<Array>, end: usize) -> Array {
        let logits = output.remove(0);
        self.install_cache_output(output, end);
        logits
    }

    fn install_cache_output(&mut self, mut output: Vec<Array>, end: usize) {
        self.last_taps = if self.variant == Variant::DSpark {
            output.pop()
        } else {
            None
        };
        self.states = output;
        self.offset = end;
    }

    pub(crate) fn trim_to(&mut self, position: usize) -> Result<()> {
        if position > self.offset {
            return Err(Error::Contract("MiniCPM rollback cannot advance state"));
        }
        self.offset = position;
        self.last_taps = None;
        Ok(())
    }
}

impl crate::mlx_decode::DeviceDecode for MiniCpm5 {
    fn stream(&self) -> &apxinf_mlx::Stream {
        self.backend.stream()
    }
    fn decode(&mut self, token: &Array) -> Result<Array> {
        if self.failure.is_some() || self.offset >= self.capacity || self.prepared.is_none() {
            return Err(Error::Contract(
                "MiniCPM decode requires valid prepared capacity",
            ));
        }
        let result = self.run_graph(
            token.cast(MlxDType::I32)?.reshape(&[1, 1])?,
            self.offset,
            false,
        );
        match result {
            Ok(output) => Ok(self.install_output(output, self.offset + 1)),
            Err(error) => {
                self.failure = Some(error.to_string());
                Err(error)
            }
        }
    }
    fn finish_decode(&mut self, outcome: crate::mlx_decode::DecodeFinish) -> Result<()> {
        if let Err(error) = self.backend.stream().synchronize() {
            self.failure = Some(error.to_string());
            return Err(error);
        }
        match outcome {
            crate::mlx_decode::DecodeFinish::PublishedPosition(position)
                if self.failure.is_none()
                    && position <= self.offset
                    && self.offset - position <= 1 =>
            {
                self.trim_to(position)
            }
            crate::mlx_decode::DecodeFinish::Invalidated => {
                self.failure
                    .get_or_insert_with(|| "greedy request failed or was abandoned".into());
                Ok(())
            }
            _ => {
                self.failure = Some("invalid greedy rollback position".into());
                Err(Error::Contract("invalid MiniCPM greedy rollback position"))
            }
        }
    }
}

impl LlmTrait for MiniCpm5 {
    fn token_sequence<'a>(
        &'a mut self,
        first: Tensor,
        sampler: Box<dyn apxinf_core::TokenSampler>,
        params: &apxinf_core::TokenSamplingParams,
        prompt_end: u32,
    ) -> Result<Box<dyn crate::llm_trait::TokenSequence + 'a>> {
        if self.variant == Variant::Compiled && crate::mlx_decode::enabled(params) {
            let first = self.backend.array(&first)?;
            Ok(Box::new(crate::mlx_decode::GreedySequence::new(
                self, first, prompt_end,
            )?))
        } else {
            Ok(Box::new(crate::llm_trait::SyncTokenSequence::new(
                self, first, sampler, prompt_end,
            )))
        }
    }

    fn preparation_status(&self) -> TextPreparationStatus {
        let ready = self.failure.is_none() && self.prepared.is_some() && !self.states.is_empty();
        let mut scopes = Vec::new();
        let mut lengths = Vec::new();
        let max_rows = if self.variant == Variant::DSpark {
            8
        } else {
            1
        };
        if ready {
            scopes.push(TextCompilationScope::LocalSubgraphs);
            if self.step.is_some() {
                scopes.push(TextCompilationScope::DecodeStep);
            }
            lengths.extend(1..=max_rows);
            lengths.extend(self.prepared.as_ref().unwrap().prefill.required_rows());
            // The overlap schedule precomputes the two token chunks' row counts,
            // so their presence here is the observable sign that this request
            // will run the wavefront instead of the serial chunks.
            #[cfg(feature = "coreml")]
            if let Some(wavefront) = self.prepared.as_ref().unwrap().wavefront {
                lengths.extend([wavefront.half0, wavefront.half1]);
            }
            lengths.sort_unstable();
            lengths.dedup();
        }
        TextPreparationStatus {
            state: if self.failure.is_some() {
                TextPreparationState::Invalidated
            } else if ready {
                TextPreparationState::Ready
            } else {
                TextPreparationState::Unprepared
            },
            implementation: "apxinf/minicpm5/mlx-0.32.3",
            variant: Some(match self.variant {
                Variant::Public => "bf16-public",
                Variant::Compiled => "bf16-compiled",
                Variant::DSpark => "dspark-target",
            }),
            compiled_scopes: scopes,
            prepared_prompt_tokens: self
                .prepared
                .as_ref()
                .filter(|_| ready)
                .map(|p| p.prefill.prompt),
            prepared_sequence_lengths: lengths,
            kv_capacity: (self.capacity > 0).then_some(self.capacity),
            max_decode_rows: ready.then_some(max_rows),
            error: self.failure.clone(),
        }
    }

    fn load(_: ModelConfig, _: HashMap<String, Tensor>, _: Device) -> Result<Self> {
        Err(Error::Contract("MiniCPM5 needs checkpoint-specific config; load through AutoModel with model_name minicpm5"))
    }
    fn forward(&mut self, ids: &[u32], start_pos: u32) -> Result<Tensor> {
        let logits = self.run_tokens(ids, start_pos as usize, false)?;
        self.backend
            .from_array(logits.reshape(&[ids.len(), self.vocab_size()])?)
    }
    fn backend(&self) -> &dyn Backend {
        &self.backend
    }
    fn reset(&mut self) {
        self.offset = 0;
        self.last_taps = None;
        self.failure = self
            .backend
            .stream()
            .synchronize()
            .err()
            .map(|e| e.to_string());
    }
    fn release_request_state_retaining_preparation(&mut self) -> Result<()> {
        if self.variant != Variant::Compiled {
            return Err(Error::Contract("prepared-code retention requires ordinary compiled MiniCPM5"));
        }
        if self.failure.is_some() { return self.release_request_state(); }
        self.backend.stream().synchronize()?;
        self.states.clear();
        self.last_taps = None;
        self.offset = 0;
        self.retained_preparation = true;
        Ok(())
    }
    fn release_request_state(&mut self) -> Result<()> {
        self.backend.stream().synchronize()?;
        self.states.clear();
        self.step = None;
        self.prepared = None;
        self.retained_preparation = false;
        self.last_taps = None;
        self.local = None;
        let stream = self.backend.stream();
        self.swiglu = Compiled::with_shapeless(stream, 1, true, |a| Ok(vec![math::swiglu(&a[0], &a[1])?]))?;
        self.local = LocalMath::new(self.variant, &self.epsilon, self.config.rms_norm_eps)?;
        self.capacity = 0;
        self.offset = 0;
        self.failure = None;
        stream.synchronize()
    }
    fn prewarm_decode(&mut self, prompt: usize, output: usize) {
        if let Err(e) = self.prepare(prompt, output) {
            self.failure = Some(e.to_string());
        }
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        self.prefill_controlled(input, &mut |_| Ok(true))?
            .ok_or(Error::Contract("uncontrolled MiniCPM prefill stopped"))
    }
    fn prefill_controlled(
        &mut self,
        input: LlmInput<'_>,
        boundary: &mut dyn FnMut(GenerationBoundary) -> Result<bool>,
    ) -> Result<Option<Tensor>> {
        if input.image.is_some() {
            return Err(Error::Contract("MiniCPM5 is text-only"));
        }
        if self.offset != 0
            || self.prepared.as_ref().map(|p| p.prefill.prompt) != Some(input.token_ids.len())
            || input.token_ids.is_empty()
            || input.token_ids.iter().any(|&id| id as usize >= self.config.vocab_size)
        {
            return Err(Error::Contract("MiniCPM prefill requires its prepared prompt and valid tokens"));
        }
        if !boundary(GenerationBoundary::PrefillStart)? { return Ok(None); }
        let plan = self.prepared.as_ref().unwrap().prefill;
        // A request that admits the wavefront runs the whole prompt in one
        // two-chunk schedule; every other request keeps the serial chunk loop
        // untouched, and a request never mixes the two.
        #[cfg(feature = "coreml")]
        if let Some(wavefront) = self.prepared.as_ref().unwrap().wavefront {
            let logits = self.prefill_wavefront(input.token_ids, wavefront)?;
            if !boundary(GenerationBoundary::PrefillProgress { consumed_tokens: plan.prompt })? {
                return Ok(None);
            }
            return self
                .backend
                .from_array(logits.reshape(&[1, self.vocab_size()])?)
                .map(Some);
        }
        let mut logits = None;
        for range in plan.chunks() {
            let end = range.end;
            let required = if self.variant == Variant::Compiled && end < plan.prompt {
                RequiredOutput::CacheOnly
            } else { RequiredOutput::Logits { last_only: true } };
            logits = self.execute_tokens(&input.token_ids[range], self.offset, required)?;
            if !boundary(GenerationBoundary::PrefillProgress { consumed_tokens: end })? {
                return Ok(None);
            }
        }
        let logits = logits.expect("final chunk produces logits");
        self.backend.from_array(logits.reshape(&[1, self.vocab_size()])?).map(Some)
    }
}
