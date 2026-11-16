//! The qualified MiniCPM5-2B geometry, independent of shared Llama defaults.
use apxinf_core::{Error, Result};
use serde::Deserialize;

pub const CONTEXT_CAPACITY: usize = 65792;
pub const PREFILL_CHUNK_TOKENS: usize = 256;
pub const DSPARK_CONTEXT_CAPACITY: usize = 4096;
#[derive(Clone, Copy)]
pub(super) enum PrefillChunk {
    Tokens256 = 256,
    Tokens512 = 512,
    Tokens1024 = 1024,
    Tokens2048 = 2048,
}

impl PrefillChunk {
    pub(super) fn from_env() -> Result<Option<Self>> {
        match std::env::var("APXINF_MINICPM5_PREFILL_CHUNK") {
            Err(std::env::VarError::NotPresent) => Ok(None),
            Ok(value) => match value.as_str() {
                "256" => Ok(Some(Self::Tokens256)),
                "512" => Ok(Some(Self::Tokens512)),
                "1024" => Ok(Some(Self::Tokens1024)),
                "2048" => Ok(Some(Self::Tokens2048)),
                _ => Err(Error::Contract(
                    "APXINF_MINICPM5_PREFILL_CHUNK must be 512, 1024 or 2048",
                )),
            },
            Err(_) => Err(Error::Contract(
                "APXINF_MINICPM5_PREFILL_CHUNK must be UTF-8",
            )),
        }
    }
}

/// Layers whose MLP runs as one FP16 Core ML program per layer on the Neural
/// Engine instead of the GPU. Empty is the qualified GPU path.
///
/// `APXINF_MINICPM5_ANE_MLP` accepts `N` (first N layers), a list of ranges
/// (`0-7,32`), or `off`/`0`. The set is resolved before any device work so a
/// typo cannot silently select a different schedule; the caller still admits
/// every program numerically before a layer becomes ANE.
#[cfg(feature = "coreml")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AneMlpPlan {
    layers: Vec<usize>,
}

#[cfg(feature = "coreml")]
impl AneMlpPlan {
    pub(super) const ENV: &'static str = "APXINF_MINICPM5_ANE_MLP";
    pub(super) const MAX_LAYERS: usize = 42;

    pub(super) fn from_env() -> Result<Option<Self>> {
        match std::env::var(Self::ENV) {
            Err(std::env::VarError::NotPresent) => Ok(None),
            Ok(value) => Self::parse(value.trim()),
            Err(_) => Err(Error::Contract("APXINF_MINICPM5_ANE_MLP must be UTF-8")),
        }
    }

    pub(super) fn parse(value: &str) -> Result<Option<Self>> {
        if value.is_empty() || value.eq_ignore_ascii_case("off") || value == "0" {
            return Ok(None);
        }
        let mut layers = Vec::new();
        for part in value.split(',') {
            let part = part.trim();
            let (start, end) = match part.split_once('-') {
                Some((start, end)) => (start, end),
                None => (part, part),
            };
            let start: usize = start
                .parse()
                .map_err(|_| Error::Contract("APXINF_MINICPM5_ANE_MLP needs layer indices"))?;
            let end: usize = end
                .parse()
                .map_err(|_| Error::Contract("APXINF_MINICPM5_ANE_MLP needs layer indices"))?;
            if start > end || end >= Self::MAX_LAYERS {
                return Err(Error::Contract(
                    "APXINF_MINICPM5_ANE_MLP layers must be within 0..42 and ordered",
                ));
            }
            layers.extend(start..=end);
        }
        layers.sort_unstable();
        layers.dedup();
        Ok(Some(Self { layers }))
    }

    pub(super) fn layers(&self) -> &[usize] {
        &self.layers
    }
}

/// `APXINF_MINICPM5_ANE_OVERLAP`: run the reference two-chunk wavefront
/// prefill instead of the serial one when `APXINF_MINICPM5_ANE_MLP` selects
/// layers. Only `1` enables it; unset, `0` and every other value keep the
/// qualified serial schedule, like `APXINF_MINICPM5_EXACT_ATTN`.
#[cfg(feature = "coreml")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OverlapKnob {
    On,
    Off,
}

#[cfg(feature = "coreml")]
impl OverlapKnob {
    pub(super) const ENV: &'static str = "APXINF_MINICPM5_ANE_OVERLAP";

    pub(super) fn from_override(value: Option<&std::ffi::OsStr>) -> Self {
        match value {
            Some(value) if value == "1" => Self::On,
            _ => Self::Off,
        }
    }

    pub(super) fn from_env() -> Self {
        Self::from_override(std::env::var_os(Self::ENV).as_deref())
    }

    pub(super) fn is_on(self) -> bool {
        matches!(self, Self::On)
    }
}

/// The token chunk a wavefront step is working on: `First` is `c0` (the first
/// half of the prefill tokens), `Second` is `c1`.
#[cfg(feature = "coreml")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum WavefrontSegment {
    First,
    Second,
}

/// One step of the K=2 wavefront: a GPU attention leg and the MLP leg issued
/// while it retires. Exactly one leg is always the `First` chunk and the other
/// the `Second`, so a step never has an intra-chunk dependency between its two
/// legs.
#[cfg(feature = "coreml")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WavefrontStep {
    /// Attention to build and submit on the GPU for this step, if any.
    pub(super) attention: Option<(WavefrontSegment, usize)>,
    /// MLP to issue (Neural Engine program or GPU) while that attention runs.
    pub(super) mlp: Option<(WavefrontSegment, usize)>,
}

/// Why a request cannot use the two-chunk wavefront and must keep the serial
/// schedule. Recorded so a fallback is explicit rather than a mixed schedule.
#[cfg(feature = "coreml")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SerialReason {
    /// `APXINF_MINICPM5_ANE_OVERLAP` is not `1`.
    KnobOff,
    /// The variant's prefill is not the qualified compiled-chunk one.
    Variant(Variant),
    /// `APXINF_MINICPM5_ANE_MLP` selected no layer.
    NoAneLayers,
    /// The prompt spans more than one prefill chunk.
    MultipleChunks,
    /// Fewer than two prompt tokens: there is no second chunk.
    TooShort,
}

#[cfg(feature = "coreml")]
impl SerialReason {
    pub(super) fn describe(self) -> &'static str {
        match self {
            Self::KnobOff => "overlap knob off",
            Self::Variant(Variant::Public) => "bf16-public variant",
            Self::Variant(Variant::DSpark) => "dspark variant",
            Self::Variant(Variant::Compiled) => "non-compiled variant",
            Self::NoAneLayers => "no Neural Engine MLP layer selected",
            Self::MultipleChunks => "prompt spans more than one prefill chunk",
            Self::TooShort => "prompt is shorter than two chunks",
        }
    }
}

/// The two token chunks of the reference's K=2 wavefront prefill: `c0` is the
/// first half of the prefill tokens, `c1` the second half.
#[cfg(feature = "coreml")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WavefrontPlan {
    pub(super) half0: usize,
    pub(super) half1: usize,
}

#[cfg(feature = "coreml")]
impl WavefrontPlan {
    /// Admits the wavefront for one prepared prefill request, or names the
    /// reason the serial schedule must be kept.
    ///
    /// `chunk` is the request's only prefill chunk; `None` means the prompt is
    /// split across several chunks, where two half-prompt chunks would be
    /// larger than the chunked schedule ever builds and the reference's
    /// two-chunk shape would no longer hold.
    pub(super) fn for_request(
        overlap: bool,
        variant: Variant,
        ane_layers: usize,
        chunk: Option<std::ops::Range<usize>>,
    ) -> std::result::Result<Self, SerialReason> {
        if !overlap {
            return Err(SerialReason::KnobOff);
        }
        if variant != Variant::Compiled {
            return Err(SerialReason::Variant(variant));
        }
        if ane_layers == 0 {
            return Err(SerialReason::NoAneLayers);
        }
        let Some(chunk) = chunk else {
            return Err(SerialReason::MultipleChunks);
        };
        let tokens = chunk.len();
        if tokens < 2 {
            return Err(SerialReason::TooShort);
        }
        Ok(Self {
            half0: tokens / 2,
            half1: tokens - tokens / 2,
        })
    }

    pub(super) fn tokens(self) -> usize {
        self.half0 + self.half1
    }

    /// The reference's layer walk (`wavefront.py:528-604`) over `layers`
    /// layers, as an explicit step list.
    ///
    /// A sequential prefill gives layer `L`'s cache the rows `c0` then `c1`, so
    /// the `First` attention of a layer always precedes its `Second` attention
    /// here. Each step overlaps a `First` leg with a `Second` leg, so the GPU
    /// attention of one chunk retires under the Neural Engine MLP of the other
    /// and both streams always have a ready input.
    ///
    /// `c0`'s MLP of the last layer is not scheduled: no output consumes it,
    /// so issuing it would only add one exposed prediction. That is the
    /// reference's measured 83-predict count (2 per layer for `c1`, 1 fewer for
    /// `c0`), not its nominal 84.
    pub(super) fn steps(self, layers: usize) -> Vec<WavefrontStep> {
        if layers == 0 {
            return Vec::new();
        }
        let mut steps = Vec::with_capacity(2 * layers + 1);
        // Prologue: the first chunk's attention at layer 0, whose output is the
        // first Neural Engine MLP input.
        steps.push(WavefrontStep {
            attention: Some((WavefrontSegment::First, 0)),
            mlp: None,
        });
        for layer in 0..layers {
            // The second chunk's attention at this layer, under the first
            // chunk's MLP of the same layer.
            steps.push(WavefrontStep {
                attention: Some((WavefrontSegment::Second, layer)),
                mlp: (layer + 1 < layers).then_some((WavefrontSegment::First, layer)),
            });
            // The first chunk's attention at the next layer, under the second
            // chunk's MLP of this layer.
            if layer + 1 < layers {
                steps.push(WavefrontStep {
                    attention: Some((WavefrontSegment::First, layer + 1)),
                    mlp: Some((WavefrontSegment::Second, layer)),
                });
            }
        }
        // The last layer's remaining MLP, which has no other chunk's attention
        // left to hide under.
        steps.push(WavefrontStep {
            attention: None,
            mlp: Some((WavefrontSegment::Second, layers - 1)),
        });
        steps
    }
}

#[derive(Clone, Copy)]
pub(super) struct PrefillPlan {
    pub(super) prompt: usize,
    chunk: usize,
}

impl PrefillPlan {
    pub(super) fn new(prompt: usize, chunk: PrefillChunk) -> Result<Self> {
        if prompt == 0 {
            return Err(Error::Contract("MiniCPM prompt must not be empty"));
        }
        Ok(Self {
            prompt,
            chunk: if prompt <= DSPARK_CONTEXT_CAPACITY {
                prompt
            } else {
                chunk as usize
            },
        })
    }

    pub(super) fn chunks(self) -> impl Iterator<Item = std::ops::Range<usize>> {
        (0..self.prompt)
            .step_by(self.chunk)
            .map(move |start| start..(start + self.chunk).min(self.prompt))
    }

    pub(super) fn admits(self, position: usize, rows: usize) -> bool {
        self.chunks()
            .any(|r| r.start == position && r.len() == rows)
    }

    pub(super) fn required_rows(self) -> Vec<usize> {
        let mut rows: Vec<_> = self.chunks().map(|r| r.len()).collect();
        rows.sort_unstable();
        rows.dedup();
        rows
    }
}

pub(super) fn attention_width(variant: Variant, end: usize, capacity: usize) -> usize {
    if variant == Variant::DSpark || capacity <= DSPARK_CONTEXT_CAPACITY {
        capacity
    } else {
        (end.div_ceil(256) * 256).min(capacity)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub hidden_act: String,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub attention_bias: bool,
}

impl Config {
    pub(super) fn request_capacity(
        &self,
        variant: Variant,
        prompt: usize,
        output: usize,
    ) -> Result<usize> {
        let total = prompt
            .checked_add(output)
            .ok_or(Error::Contract("MiniCPM context overflow"))?;
        let limit = self.max_position_embeddings.min(match variant {
            Variant::DSpark => DSPARK_CONTEXT_CAPACITY,
            _ => CONTEXT_CAPACITY,
        });
        if prompt == 0 || total > limit {
            return Err(Error::Other(format!(
                "MiniCPM5 {variant:?} requires a nonempty prompt and prompt + output <= {limit}"
            )));
        }
        Ok((total.div_ceil(256) * 256).min(limit))
    }

    pub fn from_json(raw: &str) -> Result<Self> {
        let config: Self =
            serde_json::from_str(raw).map_err(|e| Error::Other(format!("MiniCPM5 config: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "llama"
            || (
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.vocab_size,
            ) != (2048, 6144, 42, 16, 2, 128, 130560)
            || self.max_position_embeddings < DSPARK_CONTEXT_CAPACITY
            || self.rms_norm_eps != 1e-6
            || self.rope_theta != 5_000_000.0
            || self.tie_word_embeddings
            || self.hidden_act != "silu"
            || self.rope_scaling.is_some()
            || self.attention_bias
        {
            return Err(Error::Other(
                "MiniCPM5 MLX supports the official 2B BF16 geometry, unscaled NeoX RoPE and untied head".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Public,
    Compiled,
    DSpark,
}

impl Variant {
    pub fn parse(name: Option<&str>) -> Result<Self> {
        match name.unwrap_or("bf16-compiled") {
            "bf16-public" => Ok(Self::Public),
            "bf16-compiled" => Ok(Self::Compiled),
            "dspark" => Ok(Self::DSpark),
            name => Err(Error::Other(format!(
                "unsupported MiniCPM5 MLX variant: {name}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ane_mlp_plan_accepts_counts_ranges_and_off() {
        assert_eq!(AneMlpPlan::parse("off").unwrap(), None);
        assert_eq!(AneMlpPlan::parse("0").unwrap(), None);
        assert_eq!(
            AneMlpPlan::parse("0-3,10").unwrap().unwrap().layers(),
            &[0, 1, 2, 3, 10]
        );
        assert_eq!(AneMlpPlan::parse("8").unwrap().unwrap().layers(), &[8]);
        assert_eq!(AneMlpPlan::parse("0-1,1-2").unwrap().unwrap().layers(), &[0, 1, 2]);
        assert!(AneMlpPlan::parse("42").is_err());
        assert!(AneMlpPlan::parse("3-1").is_err());
        assert!(AneMlpPlan::parse("abc").is_err());
    }

    #[test]
    fn prefill_plan_preserves_short_prompts_and_admits_only_complete_chunks() {
        for chunk in [
            PrefillChunk::Tokens512,
            PrefillChunk::Tokens1024,
            PrefillChunk::Tokens2048,
        ] {
            let width = chunk as usize;
            assert!(PrefillPlan::new(0, chunk).is_err());
            for prompt in [1, 4096] {
                let short = PrefillPlan::new(prompt, chunk).unwrap();
                assert_eq!(short.chunks().collect::<Vec<_>>(), vec![0..prompt]);
                assert_eq!(short.required_rows(), vec![prompt]);
                assert!(short.admits(0, prompt));
            }
            let tail = PrefillPlan::new(4097, chunk).unwrap();
            assert_eq!(tail.chunks().last(), Some(4096..4097));
            assert_eq!(tail.required_rows(), vec![1, width]);
            assert!(tail.admits(0, width));
            assert!(tail.admits(4096, 1));
            assert!(!tail.admits(1, width));
            assert!(!tail.admits(4096, width));
            assert!(!tail.admits(4097, 1));
            let full = PrefillPlan::new(65536, chunk).unwrap();
            assert_eq!(full.required_rows(), vec![width]);
            assert_eq!(full.chunks().last(), Some(65536 - width..65536));
        }
    }

    fn config() -> Config {
        Config {
            model_type: "llama".into(),
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 42,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 128,
            vocab_size: 130560,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            tie_word_embeddings: false,
            hidden_act: "silu".into(),
            rope_scaling: None,
            attention_bias: false,
        }
    }

    #[test]
    fn refuses_nearby_llama_shapes_and_unsupported_attention_semantics() {
        assert!(config().validate().is_ok());
        let mut c = config();
        c.num_hidden_layers = 32;
        assert!(c.validate().is_err());
        let mut c = config();
        c.rope_scaling = Some(serde_json::json!({"factor": 2}));
        assert!(c.validate().is_err());
        let mut c = config();
        c.tie_word_embeddings = true;
        assert!(c.validate().is_err());
        assert_eq!(Variant::parse(None).unwrap(), Variant::Compiled);
        assert!(Variant::parse(Some("mixed-w8")).is_err());
    }

    #[test]
    fn ordinary_capacity_grows_without_widening_dspark() {
        let c = config();
        assert_eq!(c.request_capacity(Variant::Compiled, 256, 1).unwrap(), 512);
        assert_eq!(
            c.request_capacity(Variant::Compiled, 65408, 128).unwrap(),
            65536
        );
        assert_eq!(
            c.request_capacity(Variant::Compiled, 65536, 128).unwrap(),
            65792
        );
        assert_eq!(
            c.request_capacity(Variant::Compiled, 65536, 256).unwrap(),
            65792
        );
        assert!(c.request_capacity(Variant::Compiled, 65536, 257).is_err());
        assert!(c.request_capacity(Variant::Public, 0, 1).is_err());
        assert!(c.request_capacity(Variant::Public, usize::MAX, 1).is_err());
        assert_eq!(
            c.request_capacity(Variant::DSpark, 3840, 256).unwrap(),
            4096
        );
        assert!(c.request_capacity(Variant::DSpark, 3841, 256).is_err());
        let mut c = config();
        c.max_position_embeddings = 5001;
        assert_eq!(
            c.request_capacity(Variant::Compiled, 5000, 1).unwrap(),
            5001
        );
        assert!(c.request_capacity(Variant::Compiled, 5001, 1).is_err());
        assert_eq!(attention_width(Variant::Compiled, 1, 4096), 4096);
        assert_eq!(attention_width(Variant::Compiled, 256, 65536), 256);
        assert_eq!(attention_width(Variant::Compiled, 257, 65536), 512);
        assert_eq!(attention_width(Variant::Compiled, 65536, 65536), 65536);
        assert_eq!(attention_width(Variant::Compiled, 65536, 65792), 65536);
        assert_eq!(attention_width(Variant::Compiled, 65537, 65792), 65792);
        assert_eq!(attention_width(Variant::DSpark, 1, 4096), 4096);
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn overlap_knob_is_strictly_one() {
        use std::ffi::OsStr;
        let on = |value: Option<&OsStr>| OverlapKnob::from_override(value).is_on();
        assert!(on(Some(OsStr::new("1"))));
        assert!(!on(None));
        for value in ["0", "", "true", "on", "01", "2", " 1"] {
            assert!(!on(Some(OsStr::new(value))), "{value:?} must not enable overlap");
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!on(Some(OsStr::from_bytes(&[0xff]))));
        }
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn wavefront_admits_only_single_compiled_chunk_with_ane_layers() {
        use SerialReason::{MultipleChunks, NoAneLayers, TooShort};
        let split = |overlap, variant, ane_layers, chunk| {
            WavefrontPlan::for_request(overlap, variant, ane_layers, chunk)
        };
        let prompt = || Some(0..2048usize);
        assert_eq!(
            split(false, Variant::Compiled, 4, prompt()),
            Err(SerialReason::KnobOff)
        );
        assert_eq!(
            split(true, Variant::DSpark, 4, prompt()),
            Err(SerialReason::Variant(Variant::DSpark))
        );
        assert_eq!(
            split(true, Variant::Public, 4, prompt()),
            Err(SerialReason::Variant(Variant::Public))
        );
        assert_eq!(split(true, Variant::Compiled, 0, prompt()), Err(NoAneLayers));
        assert_eq!(
            split(true, Variant::Compiled, 4, None),
            Err(MultipleChunks)
        );
        for tokens in [0usize, 1] {
            assert_eq!(
                split(true, Variant::Compiled, 4, Some(0..tokens)),
                Err(TooShort)
            );
        }
        // The measured 2048-token prompt: the reference's own 1024/1024 split.
        assert_eq!(
            split(true, Variant::Compiled, 4, prompt()),
            Ok(WavefrontPlan {
                half0: 1024,
                half1: 1024
            })
        );
        // Every admitted split covers the chunk exactly with two nonempty
        // halves, so no token is dropped or prefilled twice.
        for tokens in 2..=4096usize {
            let plan = split(true, Variant::Compiled, 4, Some(0..tokens)).unwrap();
            assert!(plan.half0 >= 1 && plan.half1 >= 1, "{tokens} tokens");
            assert_eq!(plan.tokens(), tokens);
            assert!(plan.half1 >= plan.half0);
            assert!(plan.half1 - plan.half0 <= 1);
        }
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn wavefront_schedule_mirrors_the_reference_layer_walk() {
        use WavefrontSegment::{First, Second};
        let plan = WavefrontPlan {
            half0: 1024,
            half1: 1024,
        };
        // spec-et.md §1: prologue `c0` attention; per layer `c1` attention
        // under `c0`'s MLP, then the next layer's `c0` attention under `c1`'s
        // MLP; the last layer keeps one exposed `c1` MLP.
        let step = |attention, mlp| WavefrontStep { attention, mlp };
        assert_eq!(
            plan.steps(3),
            vec![
                step(Some((First, 0)), None),
                step(Some((Second, 0)), Some((First, 0))),
                step(Some((First, 1)), Some((Second, 0))),
                step(Some((Second, 1)), Some((First, 1))),
                step(Some((First, 2)), Some((Second, 1))),
                step(Some((Second, 2)), None),
                step(None, Some((Second, 2))),
            ]
        );
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn wavefront_schedule_keeps_every_layer_cache_sequential() {
        use WavefrontSegment::{First, Second};
        let plan = WavefrontPlan {
            half0: 1024,
            half1: 1024,
        };
        let layers = 42;
        let steps = plan.steps(layers);
        assert_eq!(steps.len(), 2 * layers + 1);
        for layer in 0..layers {
            let first = steps
                .iter()
                .position(|s| s.attention == Some((First, layer)))
                .unwrap_or_else(|| panic!("no first-chunk attention at layer {layer}"));
            let second = steps
                .iter()
                .position(|s| s.attention == Some((Second, layer)))
                .unwrap_or_else(|| panic!("no second-chunk attention at layer {layer}"));
            assert!(
                first < second,
                "layer {layer} must write its first-chunk KV rows before its second-chunk rows"
            );
            // Both attention legs must be issued before either chunk's MLP of
            // the next layer can run, and each layer has exactly one of each.
            assert_eq!(
                steps
                    .iter()
                    .filter(|s| s.attention == Some((First, layer)))
                    .count(),
                1
            );
            assert_eq!(
                steps
                    .iter()
                    .filter(|s| s.attention == Some((Second, layer)))
                    .count(),
                1
            );
        }
        // The MLP of a chunk at layer L is issued after that chunk's own
        // attention at L, so its input always exists.
        let mut issued = Vec::new();
        for (index, step) in steps.iter().enumerate() {
            if let Some((segment, layer)) = step.mlp {
                issued.push((segment, layer));
                let attention = steps[..index]
                    .iter()
                    .position(|s| s.attention == Some((segment, layer)))
                    .unwrap_or_else(|| {
                        panic!("{segment:?} MLP at layer {layer} runs before its attention")
                    });
                assert!(attention < index);
            }
        }
        issued.sort();
        assert_eq!(issued.len(), 2 * layers - 1);
        let mut expected: Vec<(WavefrontSegment, usize)> = (0..layers)
            .map(|layer| (Second, layer))
            .chain((0..layers - 1).map(|layer| (First, layer)))
            .collect();
        expected.sort();
        assert_eq!(issued, expected);
        // Every step except the last issues its MLP under the other chunk's
        // attention, which is the overlap the reference claims.
        for step in &steps {
            if let (Some((attention, _)), Some((mlp, _))) = (step.attention, step.mlp) {
                assert_ne!(attention, mlp, "a step cannot overlap a chunk with itself");
            }
        }
        // The prediction count: two legs per layer minus the dead last-layer
        // first-chunk MLP.
        let predicts = steps.iter().filter(|s| s.mlp.is_some()).count();
        assert_eq!(predicts, 2 * layers - 1);
        assert_eq!(plan.steps(0), Vec::new());
    }
}
