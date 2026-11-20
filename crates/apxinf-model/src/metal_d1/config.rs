//! LiquidAI d1-2B checkpoint configuration: nested `text_config` (the LFM3
//! text backbone), `vision_config` (the LFM3-VL tower), the projector and the
//! D1 readout record.
//!
//! Provenance: the `TextConfig` geometry/validation is copied from
//! `metal_lfm3` (`origin/feat/lfm3-metal` @ 1ed3f2f3, verified against
//! LiquidAI/LFM3-1.7B); d1-2B's `text_config` is field-for-field identical.
//! Everything the forward paths need is derived and validated here, once, so
//! no execution path guesses a default. Unsupported checkpoint combinations
//! are rejected at load, not silently degraded.

use apxinf_core::{Error, Result};
use serde::Deserialize;

/// Largest sequence length the P1a text path admits for one packed request.
///
/// The reference engine batches under a 65536-token budget; this port starts
/// with one sequence per forward and admits the same per-sequence envelope.
/// The full-attention mask is `[T, T]` booleans, so the admitted envelope is
/// smaller than the budget (P3 may revisit block-decomposed masks).
pub const MAX_SEQUENCE: usize = 8192;

/// Operator type of one decoder layer, from `text_config.layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Width-`conv_L_cache` gated depthwise short convolution over 2048 channels.
    Conv,
    /// Bidirectional (d1 set-attention) attention, RoPE theta 1e6 partial 0.5.
    FullAttention,
    /// Causal attention restricted to a 256-token window, RoPE theta 1e4.
    SlidingAttention,
}

impl LayerKind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "conv" => Ok(Self::Conv),
            "full_attention" => Ok(Self::FullAttention),
            "sliding_attention" => Ok(Self::SlidingAttention),
            other => Err(Error::Other(format!(
                "d1 layer type `{other}` is not implemented (conv, full_attention, sliding_attention only)"
            ))),
        }
    }

    pub fn is_conv(self) -> bool {
        matches!(self, Self::Conv)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Conv => "conv",
            Self::FullAttention => "full_attention",
            Self::SlidingAttention => "sliding_attention",
        }
    }
}

/// One `rope_parameters.<layer_type>` entry.
#[derive(Clone, Debug, Deserialize)]
struct RopeEntry {
    rope_theta: f32,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
    #[serde(default)]
    rope_type: Option<String>,
}

/// `text_config.rope_parameters`, one independent set per attention type.
#[derive(Clone, Debug, Deserialize)]
struct RopeParameters {
    full_attention: Option<RopeEntry>,
    sliding_attention: Option<RopeEntry>,
}

/// `text_config`: the LFM3 text backbone (identical geometry to LFM3-1.7B).
#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub max_position_embeddings: usize,
    pub norm_eps: f32,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub use_sandwich_norm: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    rope_parameters: Option<RopeParameters>,
}

/// One oriented RoPE program: precomputed cos/sin coverage for every position.
#[derive(Clone, Copy, Debug)]
pub struct RopeScheme {
    pub theta: f32,
    /// Rotated prefix width: `(partial_rotary_factor * head_dim) as usize / 2 * 2`.
    pub rotary_dim: usize,
    /// Number of angle frequencies that are actually non-zero (`rotary_dim / 2`).
    pub angles: usize,
}

impl RopeScheme {
    /// Inverse frequencies with the zero tail the reference implements: index
    /// pair `i` uses `theta^(-2i/head_dim)`; dimensions `>= rotary_dim` carry
    /// `cos=1, sin=0` and are therefore identity rotations.
    pub fn inv_freq(&self, head_dim: usize) -> Vec<f32> {
        let mut inv = Vec::with_capacity(head_dim / 2);
        for i in 0..self.angles {
            let exponent = (2 * i) as f64 / head_dim as f64;
            inv.push((1.0 / (self.theta as f64).powf(exponent)) as f32);
        }
        inv.resize(head_dim / 2, 0.0);
        inv
    }
}

impl TextConfig {
    /// Reject anything this port does not implement instead of guessing.
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "lfm3" {
            return Err(Error::Other(format!(
                "metal_d1 requires text_config.model_type `lfm3`, got `{}`",
                self.model_type
            )));
        }
        let dimensions = [
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.sliding_window,
            self.conv_l_cache,
        ];
        let head_width = self
            .num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::Other("d1 head geometry overflows".into()))?;
        if dimensions.iter().any(|&n| n == 0 || n > i32::MAX as usize)
            || self.head_dim % 2 != 0
            || self.num_attention_heads % self.num_key_value_heads != 0
            || head_width != self.hidden_size
            || self.conv_l_cache < 2
            || self.layer_types.len() != self.num_hidden_layers
        {
            return Err(Error::Other(
                "invalid d1 text dimensions, head grouping or layer plan".into(),
            ));
        }
        if self.conv_bias || !self.use_sandwich_norm || !self.tie_word_embeddings {
            return Err(Error::Other(
                "metal_d1 implements the bias-free conv, sandwich-norm, tied-embedding LFM3 backbone only"
                    .into(),
            ));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(Error::Other(
                "d1 norm epsilon must be positive and finite".into(),
            ));
        }
        for spec in &self.layer_types {
            LayerKind::parse(spec)?;
        }
        self.rope(LayerKind::FullAttention)?;
        self.rope(LayerKind::SlidingAttention)?;
        Ok(())
    }

    /// The pinned d1-2B text geometry (same tuple as LFM3-1.7B).
    pub fn validate_checkpoint_scope(&self) -> Result<()> {
        let conv = self
            .layer_types
            .iter()
            .filter(|spec| spec.as_str() == "conv")
            .count();
        let full = self
            .layer_types
            .iter()
            .filter(|spec| spec.as_str() == "full_attention")
            .count();
        let sliding = self
            .layer_types
            .iter()
            .filter(|spec| spec.as_str() == "sliding_attention")
            .count();
        if (
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.sliding_window,
            self.conv_l_cache,
        ) != (2048, 7680, 24, 16, 4, 128, 128000, 256, 7)
            || (conv, full, sliding) != (13, 6, 5)
            || self.norm_eps != 1e-5
        {
            return Err(Error::Other(
                "metal_d1 supports the d1-2B text geometry only: hidden 2048, intermediate 7680, 24 layers (13 conv / 6 full / 5 sliding), 16/4 heads of 128, vocab 128000, window 256, conv width 7, epsilon 1e-5"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn layer_kinds(&self) -> Result<Vec<LayerKind>> {
        self.layer_types
            .iter()
            .map(|spec| LayerKind::parse(spec))
            .collect()
    }

    pub fn layer_kind(&self, layer: usize) -> Result<LayerKind> {
        let spec = self
            .layer_types
            .get(layer)
            .ok_or_else(|| Error::Other(format!("d1 layer {layer} is outside the plan")))?;
        LayerKind::parse(spec)
    }

    /// Rotary program for one attention type; conv layers have none.
    pub fn rope(&self, kind: LayerKind) -> Result<RopeScheme> {
        let params = self
            .rope_parameters
            .as_ref()
            .ok_or_else(|| Error::Other("d1 text_config is missing rope_parameters".into()))?;
        let (entry, label) = match kind {
            LayerKind::FullAttention => (params.full_attention.as_ref(), "full_attention"),
            LayerKind::SlidingAttention => (params.sliding_attention.as_ref(), "sliding_attention"),
            LayerKind::Conv => {
                return Err(Error::Other(
                    "d1 conv layers have no rotary embedding".into(),
                ))
            }
        };
        let entry = entry.ok_or_else(|| {
            Error::Other(format!("d1 text_config is missing rope_parameters.{label}"))
        })?;
        if entry
            .rope_type
            .as_deref()
            .is_some_and(|kind| kind != "default")
        {
            return Err(Error::Other(format!(
                "d1 rope_parameters.{label}.rope_type must be `default`, got `{:?}`",
                entry.rope_type
            )));
        }
        if !entry.rope_theta.is_finite() || entry.rope_theta <= 0.0 {
            return Err(Error::Other(format!(
                "d1 rope_parameters.{label}.rope_theta must be positive"
            )));
        }
        let partial = entry.partial_rotary_factor.unwrap_or(1.0);
        if !partial.is_finite() || partial <= 0.0 || partial > 1.0 {
            return Err(Error::Other(format!(
                "d1 rope_parameters.{label}.partial_rotary_factor must be in (0, 1]"
            )));
        }
        // The reference derives the rotated prefix in whole four-angle groups:
        // `int(partial * head_dim) // 2`. Keep that exact truncation.
        let rotary_dim = ((partial as f64 * self.head_dim as f64) as usize) / 2 * 2;
        if rotary_dim == 0 || rotary_dim > self.head_dim {
            return Err(Error::Other(format!(
                "d1 rope_parameters.{label} yields an empty or oversized rotary prefix"
            )));
        }
        Ok(RopeScheme {
            theta: entry.rope_theta,
            rotary_dim,
            angles: rotary_dim / 2,
        })
    }

    /// Tokens kept between steps by the short convolution (`conv_L_cache - 1`).
    pub fn conv_state_len(&self) -> usize {
        self.conv_l_cache - 1
    }

    /// Number of positions the sliding layers may attend (checkpoint window).
    pub fn window(&self) -> usize {
        self.sliding_window
    }
}

/// `vision_config` (lfm3_vl_vision). Parsed and validated at load; the tower
/// is executed in P1b, not by the P1a text path.
#[derive(Clone, Debug, Deserialize)]
pub struct VisionConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub conv_kernel_size: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub norm_eps: f32,
    pub layer_types: Vec<String>,
    rope_parameters: Option<VisionRope>,
}

#[derive(Clone, Debug, Deserialize)]
struct VisionRope {
    rope_theta: f32,
    rope_type: String,
}

impl VisionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "lfm3_vl_vision" {
            return Err(Error::Other(format!(
                "metal_d1 requires vision_config.model_type `lfm3_vl_vision`, got `{}`",
                self.model_type
            )));
        }
        let rope = self
            .rope_parameters
            .as_ref()
            .ok_or_else(|| Error::Other("d1 vision_config is missing rope_parameters".into()))?;
        if rope.rope_type != "axial" || !rope.rope_theta.is_finite() || rope.rope_theta <= 0.0 {
            return Err(Error::Other(
                "metal_d1 implements the axial-RoPE vision tower only".into(),
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers
            || self
                .layer_types
                .iter()
                .any(|t| t != "conv" && t != "full_attention")
        {
            return Err(Error::Other(
                "d1 vision tower accepts conv and full_attention layers only".into(),
            ));
        }
        Ok(())
    }
}

/// The whole `config.json` of a d1 checkpoint.
#[derive(Clone, Debug, Deserialize)]
pub struct D1Config {
    pub model_type: String,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub downsample_factor: usize,
    pub projector_hidden_size: usize,
    pub projector_hidden_act: String,
    #[serde(default)]
    pub projector_bias: bool,
    pub projector_layer_norm_eps: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub bos_token_id: u32,
    pub pad_token_id: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    #[serde(default)]
    pub d1_codes: Option<u32>,
    #[serde(default)]
    pub d1_numbers: bool,
    #[serde(default)]
    pub d1_attention: Option<String>,
    #[serde(default)]
    pub d1_set_size: Option<usize>,
    #[serde(default)]
    pub d1_vision_max_pixels: Option<u64>,
    #[serde(default)]
    pub d1_box_edges: Option<String>,
    #[serde(default)]
    pub d1_prompt: Option<u32>,
}

impl D1Config {
    pub fn from_json(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)
            .map_err(|e| Error::Other(format!("invalid d1 config: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "lfm3_vl" {
            return Err(Error::Other(format!(
                "metal_d1 requires model_type `lfm3_vl`, got `{}`",
                self.model_type
            )));
        }
        self.text_config.validate()?;
        self.text_config.validate_checkpoint_scope()?;
        self.vision_config.validate()?;
        if !self.tie_word_embeddings || !self.text_config.tie_word_embeddings {
            return Err(Error::Other("metal_d1 implements tied embeddings only".into()));
        }
        if self.d1_attention.as_deref() != Some("set") {
            return Err(Error::Other(format!(
                "metal_d1 implements d1_attention `set` only, got {:?}",
                self.d1_attention
            )));
        }
        if self.d1_codes != Some(2) {
            return Err(Error::Other(format!(
                "metal_d1 implements d1_codes 2 only, got {:?}",
                self.d1_codes
            )));
        }
        if !self.d1_numbers {
            return Err(Error::Other(
                "metal_d1 implements the number-reading (d1_numbers) checkpoint only".into(),
            ));
        }
        let set_size = self
            .d1_set_size
            .ok_or_else(|| Error::Other("d1 config is missing d1_set_size".into()))?;
        if set_size == 0 || set_size > 128 {
            return Err(Error::Other(format!(
                "d1_set_size must be in 1..=128, got {set_size}"
            )));
        }
        if let Some(edges) = &self.d1_box_edges {
            if edges != "median" && edges != "interp" {
                return Err(Error::Other(format!(
                    "d1_box_edges must be `median` or `interp`, got {edges:?}"
                )));
            }
        }
        if let Some(prompt) = self.d1_prompt {
            if prompt != 1 && prompt != 2 {
                return Err(Error::Other(format!(
                    "d1_prompt must be 1 or 2, got {prompt}"
                )));
            }
        }
        if self.projector_hidden_act != "gelu" || !self.projector_bias {
            return Err(Error::Other(
                "metal_d1 implements the gelu projector with bias only".into(),
            ));
        }
        if !(self.projector_layer_norm_eps.is_finite() && self.projector_layer_norm_eps > 0.0) {
            return Err(Error::Other("d1 projector norm eps must be positive".into()));
        }
        Ok(())
    }

    pub fn set_size(&self) -> usize {
        self.d1_set_size.unwrap_or(4)
    }

    pub fn prompt_version(&self) -> u32 {
        self.d1_prompt.unwrap_or(1)
    }

    pub fn box_edges(&self) -> &str {
        self.d1_box_edges.as_deref().unwrap_or("median")
    }

    pub fn vision_max_pixels(&self) -> u64 {
        self.d1_vision_max_pixels.unwrap_or(1024 * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An inline copy of the pinned d1-2B config's relevant fields.
    fn checkpoint_json() -> String {
        r##"{
            "model_type": "lfm3_vl",
            "bos_token_id": 124894,
            "downsample_factor": 2,
            "eos_token_id": 124900,
            "image_token_id": 125030,
            "pad_token_id": 124893,
            "projector_bias": true,
            "projector_hidden_act": "gelu",
            "projector_hidden_size": 4096,
            "projector_layer_norm_eps": 1e-06,
            "tie_word_embeddings": true,
            "video_token_id": 125029,
            "vision_end_token_id": 125010,
            "vision_start_token_id": 125009,
            "d1_codes": 2,
            "d1_attention": "set",
            "d1_numbers": true,
            "d1_set_size": 16,
            "d1_vision_max_pixels": 1638400,
            "text_config": {
                "model_type": "lfm3",
                "hidden_size": 2048,
                "intermediate_size": 7680,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 128000,
                "max_position_embeddings": 128000,
                "norm_eps": 1e-5,
                "conv_L_cache": 7,
                "conv_bias": false,
                "sliding_window": 256,
                "layer_types": ["conv","conv","full_attention","conv","sliding_attention",
                                "conv","full_attention","conv","sliding_attention","conv",
                                "full_attention","conv","sliding_attention","conv",
                                "full_attention","conv","sliding_attention","conv",
                                "full_attention","conv","sliding_attention","conv",
                                "full_attention","conv"],
                "use_sandwich_norm": true,
                "tie_word_embeddings": true,
                "bos_token_id": 124894,
                "pad_token_id": 124893,
                "rope_parameters": {
                    "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"},
                    "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}
                }
            },
            "vision_config": {
                "model_type": "lfm3_vl_vision",
                "hidden_size": 1024,
                "intermediate_size": 2560,
                "num_hidden_layers": 14,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "conv_kernel_size": 3,
                "in_channels": 3,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "norm_eps": 1e-5,
                "layer_types": ["conv","conv","full_attention","conv","full_attention","conv",
                                "full_attention","conv","full_attention","conv","full_attention",
                                "conv","full_attention","conv"],
                "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "axial"}
            }
        }"##
        .to_string()
    }

    #[test]
    fn parses_the_pinned_d1_checkpoint() {
        let config = D1Config::from_json(&checkpoint_json()).unwrap();
        assert_eq!(config.set_size(), 16);
        assert_eq!(config.prompt_version(), 1);
        assert_eq!(config.box_edges(), "median");
        assert_eq!(config.vision_max_pixels(), 1638400);
        assert_eq!(config.text_config.conv_state_len(), 6);
        assert_eq!(config.text_config.layer_kind(2).unwrap(), LayerKind::FullAttention);
        let full = config.text_config.rope(LayerKind::FullAttention).unwrap();
        assert_eq!((full.theta, full.rotary_dim, full.angles), (1_000_000.0, 64, 32));
        let sliding = config.text_config.rope(LayerKind::SlidingAttention).unwrap();
        assert_eq!((sliding.theta, sliding.rotary_dim, sliding.angles), (10_000.0, 128, 64));
    }

    #[test]
    fn rejects_unsupported_records() {
        for (key, value) in [
            ("d1_attention", serde_json::json!("causal")),
            ("d1_codes", serde_json::json!(1)),
            ("d1_numbers", serde_json::json!(false)),
            ("d1_set_size", serde_json::json!(0)),
            ("model_type", serde_json::json!("lfm3")),
            ("projector_hidden_act", serde_json::json!("silu")),
        ] {
            let mut config: serde_json::Value = serde_json::from_str(&checkpoint_json()).unwrap();
            config[key] = value;
            assert!(D1Config::from_json(&config.to_string()).is_err(), "accepted {key}");
        }
        let mut config: serde_json::Value = serde_json::from_str(&checkpoint_json()).unwrap();
        config["text_config"]["use_sandwich_norm"] = serde_json::json!(false);
        assert!(D1Config::from_json(&config.to_string()).is_err());
        let mut config: serde_json::Value = serde_json::from_str(&checkpoint_json()).unwrap();
        config["vision_config"]["rope_parameters"]["rope_type"] = serde_json::json!("default");
        assert!(D1Config::from_json(&config.to_string()).is_err());
    }
}
