//! Unified model loading for text, vision-language, and VLA models.

use std::path::{Path, PathBuf};

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::accelerator::create_backend;
use crate::builtin::register_builtin_models;
use crate::llm_trait::{LlmCapabilities, LlmInput, LlmTrait};
#[cfg(feature = "model-pi05")]
use crate::pi05::Pi05Config;
use crate::profiling::GenerationProfile;
use crate::registry;
use crate::vla::{Action, InferenceSpec, Observation, PreparedInference, VlaRuntime};

/// User-level precision policy. Hardware/tactic dispatch remains in kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelPrecision {
    #[default]
    Auto,
    Bf16,
    Fp8,
    W8A8,
}

/// Request checkpoint-free random weights for a benchmark. Latency depends only
/// on shape and dtype, so no trained weights are needed to measure the engine.
#[derive(Clone, Copy, Debug)]
pub struct SyntheticWeights {
    pub seed: u64,
}

/// Optional loading policy for models that have calibrated or tuned variants.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// Registry name override. `None` detects the model from
    /// `config.json:model_type`.
    pub model_name: Option<String>,
    pub precision: ModelPrecision,
    /// Optional text-model weight dtype. `None` preserves checkpoint dtype
    /// (except CPU backends, which currently require f32).
    pub text_weight_dtype: Option<DType>,
    /// Maximum context allocated by text-model caches. This is deliberately
    /// independent of the checkpoint's advertised maximum: modern models can
    /// declare hundreds of thousands of tokens, which is not a safe default
    /// on a memory-constrained local machine.
    pub max_context: Option<usize>,
    pub calibration_path: Option<PathBuf>,
    pub tuning_path: Option<PathBuf>,
    /// Explicit architecture config, overriding any on-disk `config.json`.
    #[cfg(feature = "model-pi05")]
    pub config: Option<Pi05Config>,
    /// When set, load deterministic random weights instead of a checkpoint.
    pub synthetic: Option<SyntheticWeights>,
    /// Uniform FP8 activation scale, replacing a calibration file (synthetic use).
    pub uniform_fp8_scale: Option<f32>,
    /// Explicitly replace only Qwen3.5's decode-time tied output projection
    /// and argmax with the feature-gated Metal W8 path. Prompt prefill and the
    /// model body remain on CPU/Accelerate. There is no implicit fallback.
    pub metal_w8_lm_head: bool,
    /// Explicitly replace every Qwen3.5 decode-time MLP with the complete
    /// feature-gated Metal W8 block. Prefill, attention, residuals, and state
    /// remain CPU/F32. This can be combined with `metal_w8_lm_head`.
    pub metal_w8_mlp_block: bool,
}

/// A loaded autoregressive language model (text-only or VLM), or a VLA model.
/// Language generation and observation-to-action inference intentionally use
/// separate traits.
pub enum LoadedModel {
    Text(Box<dyn LlmTrait>),
    Vla(Box<dyn VlaRuntime>),
}

impl LoadedModel {
    pub fn text_mut(&mut self) -> Result<&mut dyn LlmTrait> {
        match self {
            Self::Text(model) => Ok(&mut **model),
            Self::Vla(_) => Err(Error::Other("loaded model is VLA, not text".into())),
        }
    }

    pub fn vla(&self) -> Result<&dyn VlaRuntime> {
        match self {
            Self::Vla(model) => Ok(&**model),
            Self::Text(_) => Err(Error::Other("loaded model is text, not VLA".into())),
        }
    }

    pub fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        self.text_mut()?.forward(token_ids, start_pos)
    }

    pub fn text_capabilities(&self) -> Result<LlmCapabilities> {
        match self {
            Self::Text(model) => Ok(model.capabilities()),
            Self::Vla(_) => Err(Error::Other("loaded model is VLA, not text".into())),
        }
    }

    pub fn generation_path_receipt(&self) -> Result<Option<serde_json::Value>> {
        match self {
            Self::Text(model) => Ok(model.generation_path_receipt()),
            Self::Vla(_) => Err(Error::Other("loaded model is VLA, not text".into())),
        }
    }

    /// Generate from the same request shape for text-only and VLM models.
    pub fn generate_streaming(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        mut on_token: impl FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, GenerationProfile)> {
        self.text_mut()?
            .generate_streaming_dyn(input, max_new_tokens, &mut on_token, eos_token_id)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.text_mut()?.reset();
        Ok(())
    }

    pub fn infer(&self, observation: &Observation) -> Result<Action> {
        self.vla()?.infer(observation)
    }

    /// Run VLA inference and copy the action to host as `f32`. Convenience for
    /// callers that need host values without holding a backend handle.
    pub fn infer_host_f32(&self, observation: &Observation) -> Result<Vec<f32>> {
        self.vla()?.infer_host_f32(observation)
    }

    pub fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.vla()?.prepare(spec)
    }
}

/// Stateless unified frontend. It creates one shared backend, loads weights,
/// and dispatches by model name plus device-specific registry suffix.
pub struct AutoModel;

impl AutoModel {
    /// Read a Hugging Face `model_type` and return its registry name.
    pub fn detect_model_name(path: impl AsRef<Path>) -> Result<String> {
        let path = path.as_ref();
        let model_dir = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or_else(|| Path::new("."))
        };
        let config_path = model_dir.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .map_err(|error| Error::Other(format!("read {}: {error}", config_path.display())))?;
        let config: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|error| Error::Other(format!("parse {}: {error}", config_path.display())))?;
        let model_type = config
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                Error::Other(format!(
                    "{} does not contain a string model_type",
                    config_path.display()
                ))
            })?;
        Ok(model_type.to_owned())
    }

    /// Load any supported model through one entry point.
    ///
    /// By default, the registry name is detected from
    /// `config.json:model_type`. Set [`LoadOptions::model_name`] only when a
    /// checkpoint needs an explicit registry-name override.
    pub fn load_model(
        device: Device,
        path: impl AsRef<Path>,
        options: &LoadOptions,
    ) -> Result<LoadedModel> {
        if options.max_context == Some(0) {
            return Err(Error::Other("max_context must be greater than zero".into()));
        }
        let path = path.as_ref();
        let detected_model_name;
        let model_name = match options.model_name.as_deref() {
            Some(model_name) => model_name,
            None => {
                detected_model_name = Self::detect_model_name(path)?;
                &detected_model_name
            }
        };

        if options.metal_w8_lm_head || options.metal_w8_mlp_block {
            let requested_path = match (options.metal_w8_mlp_block, options.metal_w8_lm_head) {
                (true, true) => "Metal W8 MLP block + lm_head",
                (true, false) => "Metal W8 MLP block",
                (false, true) => "Metal W8 lm_head",
                (false, false) => unreachable!(),
            };
            if !cfg!(feature = "metal-w8-head-mlp") {
                return Err(Error::Other(format!(
                    "{requested_path} was requested, but this binary was not built with the `metal-w8` feature"
                )));
            }
            if !matches!(model_name, "qwen3_5" | "qwen35") {
                return Err(Error::Other(format!(
                    "{requested_path} supports Qwen3.5 only, not `{model_name}`"
                )));
            }
            if device != Device::Cpu {
                return Err(Error::Other(format!(
                    "{requested_path} requires the Qwen3.5 CPU/Accelerate body (`--device cpu`)"
                )));
            }
            if matches!(options.text_weight_dtype, Some(dtype) if dtype != DType::F32)
                || options.precision != ModelPrecision::Auto
            {
                return Err(Error::Other(format!(
                    "{requested_path} requires F32 native text weights (`--dtype fp32`)"
                )));
            }
            if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                return Err(Error::Other(format!(
                    "{requested_path} requires Apple Silicon macOS"
                )));
            }
        }

        register_builtin_models();
        let backend = create_backend(device)?;
        let device_name = match device {
            Device::Cuda(_) => Some("cuda"),
            Device::Cpu => None,
        };

        let specific_name = device_name.map(|suffix| format!("{model_name}-{suffix}"));
        let factory = specific_name
            .as_deref()
            .and_then(registry::get)
            .or_else(|| registry::get(model_name))
            .ok_or_else(|| {
                Error::Other(format!(
                    "no model implementation for `{model_name}` on {device}"
                ))
            })?;
        factory(path, device, backend, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "metal-w8-head-mlp"))]
    #[test]
    fn explicit_metal_request_fails_when_feature_is_absent() {
        let options = LoadOptions {
            model_name: Some("qwen3_5".into()),
            metal_w8_lm_head: true,
            ..LoadOptions::default()
        };
        let error = AutoModel::load_model(Device::Cpu, Path::new("/does/not/matter"), &options)
            .err()
            .expect("an unavailable Metal W8 feature must fail closed");
        assert!(error
            .to_string()
            .contains("not built with the `metal-w8` feature"));
    }

    #[cfg(not(feature = "metal-w8-head-mlp"))]
    #[test]
    fn explicit_metal_mlp_request_fails_when_feature_is_absent() {
        let options = LoadOptions {
            model_name: Some("qwen3_5".into()),
            text_weight_dtype: Some(DType::F32),
            metal_w8_mlp_block: true,
            ..LoadOptions::default()
        };
        let error = AutoModel::load_model(Device::Cpu, Path::new("/does/not/matter"), &options)
            .err()
            .expect("an unavailable Metal W8 MLP feature must fail closed");
        assert!(error
            .to_string()
            .contains("not built with the `metal-w8` feature"));
    }

    #[cfg(feature = "metal-w8-head-mlp")]
    #[test]
    fn explicit_metal_request_rejects_another_model_family() {
        let options = LoadOptions {
            model_name: Some("llama".into()),
            metal_w8_lm_head: true,
            ..LoadOptions::default()
        };
        let error = AutoModel::load_model(Device::Cpu, Path::new("/does/not/matter"), &options)
            .err()
            .expect("Metal W8 must not fall back on another model family");
        assert!(error.to_string().contains("supports Qwen3.5 only"));
    }

    #[cfg(feature = "metal-w8-head-mlp")]
    #[test]
    fn explicit_metal_mlp_request_rejects_another_model_family() {
        let options = LoadOptions {
            model_name: Some("llama".into()),
            text_weight_dtype: Some(DType::F32),
            metal_w8_mlp_block: true,
            ..LoadOptions::default()
        };
        let error = AutoModel::load_model(Device::Cpu, Path::new("/does/not/matter"), &options)
            .err()
            .expect("Metal W8 MLP must not be ignored by another model family");
        assert!(error.to_string().contains("supports Qwen3.5 only"));
    }

    #[cfg(feature = "metal-w8-head-mlp")]
    #[test]
    fn explicit_metal_mlp_request_rejects_cuda_before_loading() {
        let options = LoadOptions {
            model_name: Some("qwen3_5".into()),
            text_weight_dtype: Some(DType::F32),
            metal_w8_mlp_block: true,
            ..LoadOptions::default()
        };
        let error = AutoModel::load_model(Device::Cuda(0), Path::new("/does/not/matter"), &options)
            .err()
            .expect("Metal W8 MLP must reject CUDA before filesystem access");
        assert!(error.to_string().contains("--device cpu"));
    }

    #[cfg(feature = "metal-w8-head-mlp")]
    #[test]
    fn explicit_metal_mlp_request_rejects_bf16_before_loading() {
        let options = LoadOptions {
            model_name: Some("qwen3_5".into()),
            text_weight_dtype: Some(DType::BF16),
            metal_w8_mlp_block: true,
            ..LoadOptions::default()
        };
        let error = AutoModel::load_model(Device::Cpu, Path::new("/does/not/matter"), &options)
            .err()
            .expect("Metal W8 MLP must reject BF16 before filesystem access");
        assert!(error.to_string().contains("requires F32"));
    }
}
