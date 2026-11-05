//! Built-in model registrations used by [`crate::AutoModel`].

#[cfg(feature = "model-llama")]
use std::collections::HashMap;
#[cfg(any(
    feature = "model-llama",
    feature = "model-qwen35",
    feature = "model-qwen3vl"
))]
use std::{path::Path, sync::Arc};

#[cfg(any(
    feature = "model-llama",
    feature = "model-qwen35",
    feature = "model-qwen3vl"
))]
use apxinf_core::{Backend, Device, Error, Result};
#[cfg(feature = "model-llama")]
use apxinf_core::{DType, Tensor};

#[cfg(any(
    feature = "model-llama",
    feature = "model-qwen35",
    feature = "model-qwen3vl"
))]
use crate::auto::{LoadOptions, LoadedModel};
#[cfg(feature = "model-llama")]
use crate::llama::{GeneralLlama, LlamaWeights};
#[cfg(feature = "model-qwen35")]
use crate::qwen35::{GeneralQwen35, Qwen35Config};
#[cfg(feature = "model-qwen3vl")]
use crate::qwen3vl::{GeneralQwen3VL, Qwen3VLConfig};
#[cfg(any(
    feature = "model-llama",
    feature = "model-qwen35",
    feature = "model-qwen3vl"
))]
use crate::registry;

/// Register every implementation shipped in this crate. Re-registering is
/// harmless and keeps `AutoModel::load_model` self-contained for users.
pub fn register_builtin_models() {
    #[cfg(feature = "model-llama")]
    registry::register("llama", load_llama);
    #[cfg(feature = "model-qwen3vl")]
    {
        registry::register("qwen3_vl", load_qwen3vl);
        registry::register("qwen3vl", load_qwen3vl);
    }
    #[cfg(feature = "model-qwen35")]
    {
        registry::register("qwen3_5", load_qwen35);
        registry::register("qwen35", load_qwen35);
    }

    #[cfg(all(feature = "model-pi05", feature = "cuda"))]
    crate::pi05::register_builtin();
}

#[cfg(feature = "model-qwen35")]
fn load_qwen35(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if device != Device::Cpu {
        return Err(Error::Other(
            "Qwen3.5 native inference currently supports CPU/Accelerate only; Metal is the next backend milestone"
                .into(),
        ));
    }
    let model_dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let config = Qwen35Config::from_json_file(&model_dir.join("config.json"))?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path_filtered(path, |name| {
        name.starts_with("model.language_model.") || name == "lm_head.weight"
    })
    .map_err(|error| Error::Other(format!("load {}: {error}", path.display())))?;
    let max_context = options.max_context.unwrap_or(4096);
    let model = GeneralQwen35::from_weights_with_backend_metal_options(
        config,
        tensors,
        backend,
        max_context,
        options.metal_w8_lm_head,
        options.metal_w8_mlp_block,
    )?;
    Ok(LoadedModel::Text(Box::new(model)))
}

#[cfg(feature = "model-llama")]
fn load_llama(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    reject_qwen35_metal_options(options, "Llama")?;
    let (mut tensors, metadata) = apxinf_loader::safetensors::load_native_path(path)
        .map_err(|error| Error::Other(format!("load {}: {error}", path.display())))?;
    if let Some(dtype) = options.text_weight_dtype {
        if !matches!(dtype, DType::F32 | DType::BF16) {
            return Err(Error::Other(format!(
                "Llama text weights support f32 or bf16, not {dtype}"
            )));
        }
    }
    if matches!(device, Device::Cpu) || options.text_weight_dtype == Some(DType::F32) {
        upcast_bf16_weights(&mut tensors)?;
    }
    let config = apxinf_loader::safetensors::config_from_metadata(&metadata);
    let weights = LlamaWeights::from_map(&config, tensors)?;
    Ok(LoadedModel::Text(Box::new(GeneralLlama::new(
        config, weights, backend,
    )?)))
}

#[cfg(feature = "model-qwen3vl")]
fn load_qwen3vl(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    reject_qwen35_metal_options(options, "Qwen3-VL")?;
    let model_dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let config = Qwen3VLConfig::from_json_file(&model_dir.join("config.json"))?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
        .map_err(|error| Error::Other(format!("load {}: {error}", path.display())))?;
    let model = GeneralQwen3VL::from_weights_with_backend(config, tensors, backend)?;
    Ok(LoadedModel::Text(Box::new(model)))
}

#[cfg(any(feature = "model-llama", feature = "model-qwen3vl"))]
fn reject_qwen35_metal_options(options: &LoadOptions, model_family: &str) -> Result<()> {
    if options.metal_w8_lm_head || options.metal_w8_mlp_block {
        return Err(Error::Other(format!(
            "Qwen3.5 Metal W8 options cannot be used with {model_family}"
        )));
    }
    Ok(())
}

#[cfg(feature = "model-llama")]
fn upcast_bf16_weights(tensors: &mut HashMap<String, Tensor>) -> Result<()> {
    for tensor in tensors.values_mut() {
        if tensor.dtype() != DType::BF16 {
            continue;
        }
        let shape = tensor.shape().dims().to_vec();
        *tensor = Tensor::from_f32(shape, &tensor.to_f32_vec()?)?;
    }
    Ok(())
}
