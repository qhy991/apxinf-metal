//! LLM model architectures and abstractions.

mod accelerator;
#[cfg(feature = "model-registry")]
pub mod auto;
#[cfg(feature = "model-registry")]
pub mod builtin;
#[cfg(feature = "model-llama")]
pub mod debug;
#[cfg(feature = "model-llama")]
pub mod llama;
pub mod llm_trait;
#[cfg(feature = "model-pi05")]
pub mod pi05;
pub mod profiling;
#[cfg(feature = "model-qwen35")]
pub mod qwen35;
#[cfg(feature = "model-qwen3vl")]
pub mod qwen3vl;
#[cfg(feature = "model-registry")]
pub mod registry;
#[cfg(feature = "model-registry")]
pub mod vla;

#[cfg(feature = "model-registry")]
pub use auto::{AutoModel, LoadOptions, LoadedModel, ModelPrecision, SyntheticWeights};
#[cfg(feature = "model-registry")]
pub use builtin::register_builtin_models;
#[cfg(feature = "model-llama")]
pub use debug::{DebugCapture, DebugConfig};
#[cfg(all(feature = "model-llama", feature = "cuda"))]
pub use llama::{DecodeGraph, DecodeGraphConfig, DecodeGraphWeights, DecodeLayerWeights};
#[cfg(feature = "model-llama")]
pub use llama::{GeneralLlama, KVCache, LlamaModel, LlamaWeights, TransformerLayer};
pub use llm_trait::{generate_streaming, ImageInput, LlmCapabilities, LlmInput, LlmTrait};
#[cfg(feature = "model-pi05")]
pub use pi05::{Pi05Config, Pi05PerformanceProfile};
pub use profiling::GenerationProfile;
#[cfg(feature = "model-qwen35")]
pub use qwen35::{
    GeneralQwen35, Qwen35AttentionWeights, Qwen35Config, Qwen35HybridState, Qwen35LayerType,
    Qwen35LinearState, Qwen35TextWeights, Qwen35WeightMetadata, Qwen35WeightSchema,
    Qwen35WeightValidation,
};
#[cfg(feature = "model-qwen3vl")]
pub use qwen3vl::{GeneralQwen3VL, Qwen3VLConfig, Qwen3VLTextWeights};
#[cfg(feature = "model-registry")]
pub use registry::{get, list, register};
#[cfg(feature = "model-registry")]
pub use vla::{
    Action, ImageLayout, InferenceSpec, Observation, PreparedInference, VisionObservation,
    VlaRuntime,
};
