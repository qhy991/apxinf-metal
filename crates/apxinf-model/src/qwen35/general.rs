//! Correctness-first Qwen3.5 text runtime.
//!
//! The initial implementation deliberately supports only the CPU backend. HF
//! checkpoint matrices arrive as `[out, in]`; construction converts them to
//! FP32 and physically packs `[in, out]` matrices once. This also lets the
//! runtime split GDN Q/K/V channels and deinterleave each full-attention
//! query head's `[query, output_gate]` rows without a hot-path slice op.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};
use apxinf_loader::ModelConfig;

use super::config::{Qwen35Config, Qwen35LayerType, Qwen35TextConfig};
use super::state::{Qwen35HybridState, Qwen35LinearState};
use super::weights::{
    Qwen35AttentionWeights, Qwen35FullAttentionWeights, Qwen35LayerWeights,
    Qwen35LinearAttentionWeights, Qwen35MlpWeights, Qwen35TextWeights,
};
use crate::accelerator::create_backend;
use crate::llm_trait::{LlmInput, LlmTrait};

pub struct GeneralQwen35 {
    config: Qwen35Config,
    weights: RuntimeWeights,
    backend: Arc<dyn Backend>,
    state: Qwen35HybridState,
    #[cfg(feature = "metal-w8-head-mlp")]
    metal_w8_lm_head: Option<apxinf_metal::MetalW8LmHead>,
    #[cfg(feature = "metal-w8-head-mlp")]
    metal_w8_lm_head_stats: Option<Qwen35MetalW8LmHeadStats>,
    #[cfg(feature = "metal-experiments")]
    metal_w8_body: Option<Qwen35MetalW8Body>,
    #[cfg(feature = "metal-w8-head-mlp")]
    metal_w8_mlp_blocks: Option<Qwen35MetalW8MlpBlocks>,
    #[cfg(feature = "metal-experiments")]
    metal_w8_gdn: Option<Qwen35MetalW8GdnLayer>,
    #[cfg(feature = "metal-experiments")]
    metal_w8_linear_layer: Option<Qwen35MetalW8LinearLayer>,
    #[cfg(feature = "metal-experiments")]
    metal_w8_all_linear_layers_precision_v2: Option<Qwen35MetalW8AllLinearLayersPrecisionV2>,
    #[cfg(feature = "metal-experiments")]
    metal_w8_linear_layer_stacks_v1: Option<Qwen35MetalW8LinearLayerStacksV1>,
    #[cfg(feature = "metal-experiments")]
    metal_w8_mlp_stack3_boundary_body_v1: Option<Qwen35MetalW8MlpStack3BoundaryBodyV1>,
    #[cfg(feature = "metal-experiments")]
    metal_w8_mlp_stack3_boundary_tail_head_v1: Option<Qwen35MetalW8MlpStack3BoundaryTailHeadV1>,
    #[cfg(feature = "metal-experiments")]
    metal_w8_stack3_lm_head_v2_terminal_error: Option<bool>,
    #[cfg(feature = "metal-experiments")]
    packed_w8_linear_layer_reference: Option<Qwen35PackedW8LinearLayerReference>,
    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fail_after_layer_once_for_test: Option<usize>,
    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fail_stack3_lm_head_v2_before_submit_once_for_test: bool,
    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fail_mlp_stack3_boundary_final_mlp_after_submit_once_for_test: bool,
    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    boundary_tail_head_fault_once_for_test: Option<Qwen35BoundaryTailHeadFaultV1ForTest>,
    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fail_boundary_tail_head_rerank_once_for_test: bool,
}

#[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
#[derive(Clone, Copy)]
enum Qwen35BoundaryTailHeadFaultV1ForTest {
    TailPostExecution,
    TailNonFiniteOutput,
    TailDuplicateCandidate,
    TailOutOfRangeCandidate,
}

/// Observable receipt for one explicitly selected, decode-only Metal W8 GDN
/// attention block. One successful decode is one command buffer and one wait.
#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8GdnStats {
    pub layer_index: usize,
    pub prefill_seed_calls: usize,
    pub decode_calls: usize,
    pub command_buffers: usize,
    pub waits: usize,
    pub committed_state_version: u64,
    /// Host-observed input copy, submission/wait, and output copy.
    pub block_elapsed_ns: u128,
}

#[cfg(feature = "metal-experiments")]
struct Qwen35MetalW8GdnLayer {
    layer_index: usize,
    dimensions: apxinf_metal::GdnDimensions,
    block: apxinf_metal::MetalW8GdnBlock,
    prefill_seed_calls: usize,
    block_elapsed_ns: u128,
    #[cfg(all(test, debug_assertions))]
    fail_next_decode_after_scratch: bool,
}

/// Observable receipt for one explicitly selected complete decode-only Metal
/// W8 linear-attention layer.
#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8LinearLayerStats {
    pub layer_index: usize,
    pub prefill_seed_calls: usize,
    pub decode_calls: usize,
    pub successful_decodes: usize,
    pub failed_decodes: usize,
    pub command_buffers: usize,
    pub compute_encoders: usize,
    pub commits: usize,
    pub waits: usize,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub committed_state_version: u64,
    pub terminal_error: bool,
    pub block_elapsed_ns: u128,
}

/// Fixed precision profile supported by the versioned complete-layer Metal
/// diagnostic. No default or registry path constructs this profile.
#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35MetalW8LinearLayerPrecisionProfile {
    GdnOutG32V2,
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8LinearLayerPrecisionProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GdnOutG32V2 => "gdn-out-g32-v2",
        }
    }

    pub const fn mechanism(self) -> &'static str {
        "metal-w8-linear-layer-precision-v2"
    }
}

/// Extended receipt exposed only for the versioned precision-v2 lane. The
/// legacy receipt remains byte-for-byte shape-compatible with its old API.
#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8LinearLayerPrecisionV2Stats {
    pub profile: Qwen35MetalW8LinearLayerPrecisionProfile,
    pub mechanism: &'static str,
    pub quantization: apxinf_metal::LinearLayerQuantizationLedger,
    pub execution: Qwen35MetalW8LinearLayerStats,
}

/// Aggregate receipt for the explicit all-linear-layers precision-v2 lane.
/// Linear-attention layers own their complete GDN+MLP block; only the MLPs of
/// full-attention layers use the existing standalone Metal W8 MLP block.
#[cfg(feature = "metal-experiments")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8AllLinearLayersPrecisionV2Stats {
    pub profile: Qwen35MetalW8LinearLayerPrecisionProfile,
    pub mechanism: &'static str,
    pub full_attention_mlp_mechanism: &'static str,
    pub linear_layers: Vec<Qwen35MetalW8LinearLayerPrecisionV2Stats>,
    pub full_attention_mlp_layers: Vec<Qwen35MetalW8MlpBlockStats>,
    pub terminal_error: bool,
}

#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8LinearLayerBufferLedger {
    pub layer_index: usize,
    pub ledger: apxinf_metal::LinearLayerBufferLedger,
}

#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8MlpBlockBufferLedger {
    pub layer_index: usize,
    pub ledger: apxinf_metal::MlpBlockBufferLedger,
}

#[cfg(feature = "metal-experiments")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8AllLinearLayersPrecisionV2AggregateLedger {
    pub scope: &'static str,
    pub exclusions: &'static str,
    pub includes_lm_head: bool,
    pub linear_layers: Vec<Qwen35MetalW8LinearLayerBufferLedger>,
    pub full_attention_mlp_layers: Vec<Qwen35MetalW8MlpBlockBufferLedger>,
    pub total_persistent_mtlbuffer_bytes: usize,
    pub allocated_buffers: usize,
    pub shared_buffers: usize,
    pub private_buffers: usize,
    pub host_to_device_bytes_per_decode: usize,
    pub device_to_host_bytes_per_decode: usize,
    pub state_host_transfer_bytes_per_decode: usize,
    pub command_buffers_per_decode: usize,
    pub compute_encoders_per_decode: usize,
    pub commits_per_decode: usize,
    pub waits_per_decode: usize,
}

#[cfg(feature = "metal-experiments")]
struct Qwen35MetalW8LinearLayer {
    layer_index: usize,
    dimensions: apxinf_metal::GdnDimensions,
    block: apxinf_metal::MetalW8LinearLayerBlock,
    prefill_seed_calls: usize,
    block_elapsed_ns: u128,
    seeded: bool,
    terminal_error: bool,
    precision_profile: Option<Qwen35MetalW8LinearLayerPrecisionProfile>,
    quantization: Option<apxinf_metal::LinearLayerQuantizationLedger>,
    #[cfg(all(test, debug_assertions))]
    fail_next_decode_after_scratch: bool,
}

#[cfg(feature = "metal-experiments")]
struct Qwen35MetalW8AllLinearLayersPrecisionV2 {
    profile: Qwen35MetalW8LinearLayerPrecisionProfile,
    layers: Vec<Option<Qwen35MetalW8LinearLayer>>,
    terminal_error: bool,
}

/// Observable receipt for one fixed-depth, three-layer Metal transaction.
/// The v1 mechanism deliberately performs no host finite check between the
/// three layers; only its final output is checked before atomic state commit.
#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8LinearLayerStack3V1Stats {
    pub layer_indices: [usize; 3],
    pub mechanism: &'static str,
    pub gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
    pub gdn_function_chain: &'static str,
    pub quantization: [apxinf_metal::LinearLayerQuantizationLedger; 3],
    pub prefill_seed_calls: [usize; 3],
    pub execution: apxinf_metal::LinearLayerStack3MetalStats,
    pub last_gdn_core_receipt: Option<apxinf_metal::GdnCoreProductionReceiptV1>,
    pub kernel_dispatches_per_decode: usize,
    pub explicit_buffer_barriers_per_decode: usize,
    pub intermediate_host_finite_checks_per_decode: usize,
    pub final_output_finite_checks_per_decode: usize,
    pub terminal_error: bool,
    pub block_elapsed_ns: u128,
}

#[cfg(feature = "metal-experiments")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8LinearLayerStacksV1Stats {
    pub mechanism: &'static str,
    pub full_attention_mlp_mechanism: &'static str,
    pub stacks: Vec<Qwen35MetalW8LinearLayerStack3V1Stats>,
    pub full_attention_mlp_layers: Vec<Qwen35MetalW8MlpBlockStats>,
    pub terminal_error: bool,
}

#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8LinearLayerStack3BufferLedger {
    pub layer_indices: [usize; 3],
    pub ledger: apxinf_metal::LinearLayerStack3BufferLedger,
}

#[cfg(feature = "metal-experiments")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8LinearLayerStacksV1AggregateLedger {
    pub scope: &'static str,
    pub exclusions: &'static str,
    pub includes_lm_head: bool,
    pub stacks: Vec<Qwen35MetalW8LinearLayerStack3BufferLedger>,
    pub full_attention_mlp_layers: Vec<Qwen35MetalW8MlpBlockBufferLedger>,
    pub total_persistent_mtlbuffer_bytes: usize,
    pub allocated_buffers: usize,
    pub shared_buffers: usize,
    pub private_buffers: usize,
    pub host_to_device_bytes_per_decode: usize,
    pub device_to_host_bytes_per_decode: usize,
    pub state_host_transfer_bytes_per_decode: usize,
    pub command_buffers_per_decode: usize,
    pub compute_encoders_per_decode: usize,
    pub commits_per_decode: usize,
    pub waits_per_decode: usize,
    pub intermediate_host_finite_checks_per_decode: usize,
    pub final_output_finite_checks_per_decode: usize,
}

/// Versioned diagnostic ledger for six Stack3 body transactions, six
/// full-attention Metal MLP blocks, and the existing tied top-4 head.
#[cfg(feature = "metal-experiments")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8Stack3LmHeadV2AggregateLedger {
    pub scope: &'static str,
    pub exclusions: &'static str,
    pub includes_lm_head: bool,
    pub body: Qwen35MetalW8LinearLayerStacksV1AggregateLedger,
    pub lm_head: apxinf_metal::LmHeadBufferLedger,
    pub total_persistent_mtlbuffer_bytes: usize,
    pub allocated_buffers: usize,
    pub shared_buffers: usize,
    pub private_buffers: usize,
    pub host_to_device_bytes_per_call: usize,
    pub device_to_host_bytes_per_call: usize,
    pub state_host_transfer_bytes_per_call: usize,
    pub command_buffers_per_call: usize,
    pub compute_encoders_per_call: usize,
    pub commits_per_call: usize,
    pub waits_per_call: usize,
    pub intermediate_host_finite_checks_per_call: usize,
    pub final_output_finite_checks_per_call: usize,
}

#[cfg(feature = "metal-experiments")]
struct Qwen35MetalW8LinearLayerStack3V1 {
    layer_indices: [usize; 3],
    gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
    dimensions: apxinf_metal::GdnDimensions,
    block: apxinf_metal::MetalW8LinearLayerStack3,
    quantization: [apxinf_metal::LinearLayerQuantizationLedger; 3],
    pending_prefill_states: [Option<apxinf_metal::GdnDecodeState>; 3],
    prefill_seed_calls: [usize; 3],
    block_elapsed_ns: u128,
    seeded: bool,
    terminal_error: bool,
    #[cfg(all(test, debug_assertions))]
    fail_next_decode_after_scratch: bool,
}

#[cfg(feature = "metal-experiments")]
struct Qwen35MetalW8LinearLayerStacksV1 {
    stacks: Vec<Qwen35MetalW8LinearLayerStack3V1>,
    layer_to_stack: Vec<Option<(usize, usize)>>,
    terminal_error: bool,
    owns_full_attention_mlp_blocks: bool,
}

#[cfg(feature = "metal-experiments")]
const QWEN35_MLP_STACK3_BOUNDARY_INITIAL_STACK_V1: [usize; 3] = [0, 1, 2];

#[cfg(feature = "metal-experiments")]
const QWEN35_MLP_STACK3_BOUNDARY_REGIONS_V1: [(usize, [usize; 3]); 5] = [
    (3, [4, 5, 6]),
    (7, [8, 9, 10]),
    (11, [12, 13, 14]),
    (15, [16, 17, 18]),
    (19, [20, 21, 22]),
];

#[cfg(feature = "metal-experiments")]
const QWEN35_MLP_STACK3_BOUNDARY_FINAL_MLP_V1: usize = 23;

#[cfg(feature = "metal-experiments")]
const fn qwen35_gdn_core_profile_v1_label(profile: apxinf_metal::GdnCoreProfileV1) -> &'static str {
    match profile {
        apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch => "legacy-four-dispatch",
        apxinf_metal::GdnCoreProfileV1::QkStagedFourDispatch => "qk-staged-four-dispatch-control",
        apxinf_metal::GdnCoreProfileV1::Fused128 => "gdn-core-fused-v1",
    }
}

#[cfg(feature = "metal-experiments")]
fn qwen35_gdn_core_production_receipt_v1_json(
    receipt: Option<apxinf_metal::GdnCoreProductionReceiptV1>,
) -> serde_json::Value {
    let Some(receipt) = receipt else {
        return serde_json::Value::Null;
    };
    serde_json::json!({
        "profile": qwen35_gdn_core_profile_v1_label(receipt.profile),
        "function_chain": receipt.function_chain,
        "gdn_core_seams": receipt.gdn_core_seams,
        "kernel_dispatches": receipt.kernel_dispatches,
        "explicit_buffer_barriers": receipt.explicit_buffer_barriers,
        "recurrent_or_fused_threads_per_threadgroup": receipt.recurrent_or_fused_threads_per_threadgroup,
        "threadgroups": receipt.threadgroups,
        "launched_threads": receipt.launched_threads,
        "pipeline_thread_execution_width": receipt.pipeline_thread_execution_width,
        "source_declared_threadgroup_memory_bytes": receipt.source_declared_threadgroup_memory_bytes,
        "pipeline_static_threadgroup_memory_bytes": receipt.pipeline_static_threadgroup_memory_bytes,
        "internal_threadgroup_barrier_sites_per_threadgroup": receipt.internal_threadgroup_barrier_sites_per_threadgroup,
        "fixed_shape_validated": receipt.fixed_shape_validated,
        "rms_norm_eps_bits": receipt.rms_norm_eps_bits,
        "persistent_output_groups_per_row": receipt.persistent_output_groups_per_row,
        "core_kernel_output_groups_per_row": receipt.core_kernel_output_groups_per_row,
    })
}

#[cfg(feature = "metal-experiments")]
const fn qwen35_stack3_mechanism_for_gdn_core_profile_v1(
    profile: apxinf_metal::GdnCoreProfileV1,
) -> &'static str {
    match profile {
        apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch => "metal-w8-linear-layer-stack3-v1",
        apxinf_metal::GdnCoreProfileV1::Fused128 => {
            "metal-w8-linear-layer-stack3-gdn-core-fused-v1"
        }
        apxinf_metal::GdnCoreProfileV1::QkStagedFourDispatch => {
            "invalid-production-qk-staged-control"
        }
    }
}

#[cfg(feature = "metal-experiments")]
const fn qwen35_boundary_mechanism_for_gdn_core_profile_v1(
    profile: apxinf_metal::GdnCoreProfileV1,
) -> &'static str {
    match profile {
        apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch => "metal-w8-mlp-stack3-boundary-v1",
        apxinf_metal::GdnCoreProfileV1::Fused128 => {
            "metal-w8-mlp-stack3-boundary-gdn-core-fused-v1"
        }
        apxinf_metal::GdnCoreProfileV1::QkStagedFourDispatch => {
            "invalid-production-qk-staged-control"
        }
    }
}

#[cfg(feature = "metal-experiments")]
const fn qwen35_boundary_tail_head_mechanism_for_gdn_core_profile_v1(
    profile: apxinf_metal::GdnCoreProfileV1,
) -> &'static str {
    match profile {
        apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch => {
            "metal-w8-mlp-stack3-boundary-tail-head-v1"
        }
        apxinf_metal::GdnCoreProfileV1::Fused128 => {
            "metal-w8-mlp-stack3-boundary-tail-head-gdn-core-fused-v1"
        }
        apxinf_metal::GdnCoreProfileV1::QkStagedFourDispatch => {
            "invalid-production-qk-staged-control"
        }
    }
}

#[cfg(feature = "metal-experiments")]
fn validate_qwen35_production_gdn_core_profile_v1(
    profile: apxinf_metal::GdnCoreProfileV1,
) -> Result<()> {
    match profile {
        apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch
        | apxinf_metal::GdnCoreProfileV1::Fused128 => Ok(()),
        apxinf_metal::GdnCoreProfileV1::QkStagedFourDispatch => Err(Error::Other(
            "qwen3.5 Metal W8 production route rejects the diagnostic-only qk-staged GDN core control"
                .into(),
        )),
    }
}

#[cfg(feature = "metal-experiments")]
fn qwen35_matches_gdn_core_fused_v1_shape(config: &Qwen35TextConfig) -> bool {
    (
        config.hidden_size,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        config.linear_conv_kernel_dim,
        config.rms_norm_eps.to_bits(),
    ) == (1_024, 16, 16, 128, 128, 4, 1.0e-6_f32.to_bits())
}

#[cfg(feature = "metal-experiments")]
fn validate_qwen35_gdn_core_fused_v1_shape(config: &Qwen35TextConfig) -> Result<()> {
    if !qwen35_matches_gdn_core_fused_v1_shape(config) {
        return Err(Error::Other(format!(
            "qwen3.5 Metal W8 gdn-core-fused-v1 production route is fixed to H=1024/KH=16/VH=16/KD=128/VD=128/conv=4/eps=1e-6, got H={}/KH={}/VH={}/KD={}/VD={}/conv={}/eps_bits=0x{:08x}",
            config.hidden_size,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
            config.rms_norm_eps.to_bits(),
        )));
    }
    Ok(())
}

#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8MlpStack3BoundaryRegionV1Stats {
    pub boundary_mlp_layer_index: usize,
    pub stack_layer_indices: [usize; 3],
    pub mechanism: &'static str,
    pub gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
    pub gdn_function_chain: &'static str,
    pub quantization: [apxinf_metal::LinearLayerQuantizationLedger; 3],
    pub prefill_seed_calls: [usize; 3],
    pub execution: apxinf_metal::MlpStack3BoundaryMetalStatsV1,
    pub last_gdn_core_receipt: Option<apxinf_metal::GdnCoreProductionReceiptV1>,
    pub kernel_dispatches_per_decode: usize,
    pub explicit_buffer_barriers_per_decode: usize,
    pub terminal_error: bool,
    pub block_elapsed_ns: u128,
}

#[cfg(feature = "metal-experiments")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8MlpStack3BoundaryBodyV1Stats {
    pub mechanism: &'static str,
    pub initial_stack: Qwen35MetalW8LinearLayerStack3V1Stats,
    pub boundaries: Vec<Qwen35MetalW8MlpStack3BoundaryRegionV1Stats>,
    pub final_mlp: Qwen35MetalW8MlpBlockStats,
    pub terminal_error: bool,
}

#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8MlpStack3BoundaryRegionBufferLedgerV1 {
    pub boundary_mlp_layer_index: usize,
    pub stack_layer_indices: [usize; 3],
    pub ledger: apxinf_metal::MlpStack3BoundaryBufferLedgerV1,
}

#[cfg(feature = "metal-experiments")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8MlpStack3BoundaryBodyV1AggregateLedger {
    pub scope: &'static str,
    pub exclusions: &'static str,
    pub includes_lm_head: bool,
    pub initial_stack: Qwen35MetalW8LinearLayerStack3BufferLedger,
    pub boundaries: Vec<Qwen35MetalW8MlpStack3BoundaryRegionBufferLedgerV1>,
    pub final_mlp: Qwen35MetalW8MlpBlockBufferLedger,
    pub total_persistent_mtlbuffer_bytes: usize,
    pub allocated_buffers: usize,
    pub shared_buffers: usize,
    pub private_buffers: usize,
    pub host_to_device_bytes_per_decode: usize,
    pub device_to_host_bytes_per_decode: usize,
    pub state_host_transfer_bytes_per_decode: usize,
    pub command_buffers_per_decode: usize,
    pub compute_encoders_per_decode: usize,
    pub kernel_dispatches_per_decode: usize,
    pub commits_per_decode: usize,
    pub waits_per_decode: usize,
    pub intermediate_host_finite_checks_per_decode: usize,
    pub final_output_finite_checks_per_decode: usize,
}

#[cfg(feature = "metal-experiments")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger {
    pub scope: &'static str,
    pub exclusions: &'static str,
    pub includes_lm_head: bool,
    pub gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
    pub gdn_function_chain: &'static str,
    pub initial_stack: Qwen35MetalW8LinearLayerStack3BufferLedger,
    pub boundaries: Vec<Qwen35MetalW8MlpStack3BoundaryRegionBufferLedgerV1>,
    pub tail_layer_index: usize,
    pub tail: apxinf_metal::TailMlpHeadBufferLedgerV1,
    pub total_persistent_mtlbuffer_bytes: usize,
    pub allocated_buffers: usize,
    pub shared_buffers: usize,
    pub private_buffers: usize,
    pub host_to_device_bytes_per_decode: usize,
    pub device_to_host_bytes_per_decode: usize,
    pub state_host_transfer_bytes_per_decode: usize,
    pub command_buffers_per_decode: usize,
    pub compute_encoders_per_decode: usize,
    pub kernel_dispatches_per_decode: usize,
    pub explicit_buffer_barriers_per_decode: usize,
    pub commits_per_decode: usize,
    pub waits_per_decode: usize,
}

#[cfg(feature = "metal-experiments")]
struct Qwen35MetalW8MlpStack3BoundaryRegionV1 {
    boundary_mlp_layer_index: usize,
    stack_layer_indices: [usize; 3],
    gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
    dimensions: apxinf_metal::GdnDimensions,
    block: apxinf_metal::MetalW8MlpStack3BoundaryV1,
    quantization: [apxinf_metal::LinearLayerQuantizationLedger; 3],
    pending_prefill_states: [Option<apxinf_metal::GdnDecodeState>; 3],
    prefill_seed_calls: [usize; 3],
    block_elapsed_ns: u128,
    seeded: bool,
    terminal_error: bool,
    #[cfg(all(test, debug_assertions))]
    fail_next_decode_after_scratch: bool,
}

#[cfg(feature = "metal-experiments")]
struct Qwen35MetalW8MlpStack3BoundaryBodyV1 {
    initial_stack: Qwen35MetalW8LinearLayerStack3V1,
    boundaries: Vec<Qwen35MetalW8MlpStack3BoundaryRegionV1>,
    final_mlp: Qwen35MetalW8MlpBlockLayer,
    terminal_error: bool,
}

/// Independent receipt for the diagnostic lane that replaces the standalone
/// layer-23 MLP and the ordinary output projection with one tail transaction.
#[cfg(feature = "metal-experiments")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8MlpStack3BoundaryTailHeadV1Stats {
    pub mechanism: &'static str,
    pub gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
    pub gdn_function_chain: &'static str,
    pub initial_stack: Qwen35MetalW8LinearLayerStack3V1Stats,
    pub boundaries: Vec<Qwen35MetalW8MlpStack3BoundaryRegionV1Stats>,
    pub tail_layer_index: usize,
    pub tail: apxinf_metal::TailMlpHeadMetalStatsV1,
    pub prefill_body_calls: usize,
    pub prefill_cpu_head_calls: usize,
    pub decode_calls: usize,
    pub teacher_calls: usize,
    pub rerank_elapsed_ns: u128,
    pub terminal_error: bool,
}

#[cfg(feature = "metal-experiments")]
struct Qwen35MetalW8MlpStack3BoundaryTailHeadV1 {
    gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
    initial_stack: Qwen35MetalW8LinearLayerStack3V1,
    boundaries: Vec<Qwen35MetalW8MlpStack3BoundaryRegionV1>,
    tail_layer_index: usize,
    tail: apxinf_metal::MetalW8TailMlpHeadV1,
    prefill_body_calls: usize,
    prefill_cpu_head_calls: usize,
    decode_calls: usize,
    teacher_calls: usize,
    rerank_elapsed_ns: u128,
    terminal_error: bool,
}

/// Receipt for the CPU implementation of the exact packed W8 complete-layer
/// oracle. This exists only to distinguish quantization error from Metal
/// arithmetic in an explicit quality gate.
#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35PackedW8LinearLayerReferenceProfile {
    G64,
    GdnOutG32,
    MlpDownG32,
    GdnOutAndMlpDownG32,
}

#[cfg(feature = "metal-experiments")]
impl Qwen35PackedW8LinearLayerReferenceProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::G64 => "g64",
            Self::GdnOutG32 => "gdn-out-g32",
            Self::MlpDownG32 => "mlp-down-g32",
            Self::GdnOutAndMlpDownG32 => "gdn-out-and-mlp-down-g32",
        }
    }

    const fn gdn_output_group_size(self) -> apxinf_metal::W8GroupSize {
        match self {
            Self::GdnOutG32 | Self::GdnOutAndMlpDownG32 => apxinf_metal::W8GroupSize::G32,
            Self::G64 | Self::MlpDownG32 => apxinf_metal::W8GroupSize::G64,
        }
    }

    const fn mlp_down_group_size(self) -> apxinf_metal::W8GroupSize {
        match self {
            Self::MlpDownG32 | Self::GdnOutAndMlpDownG32 => apxinf_metal::W8GroupSize::G32,
            Self::G64 | Self::GdnOutG32 => apxinf_metal::W8GroupSize::G64,
        }
    }
}

#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35PackedW8LinearLayerReferenceStats {
    pub layer_index: usize,
    pub profile: Qwen35PackedW8LinearLayerReferenceProfile,
    pub quantization: apxinf_metal::LinearLayerQuantizationLedger,
    pub prefill_seed_calls: usize,
    pub decode_calls: usize,
    pub successful_decodes: usize,
    pub failed_decodes: usize,
    pub committed_state_version: u64,
    pub terminal_error: bool,
    pub block_elapsed_ns: u128,
}

#[cfg(feature = "metal-experiments")]
struct Qwen35PackedW8LinearLayerReference {
    layer_index: usize,
    profile: Qwen35PackedW8LinearLayerReferenceProfile,
    quantization: apxinf_metal::LinearLayerQuantizationLedger,
    dimensions: apxinf_metal::GdnDimensions,
    packed: apxinf_metal::PackedW8LinearLayerBlock,
    state: Option<apxinf_metal::GdnDecodeState>,
    prefill_seed_calls: usize,
    decode_calls: usize,
    successful_decodes: usize,
    failed_decodes: usize,
    committed_state_version: u64,
    terminal_error: bool,
    block_elapsed_ns: u128,
    #[cfg(all(test, debug_assertions))]
    fail_next_decode_after_reference: bool,
}

#[cfg(feature = "metal-experiments")]
struct Qwen35PackedW8LinearLayerWeights {
    dimensions: apxinf_metal::GdnDimensions,
    packed: apxinf_metal::PackedW8LinearLayerBlock,
}

/// Observable receipt for the explicitly selected decode-only body lane.
#[cfg(feature = "metal-experiments")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8BodyStats {
    pub layer_index: usize,
    pub decode_calls: usize,
    /// Host-observed input copy, command submission, GPU wait, and output copy.
    pub projection_elapsed_ns: u128,
}

#[cfg(feature = "metal-experiments")]
struct Qwen35MetalW8Body {
    layers: Vec<Option<Qwen35MetalW8BodyLayer>>,
}

#[cfg(feature = "metal-experiments")]
struct Qwen35MetalW8BodyLayer {
    layer_index: usize,
    mlp_gate_up: apxinf_metal::MetalW8MatVec,
    decode_calls: usize,
    projection_elapsed_ns: u128,
}

/// Observable receipt for the complete decode-only Metal W8 MLP block lane.
#[cfg(feature = "metal-w8-head-mlp")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35MetalW8MlpBlockStats {
    pub layer_index: usize,
    pub decode_calls: usize,
    /// Host-observed input copy, one submission/wait, and output copy.
    pub block_elapsed_ns: u128,
}

/// Observable receipt for the explicitly enabled Metal W8 top-4 head. Calls
/// are separated by phase so a combined tracer can prove that CPU body prefill
/// did not accidentally exercise any decode-only MLP block.
#[cfg(feature = "metal-w8-head-mlp")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Qwen35MetalW8LmHeadStats {
    pub prefill_calls: usize,
    pub decode_calls: usize,
    pub teacher_calls: usize,
    pub topk_elapsed_ns: u128,
    pub rerank_elapsed_ns: u128,
}

#[cfg(feature = "metal-w8-head-mlp")]
#[derive(Clone, Copy)]
enum Qwen35MetalW8LmHeadPhase {
    Prefill,
    Decode,
}

#[cfg(feature = "metal-w8-head-mlp")]
impl Qwen35MetalW8LmHeadPhase {
    fn label(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

#[cfg(feature = "metal-w8-head-mlp")]
struct Qwen35MetalW8MlpBlocks {
    layers: Vec<Option<Qwen35MetalW8MlpBlockLayer>>,
}

#[cfg(feature = "metal-w8-head-mlp")]
struct Qwen35MetalW8MlpBlockLayer {
    layer_index: usize,
    mlp: apxinf_metal::MetalW8MlpBlock,
    decode_calls: usize,
    block_elapsed_ns: u128,
}

/// One diagnostic teacher-forced comparison. Timings exclude the earlier
/// model body and the full native oracle projection. The legacy
/// `topk_elapsed_ns` field names the accelerator candidate transaction: for a
/// standalone Metal head it is only the top-4 submission/wait, while the
/// boundary + tail-head v1 lane measures its whole fused layer-23 MLP + final
/// RMS + top-4 transaction.
#[cfg(feature = "metal-experiments")]
pub struct Qwen35MetalTeacherStep {
    pub cpu_token: u32,
    pub w8_candidates: [u32; apxinf_metal::W8_TOP_K],
    pub reranked_token: u32,
    /// Backward-compatible field name; see [`Self::accelerator_candidate_elapsed_ns`].
    pub topk_elapsed_ns: u128,
    /// Exact four-row F32 tied-embedding rerank only.
    pub rerank_elapsed_ns: u128,
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalTeacherStep {
    /// Elapsed time for the accelerator transaction that publishes the four
    /// candidates. This is top-4-only for the standalone head, but includes
    /// fused MLP23 + final RMS + top-4 for boundary + tail-head v1.
    pub const fn accelerator_candidate_elapsed_ns(&self) -> u128 {
        self.topk_elapsed_ns
    }
}

#[cfg(feature = "metal-experiments")]
struct Qwen35BoundaryTailHeadOutputV1 {
    normalized_hidden: Tensor,
    candidates: [u32; apxinf_metal::W8_TOP_K],
    tail_elapsed_ns: u128,
}

impl GeneralQwen35 {
    /// Build a CPU text runtime from an already validated HF-shaped weight map.
    pub fn from_weights(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
    ) -> Result<Self> {
        let backend = create_backend(device)?;
        Self::from_weights_with_backend(config, tensors, backend, max_context)
    }

    /// Explicit constructor for the decode-only Metal W8 tied lm_head.
    /// Unsupported platforms, shapes, or untied checkpoints return an error.
    #[cfg(feature = "metal-w8-head-mlp")]
    pub fn from_weights_with_metal_w8(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
    ) -> Result<Self> {
        let backend = create_backend(device)?;
        Self::from_weights_with_backend_options(config, tensors, backend, max_context, true)
    }

    /// Diagnostic opt-in for exactly one state-resident decode-only Metal W8
    /// GDN attention block. CPU prefill remains authoritative and seeds the
    /// selected layer; ordinary constructors never create this lane.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_metal_w8_gdn_layer(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
        layer_index: usize,
    ) -> Result<Self> {
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        let metal_w8_gdn = Qwen35MetalW8GdnLayer::pack(&weights, &config.text, layer_index)?;
        let mut model =
            Self::new_with_metal_options(config, weights, backend, max_context, false, None, None)?;
        model.metal_w8_gdn = Some(metal_w8_gdn);
        Ok(model)
    }

    /// Diagnostic opt-in for exactly one complete state-resident decode-only
    /// Metal W8 linear-attention layer. CPU prefill remains authoritative and
    /// seeds the selected GDN state; ordinary constructors never create it.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_metal_w8_linear_layer(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
        layer_index: usize,
    ) -> Result<Self> {
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        let metal_w8_linear_layer =
            Qwen35MetalW8LinearLayer::pack(&weights, &config.text, layer_index)?;
        let mut model =
            Self::new_with_metal_options(config, weights, backend, max_context, false, None, None)?;
        model.metal_w8_linear_layer = Some(metal_w8_linear_layer);
        Ok(model)
    }

    /// Diagnostic-only precision-v2 complete linear-attention layer. The
    /// selected layer keeps G64 everywhere except its GDN output projection,
    /// which uses the dedicated G32 Metal ABI and kernel.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_metal_w8_linear_layer_precision_v2(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
        layer_index: usize,
        profile: Qwen35MetalW8LinearLayerPrecisionProfile,
    ) -> Result<Self> {
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        let metal_w8_linear_layer = Qwen35MetalW8LinearLayer::pack_precision_v2(
            &weights,
            &config.text,
            layer_index,
            profile,
        )?;
        let mut model =
            Self::new_with_metal_options(config, weights, backend, max_context, false, None, None)?;
        model.metal_w8_linear_layer = Some(metal_w8_linear_layer);
        Ok(model)
    }

    /// Diagnostic-only precision-v2 route for every linear-attention layer.
    /// Each selected linear layer owns its complete GDN+MLP decode block. The
    /// full-attention layers keep CPU attention and use standalone Metal W8
    /// MLP blocks, so no layer executes its MLP twice.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_metal_w8_all_linear_layers_precision_v2(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
        profile: Qwen35MetalW8LinearLayerPrecisionProfile,
    ) -> Result<Self> {
        let full_attention_mlp_layers = config
            .text
            .layer_types
            .iter()
            .enumerate()
            .filter_map(|(layer_index, layer_type)| {
                (*layer_type == Qwen35LayerType::FullAttention).then_some(layer_index)
            })
            .collect::<Vec<_>>();
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        let all_linear_layers =
            Qwen35MetalW8AllLinearLayersPrecisionV2::pack(&weights, &config.text, profile)?;
        let full_attention_mlp_layers =
            (!full_attention_mlp_layers.is_empty()).then_some(full_attention_mlp_layers);
        let mut model = Self::new_with_metal_options(
            config,
            weights,
            backend,
            max_context,
            false,
            None,
            full_attention_mlp_layers,
        )?;
        model.metal_w8_all_linear_layers_precision_v2 = Some(all_linear_layers);
        Ok(model)
    }

    /// Diagnostic-only stack3-v1 route for one explicit run of three
    /// consecutive linear-attention layers. CPU prefill remains authoritative;
    /// ordinary constructors and registry/default paths never create it.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_metal_w8_linear_layer_stack3_v1(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
        first_layer_index: usize,
    ) -> Result<Self> {
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        let stacks = Qwen35MetalW8LinearLayerStacksV1::pack(
            &weights,
            &config.text,
            &[first_layer_index],
            false,
        )?;
        let mut model =
            Self::new_with_metal_options(config, weights, backend, max_context, false, None, None)?;
        model.metal_w8_linear_layer_stacks_v1 = Some(stacks);
        Ok(model)
    }

    /// Diagnostic-only stack3-v1 route for every maximal three-layer linear-
    /// attention run. Full-attention layers retain CPU attention/KV and use
    /// one standalone Metal W8 MLP, so no MLP is executed twice.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_metal_w8_all_linear_layer_stacks_v1(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
    ) -> Result<Self> {
        let mut stack_starts = Vec::new();
        let mut full_attention_mlp_layers = Vec::new();
        let mut layer_index = 0;
        while layer_index < config.text.layer_types.len() {
            match config.text.layer_types[layer_index] {
                Qwen35LayerType::LinearAttention => {
                    let run_start = layer_index;
                    while layer_index < config.text.layer_types.len()
                        && config.text.layer_types[layer_index] == Qwen35LayerType::LinearAttention
                    {
                        layer_index += 1;
                    }
                    if layer_index - run_start != 3 {
                        return Err(Error::Other(format!(
                            "qwen3.5 Metal W8 all-stack3-v1 requires linear-attention runs of exactly three layers, got {} at {run_start}",
                            layer_index - run_start
                        )));
                    }
                    stack_starts.push(run_start);
                }
                Qwen35LayerType::FullAttention => {
                    full_attention_mlp_layers.push(layer_index);
                    layer_index += 1;
                }
            }
        }
        if stack_starts.is_empty() || full_attention_mlp_layers.is_empty() {
            return Err(Error::Other(
                "qwen3.5 Metal W8 all-stack3-v1 requires both linear stacks and full-attention layers"
                    .into(),
            ));
        }
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        let stacks =
            Qwen35MetalW8LinearLayerStacksV1::pack(&weights, &config.text, &stack_starts, true)?;
        let mut model = Self::new_with_metal_options(
            config,
            weights,
            backend,
            max_context,
            false,
            None,
            Some(full_attention_mlp_layers),
        )?;
        model.metal_w8_linear_layer_stacks_v1 = Some(stacks);
        Ok(model)
    }

    /// Diagnostic-only v1 body route. The initial three linear-attention
    /// layers remain one Stack3 transaction, five full-attention MLP
    /// boundaries each own the following Stack3 transaction, and layer 23
    /// retains one standalone Metal MLP. Ordinary constructors never create
    /// this lane.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_metal_w8_mlp_stack3_boundary_body_v1(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
    ) -> Result<Self> {
        Qwen35MetalW8MlpStack3BoundaryBodyV1::validate_config_schedule(&config.text)?;
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        let body = Qwen35MetalW8MlpStack3BoundaryBodyV1::pack(&weights, &config.text)?;
        let mut model =
            Self::new_with_metal_options(config, weights, backend, max_context, false, None, None)?;
        model.metal_w8_mlp_stack3_boundary_body_v1 = Some(body);
        Ok(model)
    }

    /// Diagnostic-only composite lane. Decode owns the initial Stack3, five
    /// full-attention MLP→Stack3 boundaries, and one fused layer-23 MLP +
    /// final RMS + tied top-4 transaction. CPU/F32 remains authoritative for
    /// prefill and for the exact four-candidate rerank.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
    ) -> Result<Self> {
        Self::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_profile_v1(
            config,
            tensors,
            device,
            max_context,
            apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch,
        )
    }

    /// Explicit-only production-topology continuation of boundary + tail-head
    /// v1. Only the 18 GDN cores inside the initial Stack3 and five boundary
    /// Stack3 transactions select `gdn_core_fused_v1`; prefill, full
    /// attention, MLPs, tail/head, state publication, and submission topology
    /// remain unchanged. No default, AutoModel, registry, or CLI path calls
    /// this constructor.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
    ) -> Result<Self> {
        Self::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_profile_v1(
            config,
            tensors,
            device,
            max_context,
            apxinf_metal::GdnCoreProfileV1::Fused128,
        )
    }

    #[cfg(feature = "metal-experiments")]
    fn from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_profile_v1(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
        gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
    ) -> Result<Self> {
        validate_qwen35_production_gdn_core_profile_v1(gdn_core_profile)?;
        if gdn_core_profile == apxinf_metal::GdnCoreProfileV1::Fused128 {
            validate_qwen35_gdn_core_fused_v1_shape(&config.text)?;
        }
        if !config.text.tie_word_embeddings {
            return Err(Error::Other(
                "qwen3.5 Metal W8 boundary + tail-head v1 requires tied word embeddings".into(),
            ));
        }
        Qwen35MetalW8MlpStack3BoundaryBodyV1::validate_config_schedule(&config.text)?;
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        let lane = Qwen35MetalW8MlpStack3BoundaryTailHeadV1::pack_with_gdn_core_profile_v1(
            &weights,
            &config.text,
            gdn_core_profile,
        )?;
        let mut model =
            Self::new_with_metal_options(config, weights, backend, max_context, false, None, None)?;
        model.metal_w8_mlp_stack3_boundary_tail_head_v1 = Some(lane);
        Ok(model)
    }

    /// Diagnostic-only v2 composite: every maximal three-layer linear run is
    /// one Stack3 transaction, each full-attention layer owns one standalone
    /// Metal MLP block, and the tied output uses the existing top-4 Metal head
    /// plus exact F32 rerank. No registry or default constructor calls this.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_metal_w8_all_linear_layer_stacks_and_lm_head_v2(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
    ) -> Result<Self> {
        let mut stack_starts = Vec::new();
        let mut full_attention_mlp_layers = Vec::new();
        let mut layer_index = 0;
        while layer_index < config.text.layer_types.len() {
            match config.text.layer_types[layer_index] {
                Qwen35LayerType::LinearAttention => {
                    let run_start = layer_index;
                    while layer_index < config.text.layer_types.len()
                        && config.text.layer_types[layer_index] == Qwen35LayerType::LinearAttention
                    {
                        layer_index += 1;
                    }
                    if layer_index - run_start != 3 {
                        return Err(Error::Other(format!(
                            "qwen3.5 Metal W8 Stack3 + lm_head v2 requires linear-attention runs of exactly three layers, got {} at {run_start}",
                            layer_index - run_start
                        )));
                    }
                    stack_starts.push(run_start);
                }
                Qwen35LayerType::FullAttention => {
                    full_attention_mlp_layers.push(layer_index);
                    layer_index += 1;
                }
            }
        }
        if stack_starts.is_empty() || full_attention_mlp_layers.is_empty() {
            return Err(Error::Other(
                "qwen3.5 Metal W8 Stack3 + lm_head v2 requires both linear stacks and full-attention layers"
                    .into(),
            ));
        }
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        let stacks =
            Qwen35MetalW8LinearLayerStacksV1::pack(&weights, &config.text, &stack_starts, true)?;
        let mut model = Self::new_with_metal_options(
            config,
            weights,
            backend,
            max_context,
            true,
            None,
            Some(full_attention_mlp_layers),
        )?;
        model.metal_w8_linear_layer_stacks_v1 = Some(stacks);
        model.metal_w8_stack3_lm_head_v2_terminal_error = Some(false);
        Ok(model)
    }

    /// Diagnostic control for exactly one complete decode-only packed W8
    /// linear-attention layer executed by the canonical CPU reference. This
    /// is gate-only and ordinary constructors never create it.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_packed_w8_linear_layer_reference(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
        layer_index: usize,
    ) -> Result<Self> {
        Self::from_weights_with_packed_w8_linear_layer_reference_profile(
            config,
            tensors,
            device,
            max_context,
            layer_index,
            Qwen35PackedW8LinearLayerReferenceProfile::G64,
        )
    }

    /// CPU-only precision-screen portfolio for the explicit packed complete-
    /// layer custody lane. Ordinary constructors and Metal lanes never call
    /// this profile-aware constructor.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_packed_w8_linear_layer_reference_profile(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
        layer_index: usize,
        profile: Qwen35PackedW8LinearLayerReferenceProfile,
    ) -> Result<Self> {
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        let packed =
            Qwen35PackedW8LinearLayerReference::pack(&weights, &config.text, layer_index, profile)?;
        let mut model =
            Self::new_with_metal_options(config, weights, backend, max_context, false, None, None)?;
        model.packed_w8_linear_layer_reference = Some(packed);
        Ok(model)
    }

    /// Diagnostic opt-in for one decode-only MLP gate+up W8 projection lane.
    /// The ordinary constructors remain the kill switch and never create it.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_metal_w8_body_layer(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
        layer_index: usize,
    ) -> Result<Self> {
        Self::from_weights_with_metal_w8_body_layers(
            config,
            tensors,
            device,
            max_context,
            &[layer_index],
        )
    }

    /// Diagnostic opt-in for a selected set of decode-only MLP gate+up lanes.
    /// Empty, duplicate, and out-of-range sets fail closed.
    #[cfg(feature = "metal-experiments")]
    pub fn from_weights_with_metal_w8_body_layers(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
        layer_indices: &[usize],
    ) -> Result<Self> {
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        Self::new_with_metal_options(
            config,
            weights,
            backend,
            max_context,
            false,
            Some(layer_indices.to_vec()),
            None,
        )
    }

    /// Diagnostic opt-in for one complete decode-only Metal W8 MLP block.
    /// The ordinary and gate+up constructors remain independent kill switches.
    #[cfg(feature = "metal-w8-head-mlp")]
    pub fn from_weights_with_metal_w8_mlp_block_layer(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
        layer_index: usize,
    ) -> Result<Self> {
        Self::from_weights_with_metal_w8_mlp_block_layers(
            config,
            tensors,
            device,
            max_context,
            &[layer_index],
        )
    }

    /// Diagnostic opt-in for selected complete decode-only Metal W8 MLP blocks.
    /// Empty, duplicate, and out-of-range sets fail closed.
    #[cfg(feature = "metal-w8-head-mlp")]
    pub fn from_weights_with_metal_w8_mlp_block_layers(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
        layer_indices: &[usize],
    ) -> Result<Self> {
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        Self::new_with_metal_options(
            config,
            weights,
            backend,
            max_context,
            false,
            None,
            Some(layer_indices.to_vec()),
        )
    }

    /// Diagnostic opt-in combining complete decode-only Metal W8 MLP blocks
    /// for every layer with the existing tied top-4 + F32-rerank head. The
    /// ordinary and single-lane constructors remain independent kill switches.
    #[cfg(feature = "metal-w8-head-mlp")]
    pub fn from_weights_with_metal_w8_mlp_blocks_and_lm_head(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        device: Device,
        max_context: usize,
    ) -> Result<Self> {
        let layer_indices = (0..config.text.n_layers).collect::<Vec<_>>();
        let backend = create_backend(device)?;
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        Self::new_with_metal_options(
            config,
            weights,
            backend,
            max_context,
            true,
            None,
            Some(layer_indices),
        )
    }

    /// Registry-facing constructor that shares the backend created by
    /// `AutoModel` instead of allocating another backend instance.
    pub(crate) fn from_weights_with_backend(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        backend: Arc<dyn Backend>,
        max_context: usize,
    ) -> Result<Self> {
        Self::from_weights_with_backend_options(config, tensors, backend, max_context, false)
    }

    /// Build the CPU body and, only when explicitly requested, a persistent
    /// decode-only Metal W8 tied output projection.
    pub(crate) fn from_weights_with_backend_options(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        backend: Arc<dyn Backend>,
        max_context: usize,
        metal_w8_lm_head: bool,
    ) -> Result<Self> {
        Self::from_weights_with_backend_metal_options(
            config,
            tensors,
            backend,
            max_context,
            metal_w8_lm_head,
            false,
        )
    }

    /// Registry-facing explicit Metal route. A single construction chooses
    /// head-only, all-layer MLP-only, or their verified combination, avoiding
    /// a second model build or duplicate lane packing.
    pub(crate) fn from_weights_with_backend_metal_options(
        config: Qwen35Config,
        tensors: HashMap<String, Tensor>,
        backend: Arc<dyn Backend>,
        max_context: usize,
        metal_w8_lm_head: bool,
        metal_w8_mlp_block: bool,
    ) -> Result<Self> {
        let metal_w8_mlp_block_layers =
            metal_w8_mlp_block.then(|| (0..config.text.n_layers).collect::<Vec<_>>());
        let weights = Qwen35TextWeights::from_map(&config, tensors)?;
        Self::new_with_metal_options(
            config,
            weights,
            backend,
            max_context,
            metal_w8_lm_head,
            None,
            metal_w8_mlp_block_layers,
        )
    }

    /// Pack raw HF `[out, in]` tensors into the CPU runtime representation.
    pub fn new(
        config: Qwen35Config,
        weights: Qwen35TextWeights,
        backend: Arc<dyn Backend>,
        max_context: usize,
    ) -> Result<Self> {
        Self::new_with_options(config, weights, backend, max_context, false)
    }

    fn new_with_options(
        config: Qwen35Config,
        weights: Qwen35TextWeights,
        backend: Arc<dyn Backend>,
        max_context: usize,
        metal_w8_lm_head: bool,
    ) -> Result<Self> {
        Self::new_with_metal_options(
            config,
            weights,
            backend,
            max_context,
            metal_w8_lm_head,
            None,
            None,
        )
    }

    fn new_with_metal_options(
        config: Qwen35Config,
        weights: Qwen35TextWeights,
        backend: Arc<dyn Backend>,
        max_context: usize,
        metal_w8_lm_head: bool,
        metal_w8_body_layers: Option<Vec<usize>>,
        metal_w8_mlp_block_layers: Option<Vec<usize>>,
    ) -> Result<Self> {
        config.text.validate()?;
        if backend.device() != Device::Cpu {
            return Err(Error::Other(
                "Qwen3.5 native runtime currently supports CPU only; Metal/Accelerate kernels are the next backend slice"
                    .into(),
            ));
        }
        if weights.layers.len() != config.text.n_layers {
            return Err(Error::Other(format!(
                "qwen3.5: received {} weight layers, expected {}",
                weights.layers.len(),
                config.text.n_layers
            )));
        }

        // CPU weights stay where they are. In particular, do not call
        // CpuBackend::to_device, which would deep-clone every tensor and double
        // peak memory on a 16-GB Mac.
        #[cfg(feature = "metal-experiments")]
        let metal_w8_body = metal_w8_body_layers
            .map(|layer_indices| Qwen35MetalW8Body::pack(&weights, &config.text, &layer_indices))
            .transpose()?;
        #[cfg(feature = "metal-w8-head-mlp")]
        let metal_w8_mlp_blocks = metal_w8_mlp_block_layers
            .map(|layer_indices| {
                Qwen35MetalW8MlpBlocks::pack(&weights, &config.text, &layer_indices)
            })
            .transpose()?;
        #[cfg(not(feature = "metal-experiments"))]
        if metal_w8_body_layers.is_some() {
            return Err(Error::Other(
                "qwen3.5 Metal W8 body requires the `metal-w8` build feature".into(),
            ));
        }
        #[cfg(not(feature = "metal-w8-head-mlp"))]
        if metal_w8_mlp_block_layers.is_some() {
            return Err(Error::Other(
                "qwen3.5 Metal W8 MLP block requires the `metal-w8` build feature".into(),
            ));
        }
        let weights = RuntimeWeights::pack(weights, &config.text)?;
        #[cfg(feature = "metal-w8-head-mlp")]
        let metal_w8_lm_head = if metal_w8_lm_head {
            if !config.text.tie_word_embeddings || weights.lm_head.is_some() {
                return Err(Error::Other(
                    "qwen3.5 Metal W8 lm_head requires a checkpoint with tied word embeddings"
                        .into(),
                ));
            }
            let embedding = weights.token_embedding.as_f32()?;
            Some(
                apxinf_metal::MetalW8LmHead::from_f32_rows(
                    embedding,
                    config.text.vocab_size,
                    config.text.hidden_size,
                )
                .map_err(|error| Error::Other(format!("qwen3.5 Metal W8 lm_head: {error}")))?,
            )
        } else {
            None
        };
        #[cfg(feature = "metal-w8-head-mlp")]
        let metal_w8_lm_head_stats = metal_w8_lm_head
            .as_ref()
            .map(|_| Qwen35MetalW8LmHeadStats::default());
        #[cfg(not(feature = "metal-w8-head-mlp"))]
        if metal_w8_lm_head {
            return Err(Error::Other(
                "qwen3.5 Metal W8 lm_head requires the `metal-w8` build feature".into(),
            ));
        }
        let state = Qwen35HybridState::new(&config.text, &*backend, max_context)?;
        Ok(Self {
            config,
            weights,
            backend,
            state,
            #[cfg(feature = "metal-w8-head-mlp")]
            metal_w8_lm_head,
            #[cfg(feature = "metal-w8-head-mlp")]
            metal_w8_lm_head_stats,
            #[cfg(feature = "metal-experiments")]
            metal_w8_body,
            #[cfg(feature = "metal-w8-head-mlp")]
            metal_w8_mlp_blocks,
            #[cfg(feature = "metal-experiments")]
            metal_w8_gdn: None,
            #[cfg(feature = "metal-experiments")]
            metal_w8_linear_layer: None,
            #[cfg(feature = "metal-experiments")]
            metal_w8_all_linear_layers_precision_v2: None,
            #[cfg(feature = "metal-experiments")]
            metal_w8_linear_layer_stacks_v1: None,
            #[cfg(feature = "metal-experiments")]
            metal_w8_mlp_stack3_boundary_body_v1: None,
            #[cfg(feature = "metal-experiments")]
            metal_w8_mlp_stack3_boundary_tail_head_v1: None,
            #[cfg(feature = "metal-experiments")]
            metal_w8_stack3_lm_head_v2_terminal_error: None,
            #[cfg(feature = "metal-experiments")]
            packed_w8_linear_layer_reference: None,
            #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
            fail_after_layer_once_for_test: None,
            #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
            fail_stack3_lm_head_v2_before_submit_once_for_test: false,
            #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
            fail_mlp_stack3_boundary_final_mlp_after_submit_once_for_test: false,
            #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
            boundary_tail_head_fault_once_for_test: None,
            #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
            fail_boundary_tail_head_rerank_once_for_test: false,
        })
    }

    pub fn config_ref(&self) -> &Qwen35Config {
        &self.config
    }

    pub fn state_ref(&self) -> &Qwen35HybridState {
        &self.state
    }

    pub fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_body_stats(&self) -> Option<Qwen35MetalW8BodyStats> {
        let mut stats = self.metal_w8_body_layer_stats().into_iter();
        let only = stats.next()?;
        stats.next().is_none().then_some(only)
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_body_layer_stats(&self) -> Vec<Qwen35MetalW8BodyStats> {
        self.metal_w8_body
            .as_ref()
            .map(Qwen35MetalW8Body::stats)
            .unwrap_or_default()
    }

    #[cfg(feature = "metal-w8-head-mlp")]
    pub fn metal_w8_mlp_block_stats(&self) -> Option<Qwen35MetalW8MlpBlockStats> {
        let mut stats = self.metal_w8_mlp_block_layer_stats().into_iter();
        let only = stats.next()?;
        stats.next().is_none().then_some(only)
    }

    #[cfg(feature = "metal-w8-head-mlp")]
    pub fn metal_w8_mlp_block_layer_stats(&self) -> Vec<Qwen35MetalW8MlpBlockStats> {
        self.metal_w8_mlp_blocks
            .as_ref()
            .map(Qwen35MetalW8MlpBlocks::stats)
            .unwrap_or_default()
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_gdn_stats(&self) -> Option<Qwen35MetalW8GdnStats> {
        self.metal_w8_gdn.as_ref().map(Qwen35MetalW8GdnLayer::stats)
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_linear_layer_stats(&self) -> Option<Qwen35MetalW8LinearLayerStats> {
        self.metal_w8_linear_layer
            .as_ref()
            .map(Qwen35MetalW8LinearLayer::stats)
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_linear_layer_precision_v2_stats(
        &self,
    ) -> Option<Qwen35MetalW8LinearLayerPrecisionV2Stats> {
        self.metal_w8_linear_layer
            .as_ref()
            .and_then(Qwen35MetalW8LinearLayer::precision_v2_stats)
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_linear_layer_buffer_ledger(
        &self,
    ) -> Option<apxinf_metal::LinearLayerBufferLedger> {
        self.metal_w8_linear_layer
            .as_ref()
            .map(|layer| layer.block.buffer_ledger())
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_all_linear_layers_precision_v2_stats(
        &self,
    ) -> Option<Qwen35MetalW8AllLinearLayersPrecisionV2Stats> {
        self.metal_w8_all_linear_layers_precision_v2
            .as_ref()
            .map(|lane| lane.stats(self.metal_w8_mlp_block_layer_stats()))
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_linear_layer_stacks_v1_stats(
        &self,
    ) -> Option<Qwen35MetalW8LinearLayerStacksV1Stats> {
        self.metal_w8_linear_layer_stacks_v1
            .as_ref()
            .map(|lane| lane.stats(self.metal_w8_mlp_block_layer_stats()))
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_mlp_stack3_boundary_body_v1_stats(
        &self,
    ) -> Option<Qwen35MetalW8MlpStack3BoundaryBodyV1Stats> {
        self.metal_w8_mlp_stack3_boundary_body_v1
            .as_ref()
            .map(Qwen35MetalW8MlpStack3BoundaryBodyV1::stats)
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_mlp_stack3_boundary_tail_head_v1_stats(
        &self,
    ) -> Option<Qwen35MetalW8MlpStack3BoundaryTailHeadV1Stats> {
        self.metal_w8_mlp_stack3_boundary_tail_head_v1
            .as_ref()
            .map(Qwen35MetalW8MlpStack3BoundaryTailHeadV1::stats)
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_mlp_stack3_boundary_tail_head_v1_aggregate_ledger(
        &self,
    ) -> Option<Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger> {
        self.metal_w8_mlp_stack3_boundary_tail_head_v1
            .as_ref()
            .and_then(|lane| lane.aggregate_ledger().ok())
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_mlp_stack3_boundary_body_v1_aggregate_ledger(
        &self,
    ) -> Option<Qwen35MetalW8MlpStack3BoundaryBodyV1AggregateLedger> {
        self.metal_w8_mlp_stack3_boundary_body_v1
            .as_ref()
            .and_then(|body| body.aggregate_ledger().ok())
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_linear_layer_stack3_v1_buffer_ledgers(
        &self,
    ) -> Option<Vec<Qwen35MetalW8LinearLayerStack3BufferLedger>> {
        self.metal_w8_linear_layer_stacks_v1
            .as_ref()
            .map(Qwen35MetalW8LinearLayerStacksV1::buffer_ledgers)
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_linear_layer_stacks_v1_aggregate_ledger(
        &self,
    ) -> Option<Qwen35MetalW8LinearLayerStacksV1AggregateLedger> {
        let stacks = self
            .metal_w8_linear_layer_stacks_v1
            .as_ref()?
            .buffer_ledgers();
        let full_attention_mlp_layers = self
            .metal_w8_mlp_blocks
            .as_ref()
            .map(Qwen35MetalW8MlpBlocks::buffer_ledgers)
            .unwrap_or_default();
        Some(Qwen35MetalW8LinearLayerStacksV1AggregateLedger::new(
            stacks,
            full_attention_mlp_layers,
        ))
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_stack3_lm_head_v2_aggregate_ledger(
        &self,
    ) -> Option<Qwen35MetalW8Stack3LmHeadV2AggregateLedger> {
        let body = self.metal_w8_linear_layer_stacks_v1_aggregate_ledger()?;
        let lm_head = self.metal_w8_lm_head.as_ref()?.buffer_ledger();
        Qwen35MetalW8Stack3LmHeadV2AggregateLedger::new(body, lm_head).ok()
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_all_linear_layers_precision_v2_buffer_ledgers(
        &self,
    ) -> Option<Vec<Qwen35MetalW8LinearLayerBufferLedger>> {
        self.metal_w8_all_linear_layers_precision_v2
            .as_ref()
            .map(Qwen35MetalW8AllLinearLayersPrecisionV2::buffer_ledgers)
    }

    #[cfg(feature = "metal-experiments")]
    pub fn metal_w8_all_linear_layers_precision_v2_aggregate_ledger(
        &self,
    ) -> Option<Qwen35MetalW8AllLinearLayersPrecisionV2AggregateLedger> {
        let linear_layers = self
            .metal_w8_all_linear_layers_precision_v2
            .as_ref()?
            .buffer_ledgers();
        let full_attention_mlp_layers = self
            .metal_w8_mlp_blocks
            .as_ref()
            .map(Qwen35MetalW8MlpBlocks::buffer_ledgers)
            .unwrap_or_default();
        Some(Qwen35MetalW8AllLinearLayersPrecisionV2AggregateLedger::new(
            linear_layers,
            full_attention_mlp_layers,
        ))
    }

    #[cfg(feature = "metal-experiments")]
    pub fn packed_w8_linear_layer_reference_stats(
        &self,
    ) -> Option<Qwen35PackedW8LinearLayerReferenceStats> {
        self.packed_w8_linear_layer_reference
            .as_ref()
            .map(Qwen35PackedW8LinearLayerReference::stats)
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_metal_w8_gdn_failure_once_for_test(&mut self) {
        self.metal_w8_gdn
            .as_mut()
            .expect("test requires the explicit Metal W8 GDN constructor")
            .fail_next_decode_after_scratch = true;
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_metal_w8_linear_layer_failure_once_for_test(&mut self) {
        self.metal_w8_linear_layer
            .as_mut()
            .expect("test requires the explicit Metal W8 linear-layer constructor")
            .fail_next_decode_after_scratch = true;
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_metal_w8_stack3_failure_once_for_test(&mut self, stack_index: usize) {
        self.metal_w8_linear_layer_stacks_v1
            .as_mut()
            .expect("test requires the explicit Metal W8 stack3-v1 constructor")
            .stacks
            .get_mut(stack_index)
            .expect("test stack3-v1 index must exist")
            .fail_next_decode_after_scratch = true;
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_metal_w8_mlp_stack3_boundary_failure_once_for_test(&mut self, boundary_index: usize) {
        self.metal_w8_mlp_stack3_boundary_body_v1
            .as_mut()
            .expect("test requires the explicit MLP→Stack3 boundary body v1 constructor")
            .boundaries
            .get_mut(boundary_index)
            .expect("test MLP→Stack3 boundary body v1 index must exist")
            .fail_next_decode_after_scratch = true;
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_metal_w8_mlp_stack3_boundary_initial_failure_once_for_test(&mut self) {
        self.metal_w8_mlp_stack3_boundary_body_v1
            .as_mut()
            .expect("test requires the explicit MLP→Stack3 boundary body v1 constructor")
            .initial_stack
            .fail_next_decode_after_scratch = true;
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_metal_w8_boundary_tail_head_initial_failure_once_for_test(&mut self) {
        self.metal_w8_mlp_stack3_boundary_tail_head_v1
            .as_mut()
            .expect("test requires the explicit boundary + tail-head v1 constructor")
            .initial_stack
            .fail_next_decode_after_scratch = true;
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_metal_w8_boundary_tail_head_tail_post_execution_failure_once_for_test(&mut self) {
        assert!(
            self.metal_w8_mlp_stack3_boundary_tail_head_v1.is_some(),
            "test requires the explicit boundary + tail-head v1 constructor"
        );
        self.boundary_tail_head_fault_once_for_test =
            Some(Qwen35BoundaryTailHeadFaultV1ForTest::TailPostExecution);
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_metal_w8_boundary_tail_head_malformed_once_for_test(
        &mut self,
        fault: Qwen35BoundaryTailHeadFaultV1ForTest,
    ) {
        assert!(
            matches!(
                fault,
                Qwen35BoundaryTailHeadFaultV1ForTest::TailNonFiniteOutput
                    | Qwen35BoundaryTailHeadFaultV1ForTest::TailDuplicateCandidate
                    | Qwen35BoundaryTailHeadFaultV1ForTest::TailOutOfRangeCandidate
            ),
            "malformed tail test hook requires a malformed-output mode"
        );
        assert!(
            self.metal_w8_mlp_stack3_boundary_tail_head_v1.is_some(),
            "test requires the explicit boundary + tail-head v1 constructor"
        );
        self.boundary_tail_head_fault_once_for_test = Some(fault);
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_metal_w8_boundary_tail_head_rerank_failure_once_for_test(&mut self) {
        assert!(
            self.metal_w8_mlp_stack3_boundary_tail_head_v1.is_some(),
            "test requires the explicit boundary + tail-head v1 constructor"
        );
        self.fail_boundary_tail_head_rerank_once_for_test = true;
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_metal_w8_mlp_stack3_boundary_final_mlp_failure_once_for_test(&mut self) {
        assert!(
            self.metal_w8_mlp_stack3_boundary_body_v1.is_some(),
            "test requires the explicit MLP→Stack3 boundary body v1 constructor"
        );
        self.fail_mlp_stack3_boundary_final_mlp_after_submit_once_for_test = true;
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_metal_w8_stack3_lm_head_failure_once_for_test(&mut self) {
        assert!(
            self.metal_w8_stack3_lm_head_v2_terminal_error.is_some(),
            "test requires the explicit Stack3 + lm_head v2 constructor"
        );
        self.fail_stack3_lm_head_v2_before_submit_once_for_test = true;
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_packed_w8_linear_layer_reference_failure_once_for_test(&mut self) {
        self.packed_w8_linear_layer_reference
            .as_mut()
            .expect("test requires the explicit packed W8 linear-layer reference constructor")
            .fail_next_decode_after_reference = true;
    }

    #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
    fn inject_failure_after_layer_once_for_test(&mut self, layer_index: usize) {
        assert!(
            layer_index < self.config.text.n_layers,
            "test fault layer must be inside the model body"
        );
        self.fail_after_layer_once_for_test = Some(layer_index);
    }

    #[cfg(feature = "metal-experiments")]
    fn ensure_complete_linear_layer_lane_is_not_terminal(&self) -> Result<()> {
        if self.metal_w8_stack3_lm_head_v2_terminal_error == Some(true) {
            return Err(Error::Other(
                "qwen3.5 Metal W8 Stack3 + lm_head v2 lane is terminal after a post-body error; reset required"
                    .into(),
            ));
        }
        if let Some(layer) = self
            .metal_w8_linear_layer
            .as_ref()
            .filter(|layer| layer.terminal_error)
        {
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 linear layer {} is terminal after a decode error; reset required",
                layer.layer_index
            )));
        }
        if let Some(reference) = self
            .packed_w8_linear_layer_reference
            .as_ref()
            .filter(|reference| reference.terminal_error)
        {
            return Err(Error::Other(format!(
                "qwen3.5 packed W8 linear-layer reference {} is terminal after a decode error; reset required",
                reference.layer_index
            )));
        }
        if self
            .metal_w8_all_linear_layers_precision_v2
            .as_ref()
            .is_some_and(Qwen35MetalW8AllLinearLayersPrecisionV2::is_terminal)
        {
            return Err(Error::Other(
                "qwen3.5 Metal W8 all-linear-layers precision-v2 lane is terminal after a decode error; reset required"
                    .into(),
            ));
        }
        if self
            .metal_w8_linear_layer_stacks_v1
            .as_ref()
            .is_some_and(Qwen35MetalW8LinearLayerStacksV1::is_terminal)
        {
            return Err(Error::Other(
                "qwen3.5 Metal W8 stack3-v1 lane is terminal after a decode error; reset required"
                    .into(),
            ));
        }
        if self
            .metal_w8_mlp_stack3_boundary_body_v1
            .as_ref()
            .is_some_and(Qwen35MetalW8MlpStack3BoundaryBodyV1::is_terminal)
        {
            return Err(Error::Other(
                "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 lane is terminal after an error; reset required"
                    .into(),
            ));
        }
        if self
            .metal_w8_mlp_stack3_boundary_tail_head_v1
            .as_ref()
            .is_some_and(Qwen35MetalW8MlpStack3BoundaryTailHeadV1::is_terminal)
        {
            return Err(Error::Other(
                "qwen3.5 Metal W8 boundary + tail-head v1 lane is terminal after an error; reset required"
                    .into(),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "metal-experiments")]
    fn complete_linear_layer_lane_versions(&self) -> (Option<u64>, Option<u64>) {
        (
            self.metal_w8_linear_layer
                .as_ref()
                .map(|layer| layer.block.stats().committed_state_version),
            self.packed_w8_linear_layer_reference
                .as_ref()
                .map(|reference| reference.committed_state_version),
        )
    }

    #[cfg(feature = "metal-experiments")]
    fn latch_complete_linear_layer_lanes_after_partial_commit(
        &mut self,
        versions_before: (Option<u64>, Option<u64>),
    ) -> bool {
        let metal_advanced = self
            .metal_w8_linear_layer
            .as_ref()
            .zip(versions_before.0)
            .is_some_and(|(layer, before)| layer.block.stats().committed_state_version > before);
        let packed_advanced = self
            .packed_w8_linear_layer_reference
            .as_ref()
            .zip(versions_before.1)
            .is_some_and(|(reference, before)| reference.committed_state_version > before);
        // CPU recurrent/KV state and Metal seeds are applied layer by layer
        // during prefill, while Metal decode state commits layer by layer.
        // There is no practical cross-24-layer rollback. Once an all-linear
        // body starts, any later body error therefore makes the whole lane
        // terminal, even when every Metal committed version is still zero.
        let all_linear_requires_reset = self.metal_w8_all_linear_layers_precision_v2.is_some();
        let stack3_requires_reset = self.metal_w8_linear_layer_stacks_v1.is_some();
        let mlp_stack3_boundary_requires_reset =
            self.metal_w8_mlp_stack3_boundary_body_v1.is_some();
        let boundary_tail_head_requires_reset =
            self.metal_w8_mlp_stack3_boundary_tail_head_v1.is_some();
        if metal_advanced {
            self.metal_w8_linear_layer
                .as_mut()
                .expect("advanced Metal W8 linear layer must be present")
                .terminal_error = true;
        }
        if packed_advanced {
            self.packed_w8_linear_layer_reference
                .as_mut()
                .expect("advanced packed W8 linear-layer reference must be present")
                .terminal_error = true;
        }
        if all_linear_requires_reset {
            self.metal_w8_all_linear_layers_precision_v2
                .as_mut()
                .expect("failed all-linear precision-v2 lane must be present")
                .latch_terminal();
        }
        if stack3_requires_reset {
            self.metal_w8_linear_layer_stacks_v1
                .as_mut()
                .expect("failed stack3-v1 lane must be present")
                .latch_terminal();
            if self.metal_w8_stack3_lm_head_v2_terminal_error.is_some() {
                self.metal_w8_stack3_lm_head_v2_terminal_error = Some(true);
            }
        }
        if mlp_stack3_boundary_requires_reset {
            self.metal_w8_mlp_stack3_boundary_body_v1
                .as_mut()
                .expect("failed MLP→Stack3 boundary body v1 lane must be present")
                .latch_terminal();
        }
        if boundary_tail_head_requires_reset {
            self.metal_w8_mlp_stack3_boundary_tail_head_v1
                .as_mut()
                .expect("failed boundary + tail-head v1 lane must be present")
                .latch_terminal();
        }
        metal_advanced
            || packed_advanced
            || all_linear_requires_reset
            || stack3_requires_reset
            || mlp_stack3_boundary_requires_reset
            || boundary_tail_head_requires_reset
    }

    #[cfg(feature = "metal-experiments")]
    fn finish_stack3_lm_head_v2_post_body<T>(
        &mut self,
        result: Result<T>,
        stage: &str,
    ) -> Result<T> {
        match result {
            Err(error) if self.metal_w8_stack3_lm_head_v2_terminal_error.is_some() => {
                self.metal_w8_stack3_lm_head_v2_terminal_error = Some(true);
                self.metal_w8_linear_layer_stacks_v1
                    .as_mut()
                    .expect("Stack3 + lm_head v2 marker requires the Stack3 lane")
                    .latch_terminal();
                Err(Error::Other(format!(
                    "qwen3.5 Metal W8 Stack3 + lm_head v2 {stage} failed after body state advancement: {error}; entire lane is terminal, reset required"
                )))
            }
            other => other,
        }
    }

    #[cfg(feature = "metal-experiments")]
    fn maybe_fail_stack3_lm_head_v2_before_submit_for_test(&mut self) -> Result<()> {
        #[cfg(all(test, debug_assertions))]
        if self.fail_stack3_lm_head_v2_before_submit_once_for_test {
            self.fail_stack3_lm_head_v2_before_submit_once_for_test = false;
            return Err(Error::Other(
                "qwen3.5 test fault injected after body commit and before lm_head submission"
                    .into(),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "metal-experiments")]
    fn finish_boundary_tail_head_v1_post_body<T>(
        &mut self,
        result: Result<T>,
        stage: &str,
    ) -> Result<T> {
        match result {
            Err(error) if self.metal_w8_mlp_stack3_boundary_tail_head_v1.is_some() => {
                self.metal_w8_mlp_stack3_boundary_tail_head_v1
                    .as_mut()
                    .expect("boundary + tail-head v1 lane must be present")
                    .latch_terminal();
                Err(Error::Other(format!(
                    "qwen3.5 Metal W8 boundary + tail-head v1 {stage} failed after body state advancement: {error}; entire lane is terminal, reset required"
                )))
            }
            other => other,
        }
    }

    #[cfg(feature = "metal-experiments")]
    fn run_boundary_tail_head_v1(
        &mut self,
        full_attention_residual: &Tensor,
    ) -> Result<Qwen35BoundaryTailHeadOutputV1> {
        let expected = [1, self.config.text.hidden_size];
        if full_attention_residual.shape().dims() != expected {
            return Err(Error::ShapeMismatch {
                expected: format!("[1, {}]", self.config.text.hidden_size),
                got: full_attention_residual.shape().to_string(),
            });
        }
        let input = full_attention_residual.as_f32()?;
        #[cfg(all(test, debug_assertions))]
        if let Some(fault) = self.boundary_tail_head_fault_once_for_test.take() {
            let tail = &mut self
                .metal_w8_mlp_stack3_boundary_tail_head_v1
                .as_mut()
                .expect("boundary + tail-head v1 presence checked before body advancement")
                .tail;
            let injected = match fault {
                Qwen35BoundaryTailHeadFaultV1ForTest::TailPostExecution => {
                    tail.inject_failure_after_gpu_execution_for_testing(input)
                }
                Qwen35BoundaryTailHeadFaultV1ForTest::TailNonFiniteOutput => {
                    tail.inject_nonfinite_normalized_output_for_testing(input)
                }
                Qwen35BoundaryTailHeadFaultV1ForTest::TailDuplicateCandidate => {
                    tail.inject_duplicate_candidate_output_for_testing(input)
                }
                Qwen35BoundaryTailHeadFaultV1ForTest::TailOutOfRangeCandidate => {
                    tail.inject_out_of_range_candidate_output_for_testing(input)
                }
            };
            return Err(match injected {
                Err(error) => Error::Other(format!(
                    "qwen3.5 Metal W8 tail-head v1 injected transaction failed: {error}"
                )),
                Ok(()) => Error::Other(
                    "qwen3.5 Metal W8 tail-head v1 test fault unexpectedly succeeded".into(),
                ),
            });
        }
        let started = std::time::Instant::now();
        let (normalized_hidden, candidates) = {
            let output = self
                .metal_w8_mlp_stack3_boundary_tail_head_v1
                .as_mut()
                .expect("boundary + tail-head v1 presence checked before body advancement")
                .tail
                .decode(input)
                .map_err(|error| {
                    Error::Other(format!(
                        "qwen3.5 Metal W8 tail-head v1 transaction failed: {error}"
                    ))
                })?;
            (
                output.normalized_hidden.to_vec(),
                output.candidate_token_ids,
            )
        };
        Ok(Qwen35BoundaryTailHeadOutputV1 {
            normalized_hidden: Tensor::from_f32(
                vec![1, self.config.text.hidden_size],
                &normalized_hidden,
            )?,
            candidates,
            tail_elapsed_ns: started.elapsed().as_nanos(),
        })
    }

    #[cfg(feature = "metal-experiments")]
    fn rerank_boundary_tail_head_v1(
        &mut self,
        normalized_hidden: &Tensor,
        candidates: [u32; apxinf_metal::W8_TOP_K],
    ) -> Result<(u32, u128)> {
        #[cfg(all(test, debug_assertions))]
        if std::mem::take(&mut self.fail_boundary_tail_head_rerank_once_for_test) {
            return Err(Error::Other(
                "qwen3.5 test fault injected after tail output and before F32 rerank".into(),
            ));
        }
        let started = std::time::Instant::now();
        let token = rerank_tied_f32_candidates(
            self.weights.token_embedding.as_f32()?,
            normalized_hidden.as_f32()?,
            self.config.text.vocab_size,
            self.config.text.hidden_size,
            candidates,
        )?;
        let elapsed = started.elapsed().as_nanos();
        let lane = self
            .metal_w8_mlp_stack3_boundary_tail_head_v1
            .as_mut()
            .expect("boundary + tail-head v1 lane must be present");
        lane.rerank_elapsed_ns = lane.rerank_elapsed_ns.saturating_add(elapsed);
        Ok((token, elapsed))
    }

    #[cfg(feature = "metal-w8-head-mlp")]
    pub fn metal_w8_lm_head_stats(&self) -> Option<Qwen35MetalW8LmHeadStats> {
        self.metal_w8_lm_head_stats
    }

    fn forward_layer(&mut self, x: &Tensor, layer_index: usize, start_pos: u32) -> Result<Tensor> {
        let text = &self.config.text;
        let backend = &*self.backend;
        let layer = &self.weights.layers[layer_index];
        #[cfg(feature = "metal-experiments")]
        if self
            .packed_w8_linear_layer_reference
            .as_ref()
            .is_some_and(|reference| reference.layer_index == layer_index)
        {
            let decode = start_pos > 0 && x.shape().dims() == [1, text.hidden_size];
            if decode {
                return run_linear_layer_with_packed_w8_reference(
                    x,
                    self.packed_w8_linear_layer_reference
                        .as_mut()
                        .expect("selected packed W8 linear-layer reference must be present"),
                );
            }
            if start_pos > 0
                && self
                    .packed_w8_linear_layer_reference
                    .as_ref()
                    .is_some_and(|reference| reference.state.is_some() || reference.terminal_error)
            {
                return Err(Error::Other(format!(
                    "qwen3.5 packed W8 linear-layer reference {layer_index} owns decode state; only single-token decode is valid until reset"
                )));
            }
        }
        #[cfg(feature = "metal-experiments")]
        if self
            .metal_w8_linear_layer
            .as_ref()
            .is_some_and(|metal| metal.layer_index == layer_index)
        {
            let decode = start_pos > 0 && x.shape().dims() == [1, text.hidden_size];
            if decode {
                return run_linear_layer_with_metal_w8(
                    x,
                    self.metal_w8_linear_layer
                        .as_mut()
                        .expect("selected Metal W8 linear layer must be present"),
                );
            }
            if start_pos > 0
                && self
                    .metal_w8_linear_layer
                    .as_ref()
                    .is_some_and(|metal| metal.seeded || metal.terminal_error)
            {
                return Err(Error::Other(format!(
                    "qwen3.5 Metal W8 linear layer {layer_index} owns decode state; only single-token decode is valid until reset"
                )));
            }
        }
        #[cfg(feature = "metal-experiments")]
        if self
            .metal_w8_all_linear_layers_precision_v2
            .as_ref()
            .and_then(|lane| lane.layer(layer_index))
            .is_some()
        {
            let decode = start_pos > 0 && x.shape().dims() == [1, text.hidden_size];
            if decode {
                return run_linear_layer_with_metal_w8(
                    x,
                    self.metal_w8_all_linear_layers_precision_v2
                        .as_mut()
                        .and_then(|lane| lane.layer_mut(layer_index))
                        .expect("selected all-linear precision-v2 layer must be present"),
                );
            }
            if start_pos > 0
                && self
                    .metal_w8_all_linear_layers_precision_v2
                    .as_ref()
                    .and_then(|lane| lane.layer(layer_index))
                    .is_some_and(|metal| metal.seeded || metal.terminal_error)
            {
                return Err(Error::Other(format!(
                    "qwen3.5 Metal W8 all-linear precision-v2 layer {layer_index} owns decode state; only single-token decode is valid until reset"
                )));
            }
        }
        let normed =
            backend.rms_norm_offset(x, &layer.input_norm_weight, text.rms_norm_eps, 1.0)?;

        let attention = match (&layer.attention, text.layer_types[layer_index]) {
            (RuntimeAttentionWeights::Linear(weights), Qwen35LayerType::LinearAttention) => {
                #[cfg(feature = "metal-experiments")]
                {
                    let selected = self
                        .metal_w8_gdn
                        .as_ref()
                        .is_some_and(|gdn| gdn.layer_index == layer_index);
                    let selected_linear_layer = self
                        .metal_w8_linear_layer
                        .as_ref()
                        .is_some_and(|metal| metal.layer_index == layer_index);
                    let selected_packed_reference = self
                        .packed_w8_linear_layer_reference
                        .as_ref()
                        .is_some_and(|reference| reference.layer_index == layer_index);
                    let selected_all_linear_layer = self
                        .metal_w8_all_linear_layers_precision_v2
                        .as_ref()
                        .and_then(|lane| lane.layer(layer_index))
                        .is_some();
                    let selected_stack3_layer = self
                        .metal_w8_linear_layer_stacks_v1
                        .as_ref()
                        .and_then(|lane| lane.selected_slot(layer_index))
                        .is_some();
                    let selected_mlp_stack3_boundary_layer = self
                        .metal_w8_mlp_stack3_boundary_body_v1
                        .as_ref()
                        .and_then(|lane| lane.selected_linear_slot(layer_index))
                        .is_some();
                    let selected_boundary_tail_head_layer = self
                        .metal_w8_mlp_stack3_boundary_tail_head_v1
                        .as_ref()
                        .and_then(|lane| lane.selected_linear_slot(layer_index))
                        .is_some();
                    let decode =
                        selected && start_pos > 0 && normed.shape().dims() == [1, text.hidden_size];
                    if decode {
                        let gdn = self
                            .metal_w8_gdn
                            .as_mut()
                            .expect("selected Metal W8 GDN layer must be present");
                        run_linear_attention_with_metal_w8_gdn(&normed, gdn)?
                    } else {
                        if selected
                            && self
                                .metal_w8_gdn
                                .as_ref()
                                .is_some_and(Qwen35MetalW8GdnLayer::owns_recurrent_state)
                        {
                            return Err(Error::Other(format!(
                                "qwen3.5 Metal W8 GDN layer {layer_index} owns decode state; only single-token decode is valid until reset"
                            )));
                        }
                        let (attention, seed) = {
                            let state = self.state.linear_state_mut(layer_index)?;
                            let attention =
                                run_linear_attention(backend, text, &normed, weights, state)?;
                            let seed = (selected
                                || selected_linear_layer
                                || selected_packed_reference
                                || selected_all_linear_layer
                                || selected_stack3_layer
                                || selected_mlp_stack3_boundary_layer
                                || selected_boundary_tail_head_layer)
                                .then(|| gdn_decode_state_from_cpu(text, state))
                                .transpose()?;
                            (attention, seed)
                        };
                        if let Some(seed) = seed.as_ref() {
                            if selected {
                                self.metal_w8_gdn
                                    .as_mut()
                                    .expect("selected Metal W8 GDN layer must be present")
                                    .seed_after_cpu_prefill(seed)?;
                            }
                            if selected_linear_layer {
                                self.metal_w8_linear_layer
                                    .as_mut()
                                    .expect("selected Metal W8 linear layer must be present")
                                    .seed_after_cpu_prefill(seed)?;
                            }
                            if selected_packed_reference {
                                self.packed_w8_linear_layer_reference
                                    .as_mut()
                                    .expect(
                                        "selected packed W8 linear-layer reference must be present",
                                    )
                                    .seed_after_cpu_prefill(seed);
                            }
                            if selected_all_linear_layer {
                                self.metal_w8_all_linear_layers_precision_v2
                                    .as_mut()
                                    .and_then(|lane| lane.layer_mut(layer_index))
                                    .expect(
                                        "selected all-linear precision-v2 layer must be present",
                                    )
                                    .seed_after_cpu_prefill(seed)?;
                            }
                            if selected_stack3_layer {
                                self.metal_w8_linear_layer_stacks_v1
                                    .as_mut()
                                    .expect("selected stack3-v1 lane must be present")
                                    .seed_layer_after_cpu_prefill(layer_index, seed)?;
                            }
                            if selected_mlp_stack3_boundary_layer {
                                self.metal_w8_mlp_stack3_boundary_body_v1
                                    .as_mut()
                                    .expect(
                                        "selected MLP→Stack3 boundary body v1 lane must be present",
                                    )
                                    .seed_layer_after_cpu_prefill(layer_index, seed)?;
                            }
                            if selected_boundary_tail_head_layer {
                                self.metal_w8_mlp_stack3_boundary_tail_head_v1
                                    .as_mut()
                                    .expect("selected boundary + tail-head v1 lane must be present")
                                    .seed_layer_after_cpu_prefill(layer_index, seed)?;
                            }
                        }
                        attention
                    }
                }
                #[cfg(not(feature = "metal-experiments"))]
                {
                    let state = self.state.linear_state_mut(layer_index)?;
                    run_linear_attention(backend, text, &normed, weights, state)?
                }
            }
            (RuntimeAttentionWeights::Full(weights), Qwen35LayerType::FullAttention) => {
                let cache_index = self.state.full_cache_index(layer_index)?;
                let max_context = self.state.max_context();
                run_full_attention(
                    backend,
                    text,
                    &normed,
                    weights,
                    &mut *self.state.kv,
                    cache_index,
                    start_pos,
                    max_context,
                )?
            }
            _ => {
                return Err(Error::Other(format!(
                    "qwen3.5: runtime/checkpoint layer kind mismatch at layer {layer_index}"
                )));
            }
        };
        let residual = backend.add(x, &attention)?;
        let normed = backend.rms_norm_offset(
            &residual,
            &layer.post_attention_norm_weight,
            text.rms_norm_eps,
            1.0,
        )?;
        #[cfg(feature = "metal-w8-head-mlp")]
        let mlp = if start_pos > 0 && normed.shape().dims()[0] == 1 {
            if let Some(block) = self
                .metal_w8_mlp_blocks
                .as_mut()
                .and_then(|blocks| blocks.layer_mut(layer_index))
            {
                run_mlp_with_metal_w8_block(&normed, block)?
            } else {
                #[cfg(feature = "metal-experiments")]
                {
                    if let Some(body) = self
                        .metal_w8_body
                        .as_mut()
                        .and_then(|body| body.layer_mut(layer_index))
                    {
                        run_mlp_with_metal_w8(backend, &normed, &layer.mlp, body)?
                    } else {
                        run_mlp(backend, &normed, &layer.mlp)?
                    }
                }
                #[cfg(not(feature = "metal-experiments"))]
                run_mlp(backend, &normed, &layer.mlp)?
            }
        } else {
            run_mlp(backend, &normed, &layer.mlp)?
        };
        #[cfg(not(feature = "metal-w8-head-mlp"))]
        let mlp = run_mlp(backend, &normed, &layer.mlp)?;
        backend.add(&residual, &mlp)
    }

    #[cfg(feature = "metal-experiments")]
    fn forward_full_attention_residual_for_mlp_stack3_boundary_body_v1(
        &mut self,
        x: &Tensor,
        layer_index: usize,
        start_pos: u32,
    ) -> Result<Tensor> {
        let text = &self.config.text;
        let layer = self.weights.layers.get(layer_index).ok_or_else(|| {
            Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 layer {layer_index} is outside the runtime body"
            ))
        })?;
        let RuntimeAttentionWeights::Full(attention_weights) = &layer.attention else {
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 expected full attention at layer {layer_index}"
            )));
        };
        if text.layer_types.get(layer_index) != Some(&Qwen35LayerType::FullAttention) {
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 schedule changed at full-attention layer {layer_index}"
            )));
        }
        let normed =
            self.backend
                .rms_norm_offset(x, &layer.input_norm_weight, text.rms_norm_eps, 1.0)?;
        let cache_index = self.state.full_cache_index(layer_index)?;
        let max_context = self.state.max_context();
        let attention = run_full_attention(
            &*self.backend,
            text,
            &normed,
            attention_weights,
            &mut *self.state.kv,
            cache_index,
            start_pos,
            max_context,
        )?;
        self.backend.add(x, &attention)
    }

    #[cfg(feature = "metal-experiments")]
    fn forward_final_full_attention_mlp_for_mlp_stack3_boundary_body_v1(
        &mut self,
        x: &Tensor,
        layer_index: usize,
        start_pos: u32,
    ) -> Result<Tensor> {
        let residual = self.forward_full_attention_residual_for_mlp_stack3_boundary_body_v1(
            x,
            layer_index,
            start_pos,
        )?;
        let normed = self.backend.rms_norm_offset(
            &residual,
            &self.weights.layers[layer_index].post_attention_norm_weight,
            self.config.text.rms_norm_eps,
            1.0,
        )?;
        let mlp = run_mlp_with_metal_w8_block(
            &normed,
            &mut self
                .metal_w8_mlp_stack3_boundary_body_v1
                .as_mut()
                .expect("MLP→Stack3 boundary body v1 lane must be present")
                .final_mlp,
        )?;
        #[cfg(all(test, debug_assertions))]
        if std::mem::take(&mut self.fail_mlp_stack3_boundary_final_mlp_after_submit_once_for_test) {
            return Err(Error::Other(
                "qwen3.5 test fault injected after boundary body v1 final MLP submission".into(),
            ));
        }
        self.backend.add(&residual, &mlp)
    }

    /// Run the transformer/recurrent body and commit its caches exactly once.
    /// Logit projection is deliberately separate so generation prefill can
    /// retain only the final hidden row before multiplying by the vocabulary.
    fn forward_hidden(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        let sequence_length = token_ids.len();
        if sequence_length == 0 {
            return Err(Error::Other("qwen3.5 forward: empty token_ids".into()));
        }
        if let Some(token) = token_ids
            .iter()
            .copied()
            .find(|token| *token as usize >= self.config.text.vocab_size)
        {
            return Err(Error::Other(format!(
                "qwen3.5: token id {token} is outside vocabulary size {}",
                self.config.text.vocab_size
            )));
        }
        #[cfg(feature = "metal-experiments")]
        self.ensure_complete_linear_layer_lane_is_not_terminal()?;
        #[cfg(feature = "metal-experiments")]
        if (self.metal_w8_linear_layer_stacks_v1.is_some()
            || self.metal_w8_mlp_stack3_boundary_body_v1.is_some()
            || self.metal_w8_mlp_stack3_boundary_tail_head_v1.is_some())
            && start_pos > 0
            && sequence_length != 1
        {
            return Err(Error::Other(
                "qwen3.5 Metal W8 Stack3 body owns decode state; only single-token decode is valid until reset"
                    .into(),
            ));
        }
        self.state.validate_forward(start_pos, sequence_length)?;

        #[cfg(feature = "metal-experiments")]
        let lane_versions_before = self.complete_linear_layer_lane_versions();

        let body_result = (|| {
            let mut hidden = self
                .backend
                .embedding(&self.weights.token_embedding, token_ids)?;
            let mut layer_index = 0;
            while layer_index < self.config.text.n_layers {
                #[cfg(feature = "metal-experiments")]
                if start_pos > 0 && hidden.shape().dims() == [1, self.config.text.hidden_size] {
                    if self.metal_w8_mlp_stack3_boundary_tail_head_v1.is_some() {
                        if layer_index == QWEN35_MLP_STACK3_BOUNDARY_INITIAL_STACK_V1[0] {
                            hidden = run_linear_layer_stack3_with_metal_w8(
                                &hidden,
                                &mut self
                                    .metal_w8_mlp_stack3_boundary_tail_head_v1
                                    .as_mut()
                                    .expect("boundary + tail-head v1 lane must be present")
                                    .initial_stack,
                            )?;
                            layer_index = QWEN35_MLP_STACK3_BOUNDARY_INITIAL_STACK_V1[2] + 1;
                            continue;
                        }
                        if let Some(boundary_index) = self
                            .metal_w8_mlp_stack3_boundary_tail_head_v1
                            .as_ref()
                            .and_then(|lane| lane.boundary_index(layer_index))
                        {
                            let residual = self
                                .forward_full_attention_residual_for_mlp_stack3_boundary_body_v1(
                                    &hidden,
                                    layer_index,
                                    start_pos,
                                )?;
                            let completed_layers = self
                                .metal_w8_mlp_stack3_boundary_tail_head_v1
                                .as_ref()
                                .expect("boundary + tail-head v1 lane must be present")
                                .boundaries[boundary_index]
                                .stack_layer_indices;
                            hidden = run_mlp_stack3_boundary_with_metal_w8(
                                &residual,
                                &mut self
                                    .metal_w8_mlp_stack3_boundary_tail_head_v1
                                    .as_mut()
                                    .expect("boundary + tail-head v1 lane must be present")
                                    .boundaries[boundary_index],
                            )?;
                            layer_index = completed_layers[2] + 1;
                            continue;
                        }
                        if layer_index == QWEN35_MLP_STACK3_BOUNDARY_FINAL_MLP_V1 {
                            hidden = self
                                .forward_full_attention_residual_for_mlp_stack3_boundary_body_v1(
                                    &hidden,
                                    layer_index,
                                    start_pos,
                                )?;
                            layer_index += 1;
                            continue;
                        }
                        return Err(Error::Other(format!(
                            "qwen3.5 Metal W8 boundary + tail-head v1 has no decode owner for layer {layer_index}"
                        )));
                    }
                    if self.metal_w8_mlp_stack3_boundary_body_v1.is_some() {
                        if layer_index == QWEN35_MLP_STACK3_BOUNDARY_INITIAL_STACK_V1[0] {
                            hidden = run_linear_layer_stack3_with_metal_w8(
                                &hidden,
                                &mut self
                                    .metal_w8_mlp_stack3_boundary_body_v1
                                    .as_mut()
                                    .expect("MLP→Stack3 boundary body v1 lane must be present")
                                    .initial_stack,
                            )?;
                            layer_index = QWEN35_MLP_STACK3_BOUNDARY_INITIAL_STACK_V1[2] + 1;
                            continue;
                        }
                        if let Some(boundary_index) = self
                            .metal_w8_mlp_stack3_boundary_body_v1
                            .as_ref()
                            .and_then(|body| body.boundary_index(layer_index))
                        {
                            let residual = self
                                .forward_full_attention_residual_for_mlp_stack3_boundary_body_v1(
                                    &hidden,
                                    layer_index,
                                    start_pos,
                                )?;
                            let completed_layers = self
                                .metal_w8_mlp_stack3_boundary_body_v1
                                .as_ref()
                                .expect("MLP→Stack3 boundary body v1 lane must be present")
                                .boundaries[boundary_index]
                                .stack_layer_indices;
                            hidden = run_mlp_stack3_boundary_with_metal_w8(
                                &residual,
                                &mut self
                                    .metal_w8_mlp_stack3_boundary_body_v1
                                    .as_mut()
                                    .expect("MLP→Stack3 boundary body v1 lane must be present")
                                    .boundaries[boundary_index],
                            )?;
                            layer_index = completed_layers[2] + 1;
                            continue;
                        }
                        if layer_index == QWEN35_MLP_STACK3_BOUNDARY_FINAL_MLP_V1 {
                            hidden = self
                                .forward_final_full_attention_mlp_for_mlp_stack3_boundary_body_v1(
                                    &hidden,
                                    layer_index,
                                    start_pos,
                                )?;
                            layer_index += 1;
                            continue;
                        }
                        return Err(Error::Other(format!(
                            "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 has no decode owner for layer {layer_index}"
                        )));
                    }
                    if let Some(stack_index) = self
                        .metal_w8_linear_layer_stacks_v1
                        .as_ref()
                        .and_then(|lane| lane.stack_start(layer_index))
                    {
                        let completed_layers = self
                            .metal_w8_linear_layer_stacks_v1
                            .as_ref()
                            .expect("selected stack3-v1 lane must be present")
                            .stacks[stack_index]
                            .layer_indices;
                        hidden = run_linear_layer_stack3_with_metal_w8(
                            &hidden,
                            &mut self
                                .metal_w8_linear_layer_stacks_v1
                                .as_mut()
                                .expect("selected stack3-v1 lane must be present")
                                .stacks[stack_index],
                        )?;
                        #[cfg(all(test, debug_assertions))]
                        if self
                            .fail_after_layer_once_for_test
                            .is_some_and(|fault| completed_layers.contains(&fault))
                        {
                            let fault = self
                                .fail_after_layer_once_for_test
                                .take()
                                .expect("stack3-v1 fault was checked");
                            return Err(Error::Other(format!(
                                "qwen3.5 test fault injected after layer {fault}"
                            )));
                        }
                        layer_index = completed_layers[2] + 1;
                        continue;
                    }
                }
                hidden = self.forward_layer(&hidden, layer_index, start_pos)?;
                #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
                if self.fail_after_layer_once_for_test == Some(layer_index) {
                    self.fail_after_layer_once_for_test = None;
                    return Err(Error::Other(format!(
                        "qwen3.5 test fault injected after layer {layer_index}"
                    )));
                }
                layer_index += 1;
            }
            self.state.advance(sequence_length);
            #[cfg(feature = "metal-experiments")]
            if start_pos == 0 {
                if let Some(lane) = self.metal_w8_mlp_stack3_boundary_tail_head_v1.as_mut() {
                    lane.prefill_body_calls = lane.prefill_body_calls.saturating_add(1);
                }
            }
            Ok(hidden)
        })();

        match body_result {
            Ok(hidden) => Ok(hidden),
            Err(error) => {
                #[cfg(feature = "metal-experiments")]
                if self.latch_complete_linear_layer_lanes_after_partial_commit(lane_versions_before)
                {
                    if self.metal_w8_mlp_stack3_boundary_body_v1.is_some() {
                        return Err(Error::Other(format!(
                            "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 advanced partial CPU/KV or Metal state before a later body failure: {error}; entire lane is terminal, reset required"
                        )));
                    }
                    if self.metal_w8_mlp_stack3_boundary_tail_head_v1.is_some() {
                        return Err(Error::Other(format!(
                            "qwen3.5 Metal W8 boundary + tail-head v1 advanced partial CPU/KV or Metal state before a later body failure: {error}; entire lane is terminal, reset required"
                        )));
                    }
                    let lane_identity = if self.metal_w8_stack3_lm_head_v2_terminal_error.is_some()
                    {
                        "Stack3 + lm_head v2"
                    } else if self.metal_w8_linear_layer_stacks_v1.is_some() {
                        "stack3-v1"
                    } else if self.metal_w8_all_linear_layers_precision_v2.is_some() {
                        "all-linear-layers precision-v2"
                    } else {
                        "linear-layer"
                    };
                    return Err(Error::Other(format!(
                        "qwen3.5 complete W8 {lane_identity} lane committed decode state before a later body failure: {error}; lane is terminal, reset required"
                    )));
                }
                Err(error)
            }
        }
    }

    fn project_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let hidden = self.normalize_output(hidden)?;
        self.project_normalized_logits(&hidden)
    }

    fn project_normalized_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let logits = match &self.weights.lm_head {
            Some(lm_head) => self.backend.matmul(&hidden, lm_head)?,
            None => self
                .backend
                .matmul_rhs_transposed(&hidden, &self.weights.token_embedding)?,
        };
        self.backend.synchronize()?;
        Ok(logits)
    }

    fn normalize_output(&self, hidden: &Tensor) -> Result<Tensor> {
        self.backend.rms_norm_offset(
            hidden,
            &self.weights.output_norm_weight,
            self.config.text.rms_norm_eps,
            1.0,
        )
    }

    fn project_last_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let last_hidden = self.last_hidden_row(hidden)?;
        self.project_logits(&last_hidden)
    }

    fn last_hidden_row(&self, hidden: &Tensor) -> Result<Tensor> {
        let dims = hidden.shape().dims();
        if dims.len() != 2 || dims[0] == 0 || dims[1] != self.config.text.hidden_size {
            return Err(Error::ShapeMismatch {
                expected: format!("[non-zero rows, {}]", self.config.text.hidden_size),
                got: hidden.shape().to_string(),
            });
        }
        // Qwen3.5 is CPU/F32-only today, so copying one hidden row is both
        // cheap and avoids ever allocating `[prompt_len, vocab_size]`.
        let row_start = (dims[0] - 1) * dims[1];
        let last_hidden = Tensor::from_f32(
            vec![1, dims[1]],
            &hidden.as_f32()?[row_start..row_start + dims[1]],
        )?;
        Ok(last_hidden)
    }

    #[cfg(feature = "metal-w8-head-mlp")]
    fn metal_w8_reranked_token(
        &mut self,
        hidden: &Tensor,
        phase: Qwen35MetalW8LmHeadPhase,
    ) -> Result<u32> {
        let normalized = self.normalize_output(hidden)?;
        if normalized.shape().dims() != [1, self.config.text.hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: format!("[1, {}]", self.config.text.hidden_size),
                got: normalized.shape().to_string(),
            });
        }
        #[cfg(feature = "metal-experiments")]
        self.maybe_fail_stack3_lm_head_v2_before_submit_for_test()?;
        let topk_started = std::time::Instant::now();
        let candidates = self
            .metal_w8_lm_head
            .as_mut()
            .expect("Metal W8 head presence checked before advancing state")
            .topk4(normalized.as_f32()?)
            .map_err(|error| {
                Error::Other(format!(
                    "qwen3.5 Metal W8 top-4 {} failed after state advancement: {error}",
                    phase.label()
                ))
            })?;
        let topk_elapsed_ns = topk_started.elapsed().as_nanos();
        let rerank_started = std::time::Instant::now();
        let reranked = rerank_tied_f32_candidates(
            self.weights.token_embedding.as_f32()?,
            normalized.as_f32()?,
            self.config.text.vocab_size,
            self.config.text.hidden_size,
            candidates,
        )
        .map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 F32 {} rerank failed after state advancement: {error}",
                phase.label()
            ))
        })?;
        let rerank_elapsed_ns = rerank_started.elapsed().as_nanos();
        let stats = self
            .metal_w8_lm_head_stats
            .as_mut()
            .expect("Metal W8 head stats presence follows head presence");
        match phase {
            Qwen35MetalW8LmHeadPhase::Prefill => stats.prefill_calls += 1,
            Qwen35MetalW8LmHeadPhase::Decode => stats.decode_calls += 1,
        }
        stats.topk_elapsed_ns += topk_elapsed_ns;
        stats.rerank_elapsed_ns += rerank_elapsed_ns;
        Ok(reranked)
    }

    /// Compare CPU/F32, raw Metal/W8 top-4, and the F32-reranked Metal result
    /// on the same forced decode hidden state. This is a correctness gate, not
    /// the production fast path: it intentionally computes full CPU logits.
    #[cfg(feature = "metal-experiments")]
    pub fn teacher_forced_decode_candidates(
        &mut self,
        token: u32,
        pos: u32,
    ) -> Result<Qwen35MetalTeacherStep> {
        if self.metal_w8_mlp_stack3_boundary_tail_head_v1.is_some() {
            let residual = self.forward_hidden(&[token], pos)?;
            let result = (|| {
                let output = self.run_boundary_tail_head_v1(&residual)?;
                let cpu_logits = self.project_normalized_logits(&output.normalized_hidden)?;
                let cpu_token = argmax_f32_row(&cpu_logits, self.config.text.vocab_size)?;
                let (reranked_token, rerank_elapsed_ns) = self
                    .rerank_boundary_tail_head_v1(&output.normalized_hidden, output.candidates)?;
                self.metal_w8_mlp_stack3_boundary_tail_head_v1
                    .as_mut()
                    .expect("boundary + tail-head v1 lane must be present")
                    .teacher_calls += 1;
                Ok(Qwen35MetalTeacherStep {
                    cpu_token,
                    w8_candidates: output.candidates,
                    reranked_token,
                    topk_elapsed_ns: output.tail_elapsed_ns,
                    rerank_elapsed_ns,
                })
            })();
            return self.finish_boundary_tail_head_v1_post_body(result, "teacher tail/rerank");
        }
        if self.metal_w8_lm_head.is_none() {
            return Err(Error::Other(
                "qwen3.5 teacher gate requires an enabled Metal W8 lm_head".into(),
            ));
        }
        let hidden = self.forward_hidden(&[token], pos)?;
        let result = (|| {
            let normalized = self.normalize_output(&hidden)?;
            let cpu_logits = self.project_normalized_logits(&normalized)?;
            let cpu_token = argmax_f32_row(&cpu_logits, self.config.text.vocab_size)?;
            self.maybe_fail_stack3_lm_head_v2_before_submit_for_test()?;
            let topk_started = std::time::Instant::now();
            let candidates = self
                .metal_w8_lm_head
                .as_mut()
                .expect("Metal W8 head presence checked above")
                .topk4(normalized.as_f32()?)
                .map_err(|error| {
                    Error::Other(format!(
                        "qwen3.5 Metal W8 top-4 teacher gate failed after state advancement: {error}"
                    ))
                })?;
            let topk_elapsed_ns = topk_started.elapsed().as_nanos();
            let rerank_started = std::time::Instant::now();
            let reranked = rerank_tied_f32_candidates(
                self.weights.token_embedding.as_f32()?,
                normalized.as_f32()?,
                self.config.text.vocab_size,
                self.config.text.hidden_size,
                candidates,
            )?;
            let rerank_elapsed_ns = rerank_started.elapsed().as_nanos();
            let stats = self
                .metal_w8_lm_head_stats
                .as_mut()
                .expect("Metal W8 head stats presence follows head presence");
            stats.teacher_calls += 1;
            stats.topk_elapsed_ns += topk_elapsed_ns;
            stats.rerank_elapsed_ns += rerank_elapsed_ns;
            Ok(Qwen35MetalTeacherStep {
                cpu_token,
                w8_candidates: candidates,
                reranked_token: reranked,
                topk_elapsed_ns,
                rerank_elapsed_ns,
            })
        })();
        self.finish_stack3_lm_head_v2_post_body(result, "teacher head")
    }

    /// Backward-compatible teacher-gate surface. The Metal result includes the
    /// exact-F32 candidate rerank, rather than returning raw W8 top-1.
    #[cfg(feature = "metal-experiments")]
    pub fn teacher_forced_decode_argmaxes(&mut self, token: u32, pos: u32) -> Result<(u32, u32)> {
        let comparison = self.teacher_forced_decode_candidates(token, pos)?;
        Ok((comparison.cpu_token, comparison.reranked_token))
    }
}

#[cfg(any(
    feature = "metal-experiments",
    all(test, feature = "metal-w8-head-mlp")
))]
fn argmax_f32_row(logits: &Tensor, vocab_size: usize) -> Result<u32> {
    if logits.shape().dims() != [1, vocab_size] {
        return Err(Error::ShapeMismatch {
            expected: format!("[1, {vocab_size}]"),
            got: logits.shape().to_string(),
        });
    }
    let mut best = f32::NEG_INFINITY;
    let mut best_token = 0u32;
    for (token, &score) in logits.as_f32()?.iter().enumerate() {
        if score > best {
            best = score;
            best_token = token as u32;
        }
    }
    Ok(best_token)
}

/// Recompute only four candidate logits from the original tied F32 embedding.
/// Candidate order does not affect the result; exact-score ties select the
/// lowest token ID just like the native full-head argmax.
#[cfg(feature = "metal-w8-head-mlp")]
fn rerank_tied_f32_candidates(
    embedding: &[f32],
    hidden: &[f32],
    vocab_size: usize,
    hidden_size: usize,
    candidates: [u32; apxinf_metal::W8_TOP_K],
) -> Result<u32> {
    if hidden.len() != hidden_size
        || embedding.len()
            != vocab_size
                .checked_mul(hidden_size)
                .ok_or_else(|| Error::Other("qwen3.5 F32 rerank dimensions overflow".into()))?
    {
        return Err(Error::ShapeMismatch {
            expected: format!("embedding=[{vocab_size}, {hidden_size}], hidden=[{hidden_size}]"),
            got: format!(
                "embedding elements={}, hidden elements={}",
                embedding.len(),
                hidden.len()
            ),
        });
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut best_token = u32::MAX;
    for (candidate_index, &token) in candidates.iter().enumerate() {
        let token_index = token as usize;
        if token_index >= vocab_size {
            return Err(Error::Other(format!(
                "qwen3.5 F32 rerank candidate {candidate_index} token {token} is outside vocabulary {vocab_size}"
            )));
        }
        if candidates[..candidate_index].contains(&token) {
            return Err(Error::Other(format!(
                "qwen3.5 F32 rerank received duplicate candidate token {token}"
            )));
        }
        let row = &embedding[token_index * hidden_size..(token_index + 1) * hidden_size];
        let score = row
            .iter()
            .zip(hidden)
            .fold(0.0f32, |sum, (&weight, &value)| sum + weight * value);
        if !score.is_finite() {
            return Err(Error::Other(format!(
                "qwen3.5 F32 rerank produced a non-finite score for token {token}"
            )));
        }
        if score > best_score || (score == best_score && token < best_token) {
            best_score = score;
            best_token = token;
        }
    }
    Ok(best_token)
}

impl LlmTrait for GeneralQwen35 {
    fn load(
        _config: ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Other(
            "GeneralQwen35 uses the nested qwen3_5 config; load it through AutoModel or GeneralQwen35::from_weights"
                .into(),
        ))
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        let hidden = self.forward_hidden(token_ids, start_pos)?;
        #[cfg(feature = "metal-experiments")]
        let boundary_tail_decode =
            start_pos > 0 && self.metal_w8_mlp_stack3_boundary_tail_head_v1.is_some();
        #[cfg(feature = "metal-experiments")]
        let result = if boundary_tail_decode {
            (|| {
                let output = self.run_boundary_tail_head_v1(&hidden)?;
                self.project_normalized_logits(&output.normalized_hidden)
            })()
        } else {
            self.project_logits(&hidden)
        };
        #[cfg(not(feature = "metal-experiments"))]
        let result = self.project_logits(&hidden);
        #[cfg(feature = "metal-experiments")]
        {
            let result = self.finish_stack3_lm_head_v2_post_body(result, "CPU/F32 projection");
            return self.finish_boundary_tail_head_v1_post_body(result, "CPU/F32 projection");
        }
        #[cfg(not(feature = "metal-experiments"))]
        result
    }

    fn prefill_for_generation(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        if input.image.is_some() {
            return Err(Error::Other(
                "this model does not support image input".into(),
            ));
        }
        let hidden = self.forward_hidden(input.token_ids, 0)?;
        let result = self.project_last_logits(&hidden);
        #[cfg(feature = "metal-experiments")]
        {
            if result.is_ok() {
                if let Some(lane) = self.metal_w8_mlp_stack3_boundary_tail_head_v1.as_mut() {
                    lane.prefill_cpu_head_calls = lane.prefill_cpu_head_calls.saturating_add(1);
                }
            }
            let result =
                self.finish_stack3_lm_head_v2_post_body(result, "CPU/F32 prefill projection");
            return self
                .finish_boundary_tail_head_v1_post_body(result, "CPU/F32 prefill projection");
        }
        #[cfg(not(feature = "metal-experiments"))]
        result
    }

    fn prefill_token_for_generation(&mut self, input: LlmInput<'_>) -> Option<Result<u32>> {
        #[cfg(feature = "metal-w8-head-mlp")]
        {
            self.metal_w8_lm_head.as_ref()?;
            if input.image.is_some() {
                return Some(Err(Error::Other(
                    "this model does not support image input".into(),
                )));
            }
            let hidden = match self.forward_hidden(input.token_ids, 0) {
                Ok(hidden) => hidden,
                Err(error) => return Some(Err(error)),
            };
            let result = (|| {
                let last_hidden = self.last_hidden_row(&hidden)?;
                self.metal_w8_reranked_token(&last_hidden, Qwen35MetalW8LmHeadPhase::Prefill)
            })();
            #[cfg(feature = "metal-experiments")]
            let result = self.finish_stack3_lm_head_v2_post_body(result, "prefill head");
            return Some(result);
        }
        #[cfg(not(feature = "metal-w8-head-mlp"))]
        {
            let _ = input;
            None
        }
    }

    fn validate_generation_budget(&self, prompt_len: usize, max_new_tokens: usize) -> Result<()> {
        let required = prompt_len.checked_add(max_new_tokens).ok_or_else(|| {
            Error::Other("qwen3.5 generation: requested context length overflow".into())
        })?;
        let max_context = self.state.max_context();
        if required > max_context {
            return Err(Error::Other(format!(
                "qwen3.5 generation: prompt length {prompt_len} + generation budget {max_new_tokens} exceeds configured maximum {max_context}"
            )));
        }
        Ok(())
    }

    fn reset(&mut self) {
        let _ = self.state.reset();
        #[cfg(feature = "metal-experiments")]
        let reset_all_linear_precision_v2 = self.metal_w8_all_linear_layers_precision_v2.is_some();
        #[cfg(feature = "metal-experiments")]
        let reset_stack3_full_mlp = self
            .metal_w8_linear_layer_stacks_v1
            .as_ref()
            .is_some_and(|lane| lane.owns_full_attention_mlp_blocks);
        #[cfg(feature = "metal-experiments")]
        if let Some(gdn) = self.metal_w8_gdn.as_mut() {
            let _ = gdn.reset();
        }
        #[cfg(feature = "metal-experiments")]
        if let Some(linear_layer) = self.metal_w8_linear_layer.as_mut() {
            let _ = linear_layer.reset();
        }
        #[cfg(feature = "metal-experiments")]
        if let Some(all_linear_layers) = self.metal_w8_all_linear_layers_precision_v2.as_mut() {
            let _ = all_linear_layers.reset();
        }
        #[cfg(feature = "metal-experiments")]
        if let Some(stack3) = self.metal_w8_linear_layer_stacks_v1.as_mut() {
            let _ = stack3.reset();
        }
        #[cfg(feature = "metal-experiments")]
        if let Some(boundary_body) = self.metal_w8_mlp_stack3_boundary_body_v1.as_mut() {
            let _ = boundary_body.reset();
        }
        #[cfg(feature = "metal-experiments")]
        if let Some(lane) = self.metal_w8_mlp_stack3_boundary_tail_head_v1.as_mut() {
            let _ = lane.reset();
        }
        #[cfg(feature = "metal-experiments")]
        if reset_all_linear_precision_v2 || reset_stack3_full_mlp {
            if let Some(mlp_blocks) = self.metal_w8_mlp_blocks.as_mut() {
                mlp_blocks.reset_stats();
            }
        }
        #[cfg(feature = "metal-experiments")]
        if self.metal_w8_stack3_lm_head_v2_terminal_error.is_some() {
            self.metal_w8_stack3_lm_head_v2_terminal_error = Some(false);
            if let Some(stats) = self.metal_w8_lm_head_stats.as_mut() {
                *stats = Qwen35MetalW8LmHeadStats::default();
            }
        }
        #[cfg(feature = "metal-experiments")]
        if let Some(reference) = self.packed_w8_linear_layer_reference.as_mut() {
            reference.reset();
        }
        #[cfg(all(test, debug_assertions, feature = "metal-experiments"))]
        {
            self.fail_after_layer_once_for_test = None;
            self.fail_stack3_lm_head_v2_before_submit_once_for_test = false;
            self.fail_mlp_stack3_boundary_final_mlp_after_submit_once_for_test = false;
            self.boundary_tail_head_fault_once_for_test = None;
            self.fail_boundary_tail_head_rerank_once_for_test = false;
        }
    }

    fn decode_token(&mut self, token: u32, pos: u32) -> Option<Result<u32>> {
        #[cfg(feature = "metal-w8-head-mlp")]
        {
            #[cfg(feature = "metal-experiments")]
            if self.metal_w8_mlp_stack3_boundary_tail_head_v1.is_some() {
                let residual = match self.forward_hidden(&[token], pos) {
                    Ok(hidden) => hidden,
                    Err(error) => return Some(Err(error)),
                };
                let result = (|| {
                    let output = self.run_boundary_tail_head_v1(&residual)?;
                    let (token, _) = self.rerank_boundary_tail_head_v1(
                        &output.normalized_hidden,
                        output.candidates,
                    )?;
                    let lane = self
                        .metal_w8_mlp_stack3_boundary_tail_head_v1
                        .as_mut()
                        .expect("boundary + tail-head v1 lane must be present");
                    lane.decode_calls = lane.decode_calls.saturating_add(1);
                    Ok(token)
                })();
                return Some(
                    self.finish_boundary_tail_head_v1_post_body(result, "decode tail/rerank"),
                );
            }
            // Presence means the caller explicitly requested this path. Once
            // the CPU body advances state, every error must be returned to the
            // generation loop; silently falling back would advance it twice.
            self.metal_w8_lm_head.as_ref()?;
            let hidden = match self.forward_hidden(&[token], pos) {
                Ok(hidden) => hidden,
                Err(error) => return Some(Err(error)),
            };
            let result = self.metal_w8_reranked_token(&hidden, Qwen35MetalW8LmHeadPhase::Decode);
            #[cfg(feature = "metal-experiments")]
            let result = self.finish_stack3_lm_head_v2_post_body(result, "decode head");
            return Some(result);
        }
        #[cfg(not(feature = "metal-w8-head-mlp"))]
        {
            let _ = (token, pos);
            None
        }
    }

    fn generation_path_receipt(&self) -> Option<serde_json::Value> {
        #[cfg(feature = "metal-experiments")]
        {
            if let Some(lane) = self.metal_w8_mlp_stack3_boundary_tail_head_v1_stats() {
                let aggregate =
                    self.metal_w8_mlp_stack3_boundary_tail_head_v1_aggregate_ledger()?;
                let initial_execution = lane.initial_stack.execution;
                let boundaries = lane
                    .boundaries
                    .iter()
                    .map(|region| {
                        let execution = region.execution;
                        serde_json::json!({
                            "boundary_mlp_layer_index": region.boundary_mlp_layer_index,
                            "stack_layer_indices": region.stack_layer_indices,
                            "mechanism": region.mechanism,
                            "gdn_core_profile": qwen35_gdn_core_profile_v1_label(region.gdn_core_profile),
                            "gdn_function_chain": region.gdn_function_chain,
                            "last_gdn_core_receipt": qwen35_gdn_core_production_receipt_v1_json(region.last_gdn_core_receipt),
                            "kernel_dispatches_per_decode": region.kernel_dispatches_per_decode,
                            "explicit_buffer_barriers_per_decode": region.explicit_buffer_barriers_per_decode,
                            "prefill_seed_calls": region.prefill_seed_calls,
                            "decode_calls": execution.decode_calls,
                            "successful_decodes": execution.successful_decodes,
                            "failed_decodes": execution.failed_decodes,
                            "command_buffers": execution.command_buffers,
                            "compute_encoders": execution.compute_encoders,
                            "commits": execution.commits,
                            "waits": execution.waits,
                            "host_to_device_bytes": execution.host_to_device_bytes,
                            "device_to_host_bytes": execution.device_to_host_bytes,
                            "state_commits": execution.state_commits,
                            "last_state_commit_mask": execution.last_state_commit_mask,
                            "committed_stack_version": execution.committed_stack_version,
                            "terminal_error": region.terminal_error,
                            "block_elapsed_ns": region.block_elapsed_ns,
                        })
                    })
                    .collect::<Vec<_>>();
                return Some(serde_json::json!({
                    "format": "apxinf-qwen35-mlp-stack3-boundary-tail-head-generation-path-v1",
                    "mechanism": lane.mechanism,
                    "gdn_core_profile": qwen35_gdn_core_profile_v1_label(lane.gdn_core_profile),
                    "gdn_function_chain": lane.gdn_function_chain,
                    "cpu_full_attention_and_kv": true,
                    "cpu_prefill_all_24_layers": true,
                    "metal_w8_initial_complete_linear_layer_stack3": true,
                    "metal_w8_mlp_stack3_boundaries": true,
                    "metal_w8_tail_layer23_mlp_final_rms_top4": true,
                    "standalone_layer23_mlp": false,
                    "standalone_metal_lm_head": false,
                    "f32_tied_four_candidate_rerank": true,
                    "initial_stack": {
                        "layer_indices": lane.initial_stack.layer_indices,
                        "mechanism": lane.initial_stack.mechanism,
                        "gdn_core_profile": qwen35_gdn_core_profile_v1_label(lane.initial_stack.gdn_core_profile),
                        "gdn_function_chain": lane.initial_stack.gdn_function_chain,
                        "last_gdn_core_receipt": qwen35_gdn_core_production_receipt_v1_json(lane.initial_stack.last_gdn_core_receipt),
                        "kernel_dispatches_per_decode": lane.initial_stack.kernel_dispatches_per_decode,
                        "explicit_buffer_barriers_per_decode": lane.initial_stack.explicit_buffer_barriers_per_decode,
                        "prefill_seed_calls": lane.initial_stack.prefill_seed_calls,
                        "decode_calls": initial_execution.decode_calls,
                        "successful_decodes": initial_execution.successful_decodes,
                        "failed_decodes": initial_execution.failed_decodes,
                        "command_buffers": initial_execution.command_buffers,
                        "compute_encoders": initial_execution.compute_encoders,
                        "commits": initial_execution.commits,
                        "waits": initial_execution.waits,
                        "host_to_device_bytes": initial_execution.host_to_device_bytes,
                        "device_to_host_bytes": initial_execution.device_to_host_bytes,
                        "state_commits": initial_execution.state_commits,
                        "last_state_commit_mask": initial_execution.last_state_commit_mask,
                        "committed_stack_version": initial_execution.committed_stack_version,
                        "terminal_error": lane.initial_stack.terminal_error,
                        "block_elapsed_ns": lane.initial_stack.block_elapsed_ns,
                    },
                    "boundaries": boundaries,
                    "prefill_body_calls": lane.prefill_body_calls,
                    "prefill_head": {
                        "mechanism": "cpu-f32-tied",
                        "calls": lane.prefill_cpu_head_calls,
                        "tail_transactions": 0,
                    },
                    "decode_head": {
                        "mechanism": "metal-w8-tail-v1",
                        "layer_index": lane.tail_layer_index,
                        "calls": lane.decode_calls,
                        "teacher_calls": lane.teacher_calls,
                        "tail_transactions": lane.tail.decode_calls,
                        "successful_transactions": lane.tail.successful_decodes,
                        "failed_transactions": lane.tail.failed_decodes,
                        "command_buffers": lane.tail.command_buffers,
                        "compute_encoders": lane.tail.compute_encoders,
                        "kernel_dispatches": lane.tail.kernel_dispatches,
                        "commits": lane.tail.commits,
                        "waits": lane.tail.waits,
                        "host_to_device_bytes": lane.tail.host_to_device_bytes,
                        "device_to_host_bytes": lane.tail.device_to_host_bytes,
                        "output_commits": lane.tail.output_commits,
                        "last_output_commit_mask": lane.tail.last_output_commit_mask,
                        "rerank_elapsed_ns": lane.rerank_elapsed_ns,
                        "terminal_error": lane.tail.terminal_error,
                    },
                    "aggregate": {
                        "scope": aggregate.scope,
                        "includes_lm_head": aggregate.includes_lm_head,
                        "persistent_mtlbuffer_bytes": aggregate.total_persistent_mtlbuffer_bytes,
                        "allocated_buffers": aggregate.allocated_buffers,
                        "shared_buffers": aggregate.shared_buffers,
                        "private_buffers": aggregate.private_buffers,
                        "host_to_device_bytes_per_decode": aggregate.host_to_device_bytes_per_decode,
                        "device_to_host_bytes_per_decode": aggregate.device_to_host_bytes_per_decode,
                        "state_host_transfer_bytes_per_decode": aggregate.state_host_transfer_bytes_per_decode,
                        "command_buffers_per_decode": aggregate.command_buffers_per_decode,
                        "compute_encoders_per_decode": aggregate.compute_encoders_per_decode,
                        "kernel_dispatches_per_decode": aggregate.kernel_dispatches_per_decode,
                        "explicit_buffer_barriers_per_decode": aggregate.explicit_buffer_barriers_per_decode,
                        "gdn_core_profile": qwen35_gdn_core_profile_v1_label(aggregate.gdn_core_profile),
                        "gdn_function_chain": aggregate.gdn_function_chain,
                        "commits_per_decode": aggregate.commits_per_decode,
                        "waits_per_decode": aggregate.waits_per_decode,
                    },
                    "terminal_error": lane.terminal_error,
                }));
            }
            if let Some(body) = self.metal_w8_mlp_stack3_boundary_body_v1_stats() {
                let aggregate = self.metal_w8_mlp_stack3_boundary_body_v1_aggregate_ledger()?;
                let initial_execution = body.initial_stack.execution;
                let initial_stack = serde_json::json!({
                    "layer_indices": body.initial_stack.layer_indices,
                    "mechanism": body.initial_stack.mechanism,
                    "gdn_output_group_sizes": body
                        .initial_stack
                        .quantization
                        .map(|ledger| ledger.gdn_output_group_size.columns()),
                    "prefill_seed_calls": body.initial_stack.prefill_seed_calls,
                    "decode_calls": initial_execution.decode_calls,
                    "successful_decodes": initial_execution.successful_decodes,
                    "failed_decodes": initial_execution.failed_decodes,
                    "command_buffers": initial_execution.command_buffers,
                    "compute_encoders": initial_execution.compute_encoders,
                    "commits": initial_execution.commits,
                    "waits": initial_execution.waits,
                    "host_to_device_bytes": initial_execution.host_to_device_bytes,
                    "device_to_host_bytes": initial_execution.device_to_host_bytes,
                    "state_commits": initial_execution.state_commits,
                    "last_state_commit_mask": initial_execution.last_state_commit_mask,
                    "committed_stack_version": initial_execution.committed_stack_version,
                    "terminal_error": body.initial_stack.terminal_error,
                    "block_elapsed_ns": body.initial_stack.block_elapsed_ns,
                });
                let boundaries = body
                    .boundaries
                    .iter()
                    .map(|region| {
                        let execution = region.execution;
                        serde_json::json!({
                            "boundary_mlp_layer_index": region.boundary_mlp_layer_index,
                            "stack_layer_indices": region.stack_layer_indices,
                            "mechanism": region.mechanism,
                            "gdn_output_group_sizes": region
                                .quantization
                                .map(|ledger| ledger.gdn_output_group_size.columns()),
                            "prefill_seed_calls": region.prefill_seed_calls,
                            "decode_calls": execution.decode_calls,
                            "successful_decodes": execution.successful_decodes,
                            "failed_decodes": execution.failed_decodes,
                            "command_buffers": execution.command_buffers,
                            "compute_encoders": execution.compute_encoders,
                            "commits": execution.commits,
                            "waits": execution.waits,
                            "host_to_device_bytes": execution.host_to_device_bytes,
                            "device_to_host_bytes": execution.device_to_host_bytes,
                            "state_commits": execution.state_commits,
                            "last_state_commit_mask": execution.last_state_commit_mask,
                            "committed_stack_version": execution.committed_stack_version,
                            "terminal_error": region.terminal_error,
                            "block_elapsed_ns": region.block_elapsed_ns,
                        })
                    })
                    .collect::<Vec<_>>();
                return Some(serde_json::json!({
                    "format": "apxinf-qwen35-mlp-stack3-boundary-body-generation-path-v1",
                    "mechanism": body.mechanism,
                    "metal_w8_initial_complete_linear_layer_stack3": true,
                    "metal_w8_mlp_stack3_boundaries": true,
                    "metal_w8_final_standalone_mlp": true,
                    "metal_w8_lm_head": false,
                    "cpu_full_attention_and_kv": true,
                    "cpu_prefill_all_24_layers": true,
                    "intermediate_host_finite_checks": false,
                    "final_output_finite_checks": true,
                    "initial_stack": initial_stack,
                    "boundaries": boundaries,
                    "final_mlp": {
                        "layer_index": body.final_mlp.layer_index,
                        "mechanism": "metal-w8-mlp-block-g64",
                        "decode_calls": body.final_mlp.decode_calls,
                        "block_elapsed_ns": body.final_mlp.block_elapsed_ns,
                    },
                    "aggregate": {
                        "scope": aggregate.scope,
                        "includes_lm_head": aggregate.includes_lm_head,
                        "persistent_mtlbuffer_bytes": aggregate.total_persistent_mtlbuffer_bytes,
                        "allocated_buffers": aggregate.allocated_buffers,
                        "shared_buffers": aggregate.shared_buffers,
                        "private_buffers": aggregate.private_buffers,
                        "host_to_device_bytes_per_decode": aggregate.host_to_device_bytes_per_decode,
                        "device_to_host_bytes_per_decode": aggregate.device_to_host_bytes_per_decode,
                        "state_host_transfer_bytes_per_decode": aggregate.state_host_transfer_bytes_per_decode,
                        "command_buffers_per_decode": aggregate.command_buffers_per_decode,
                        "compute_encoders_per_decode": aggregate.compute_encoders_per_decode,
                        "kernel_dispatches_per_decode": aggregate.kernel_dispatches_per_decode,
                        "commits_per_decode": aggregate.commits_per_decode,
                        "waits_per_decode": aggregate.waits_per_decode,
                    },
                    "terminal_error": body.terminal_error,
                }));
            }
            if let Some(composite_terminal) = self.metal_w8_stack3_lm_head_v2_terminal_error {
                let stack_lane = self.metal_w8_linear_layer_stacks_v1_stats()?;
                let stacks = stack_lane
                    .stacks
                    .into_iter()
                    .map(|stack| {
                        let execution = stack.execution;
                        serde_json::json!({
                            "layer_indices": stack.layer_indices,
                            "mechanism": stack.mechanism,
                            "gdn_output_group_sizes": stack
                                .quantization
                                .map(|ledger| ledger.gdn_output_group_size.columns()),
                            "prefill_seed_calls": stack.prefill_seed_calls,
                            "decode_calls": execution.decode_calls,
                            "successful_decodes": execution.successful_decodes,
                            "failed_decodes": execution.failed_decodes,
                            "command_buffers": execution.command_buffers,
                            "compute_encoders": execution.compute_encoders,
                            "commits": execution.commits,
                            "waits": execution.waits,
                            "host_to_device_bytes": execution.host_to_device_bytes,
                            "device_to_host_bytes": execution.device_to_host_bytes,
                            "state_commits": execution.state_commits,
                            "last_state_commit_mask": execution.last_state_commit_mask,
                            "committed_stack_version": execution.committed_stack_version,
                            "intermediate_host_finite_checks_per_decode": stack.intermediate_host_finite_checks_per_decode,
                            "final_output_finite_checks_per_decode": stack.final_output_finite_checks_per_decode,
                            "terminal_error": stack.terminal_error,
                            "block_elapsed_ns": stack.block_elapsed_ns,
                        })
                    })
                    .collect::<Vec<_>>();
                let full_attention_mlp_layers = stack_lane
                    .full_attention_mlp_layers
                    .into_iter()
                    .map(|stats| {
                        serde_json::json!({
                            "layer_index": stats.layer_index,
                            "decode_calls": stats.decode_calls,
                            "block_elapsed_ns": stats.block_elapsed_ns,
                        })
                    })
                    .collect::<Vec<_>>();
                let head = self.metal_w8_lm_head_stats()?;
                return Some(serde_json::json!({
                    "format": "apxinf-qwen35-stack3-lm-head-generation-path-v2",
                    "mechanism": "metal-w8-stack3-lm-head-v2",
                    "stack3_mechanism": stack_lane.mechanism,
                    "full_attention_mlp_mechanism": stack_lane.full_attention_mlp_mechanism,
                    "metal_w8_complete_linear_layer_stacks": true,
                    "metal_w8_full_attention_mlp_blocks": true,
                    "metal_w8_tied_lm_head_topk4_f32_rerank": true,
                    "intermediate_host_finite_checks": false,
                    "final_output_finite_checks": true,
                    "stacks": stacks,
                    "full_attention_mlp_layers": full_attention_mlp_layers,
                    "lm_head": {
                        "mechanism": "metal-w8-top4-f32-rerank",
                        "prefill_calls": head.prefill_calls,
                        "decode_calls": head.decode_calls,
                        "teacher_calls": head.teacher_calls,
                        "topk_elapsed_ns": head.topk_elapsed_ns,
                        "rerank_elapsed_ns": head.rerank_elapsed_ns,
                    },
                    "terminal_error": composite_terminal || stack_lane.terminal_error,
                }));
            }
            if let Some(stack_lane) = self.metal_w8_linear_layer_stacks_v1_stats() {
                let stacks = stack_lane
                    .stacks
                    .into_iter()
                    .map(|stack| {
                        let execution = stack.execution;
                        serde_json::json!({
                            "layer_indices": stack.layer_indices,
                            "mechanism": stack.mechanism,
                            "gdn_output_group_sizes": stack
                                .quantization
                                .map(|ledger| ledger.gdn_output_group_size.columns()),
                            "prefill_seed_calls": stack.prefill_seed_calls,
                            "decode_calls": execution.decode_calls,
                            "successful_decodes": execution.successful_decodes,
                            "failed_decodes": execution.failed_decodes,
                            "command_buffers": execution.command_buffers,
                            "compute_encoders": execution.compute_encoders,
                            "commits": execution.commits,
                            "waits": execution.waits,
                            "host_to_device_bytes": execution.host_to_device_bytes,
                            "device_to_host_bytes": execution.device_to_host_bytes,
                            "state_commits": execution.state_commits,
                            "last_state_commit_mask": execution.last_state_commit_mask,
                            "committed_stack_version": execution.committed_stack_version,
                            "intermediate_host_finite_checks_per_decode": stack.intermediate_host_finite_checks_per_decode,
                            "final_output_finite_checks_per_decode": stack.final_output_finite_checks_per_decode,
                            "terminal_error": stack.terminal_error,
                            "block_elapsed_ns": stack.block_elapsed_ns,
                        })
                    })
                    .collect::<Vec<_>>();
                let full_attention_mlp_layers = stack_lane
                    .full_attention_mlp_layers
                    .into_iter()
                    .map(|stats| {
                        serde_json::json!({
                            "layer_index": stats.layer_index,
                            "decode_calls": stats.decode_calls,
                            "block_elapsed_ns": stats.block_elapsed_ns,
                        })
                    })
                    .collect::<Vec<_>>();
                return Some(serde_json::json!({
                    "format": "apxinf-qwen35-linear-layer-stacks-generation-path-v1",
                    "mechanism": stack_lane.mechanism,
                    "full_attention_mlp_mechanism": stack_lane.full_attention_mlp_mechanism,
                    "metal_w8_complete_linear_layer_stacks": true,
                    "metal_w8_full_attention_mlp_blocks": !full_attention_mlp_layers.is_empty(),
                    "metal_w8_lm_head": self.metal_w8_lm_head.is_some(),
                    "intermediate_host_finite_checks": false,
                    "final_output_finite_checks": true,
                    "stacks": stacks,
                    "full_attention_mlp_layers": full_attention_mlp_layers,
                    "terminal_error": stack_lane.terminal_error,
                }));
            }
            if let Some(all_linear) = self.metal_w8_all_linear_layers_precision_v2_stats() {
                let linear_layers = all_linear
                    .linear_layers
                    .into_iter()
                    .map(|stats| {
                        let execution = stats.execution;
                        serde_json::json!({
                            "layer_index": execution.layer_index,
                            "profile": stats.profile.as_str(),
                            "mechanism": stats.mechanism,
                            "gdn_output_group_size": stats
                                .quantization
                                .gdn_output_group_size
                                .columns(),
                            "prefill_seed_calls": execution.prefill_seed_calls,
                            "decode_calls": execution.decode_calls,
                            "successful_decodes": execution.successful_decodes,
                            "failed_decodes": execution.failed_decodes,
                            "command_buffers": execution.command_buffers,
                            "compute_encoders": execution.compute_encoders,
                            "commits": execution.commits,
                            "waits": execution.waits,
                            "host_to_device_bytes": execution.host_to_device_bytes,
                            "device_to_host_bytes": execution.device_to_host_bytes,
                            "committed_state_version": execution.committed_state_version,
                            "terminal_error": execution.terminal_error,
                            "block_elapsed_ns": execution.block_elapsed_ns,
                        })
                    })
                    .collect::<Vec<_>>();
                let full_attention_mlp_layers = all_linear
                    .full_attention_mlp_layers
                    .into_iter()
                    .map(|stats| {
                        serde_json::json!({
                            "layer_index": stats.layer_index,
                            "decode_calls": stats.decode_calls,
                            "block_elapsed_ns": stats.block_elapsed_ns,
                        })
                    })
                    .collect::<Vec<_>>();
                return Some(serde_json::json!({
                    "format": "apxinf-qwen35-all-linear-layers-generation-path-v2",
                    "profile": all_linear.profile.as_str(),
                    "mechanism": all_linear.mechanism,
                    "full_attention_mlp_mechanism": all_linear.full_attention_mlp_mechanism,
                    "metal_w8_complete_linear_layers": true,
                    "metal_w8_full_attention_mlp_blocks": true,
                    "metal_w8_lm_head": self.metal_w8_lm_head.is_some(),
                    "linear_layers": linear_layers,
                    "full_attention_mlp_layers": full_attention_mlp_layers,
                    "terminal_error": all_linear.terminal_error,
                }));
            }
        }
        #[cfg(feature = "metal-w8-head-mlp")]
        {
            let mlp_layers = self
                .metal_w8_mlp_block_layer_stats()
                .into_iter()
                .map(|stats| {
                    serde_json::json!({
                        "layer_index": stats.layer_index,
                        "decode_calls": stats.decode_calls,
                        "block_elapsed_ns": stats.block_elapsed_ns,
                    })
                })
                .collect::<Vec<_>>();
            let head = self.metal_w8_lm_head_stats().map(|stats| {
                serde_json::json!({
                    "prefill_calls": stats.prefill_calls,
                    "decode_calls": stats.decode_calls,
                    "teacher_calls": stats.teacher_calls,
                    "topk_elapsed_ns": stats.topk_elapsed_ns,
                    "rerank_elapsed_ns": stats.rerank_elapsed_ns,
                })
            });
            return Some(serde_json::json!({
                "format": "apxinf-qwen35-generation-path-v1",
                "metal_w8_mlp_block": self.metal_w8_mlp_blocks.is_some(),
                "metal_w8_lm_head": self.metal_w8_lm_head.is_some(),
                "mlp_block_layers": mlp_layers,
                "lm_head": head,
            }));
        }
        #[cfg(not(feature = "metal-w8-head-mlp"))]
        {
            Some(serde_json::json!({
                "format": "apxinf-qwen35-generation-path-v1",
                "metal_w8_mlp_block": false,
                "metal_w8_lm_head": false,
                "mlp_block_layers": [],
                "lm_head": null,
            }))
        }
    }

    fn vocab_size(&self) -> usize {
        self.config.text.vocab_size
    }
}

fn run_linear_attention(
    backend: &dyn Backend,
    config: &Qwen35TextConfig,
    hidden: &Tensor,
    weights: &RuntimeLinearWeights,
    state: &mut Qwen35LinearState,
) -> Result<Tensor> {
    let sequence_length = hidden.shape().dims()[0];

    let query = backend.matmul(hidden, &weights.query_projection)?;
    let key = backend.matmul(hidden, &weights.key_projection)?;
    let value = backend.matmul(hidden, &weights.value_projection)?;
    let z = backend.matmul(hidden, &weights.z_projection)?;
    let a = backend.matmul(hidden, &weights.a_projection)?;
    let b = backend.matmul(hidden, &weights.b_projection)?;

    let (query, next_query_conv) = backend.causal_depthwise_conv1d(
        &query,
        &weights.query_conv_weight,
        None,
        state.query_conv.as_ref(),
    )?;
    let (key, next_key_conv) = backend.causal_depthwise_conv1d(
        &key,
        &weights.key_conv_weight,
        None,
        state.key_conv.as_ref(),
    )?;
    let (value, next_value_conv) = backend.causal_depthwise_conv1d(
        &value,
        &weights.value_conv_weight,
        None,
        state.value_conv.as_ref(),
    )?;
    let query = backend.silu(&query)?.reshape(vec![
        sequence_length,
        config.linear_num_key_heads,
        config.linear_key_head_dim,
    ])?;
    let key = backend.silu(&key)?.reshape(vec![
        sequence_length,
        config.linear_num_key_heads,
        config.linear_key_head_dim,
    ])?;
    let value = backend.silu(&value)?.reshape(vec![
        sequence_length,
        config.linear_num_value_heads,
        config.linear_value_head_dim,
    ])?;

    // FLA/Qwen3.5 normalizes both Q and K and applies the DeltaNet query
    // scale after normalization. The recurrent primitive intentionally keeps
    // this preprocessing explicit.
    let query = backend.l2_normalize(&query, -1, 1.0e-6)?;
    let query = backend.scale(&query, 1.0 / (config.linear_key_head_dim as f32).sqrt())?;
    let key = backend.l2_normalize(&key, -1, 1.0e-6)?;
    let (core, next_recurrent) = backend.gated_delta_recurrent(
        &query,
        &key,
        &value,
        &a,
        &b,
        &weights.a_log,
        &weights.dt_bias,
        state.recurrent.as_ref(),
    )?;

    let core = core.reshape(vec![
        sequence_length * config.linear_num_value_heads,
        config.linear_value_head_dim,
    ])?;
    // This is the one Qwen3.5 norm that uses the checkpoint weight directly,
    // not the zero-centred Gemma `(1 + weight)` convention.
    let core = backend.rms_norm(&core, &weights.norm_weight, config.rms_norm_eps)?;
    let z = z.reshape(vec![
        sequence_length * config.linear_num_value_heads,
        config.linear_value_head_dim,
    ])?;
    let z = backend.silu(&z)?;
    let output = backend
        .mul(&core, &z)?
        .reshape(vec![sequence_length, config.linear_value_width()])?;
    let output = backend.matmul(&output, &weights.output_projection)?;

    // Commit state only after the complete layer path succeeds.
    state.query_conv = Some(next_query_conv);
    state.key_conv = Some(next_key_conv);
    state.value_conv = Some(next_value_conv);
    state.recurrent = Some(next_recurrent);
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn run_full_attention(
    backend: &dyn Backend,
    config: &Qwen35TextConfig,
    hidden: &Tensor,
    weights: &RuntimeFullWeights,
    kv: &mut dyn apxinf_core::KvCache,
    cache_index: usize,
    start_pos: u32,
    max_context: usize,
) -> Result<Tensor> {
    let sequence_length = hidden.shape().dims()[0];
    let query = backend.matmul(hidden, &weights.query_projection)?;
    let query = query.reshape(vec![
        sequence_length * config.n_attention_heads,
        config.head_dim,
    ])?;
    let query =
        backend.rms_norm_offset(&query, &weights.query_norm_weight, config.rms_norm_eps, 1.0)?;
    let query = query.reshape(vec![
        sequence_length,
        config.n_attention_heads,
        config.head_dim,
    ])?;

    let key = backend
        .matmul(hidden, &weights.key_projection)?
        .reshape(vec![sequence_length * config.n_kv_heads, config.head_dim])?;
    let key = backend.rms_norm_offset(&key, &weights.key_norm_weight, config.rms_norm_eps, 1.0)?;
    let key = key.reshape(vec![sequence_length, config.n_kv_heads, config.head_dim])?;
    let value = backend
        .matmul(hidden, &weights.value_projection)?
        .reshape(vec![sequence_length, config.n_kv_heads, config.head_dim])?;

    // For text all T/H/W positions are identical, so interleaved-axis mRoPE
    // reduces exactly to scalar rotate-half partial RoPE.
    let query = backend.rope_partial(
        &query,
        config.n_attention_heads,
        config.head_dim,
        config.rotary_dim(),
        config.rope.theta,
        start_pos,
        false,
    )?;
    let key = backend.rope_partial(
        &key,
        config.n_kv_heads,
        config.head_dim,
        config.rotary_dim(),
        config.rope.theta,
        start_pos,
        false,
    )?;

    backend.kv_append(kv, cache_index, &key, &value, sequence_length)?;
    let kv_length = kv.seq_len() + sequence_length;
    let attention = if sequence_length == 1 {
        backend.sdpa_decode(
            &query,
            kv,
            cache_index,
            config.n_attention_heads,
            config.n_kv_heads,
            config.head_dim,
            kv_length,
            max_context,
        )?
    } else {
        backend.sdpa_prefill(
            &query,
            kv,
            cache_index,
            config.n_attention_heads,
            config.n_kv_heads,
            config.head_dim,
            kv_length,
            max_context,
        )?
    };

    let attention = if let Some(gate_projection) = &weights.gate_projection {
        let gate = backend.sigmoid(&backend.matmul(hidden, gate_projection)?)?;
        backend.mul(&attention, &gate)?
    } else {
        attention
    };
    backend.matmul(&attention, &weights.output_projection)
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8GdnLayer {
    fn pack(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        layer_index: usize,
    ) -> Result<Self> {
        let layer = weights.layers.get(layer_index).ok_or_else(|| {
            Error::Other(format!(
                "qwen3.5 Metal W8 GDN layer {layer_index} is outside 0..{}",
                weights.layers.len()
            ))
        })?;
        if config.layer_types.get(layer_index) != Some(&Qwen35LayerType::LinearAttention) {
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 GDN layer {layer_index} is not linear attention"
            )));
        }
        let Qwen35AttentionWeights::Linear(attention) = &layer.attention else {
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 GDN layer {layer_index} checkpoint is not linear attention"
            )));
        };

        let dimensions = apxinf_metal::GdnDimensions {
            hidden_size: config.hidden_size,
            key_heads: config.linear_num_key_heads,
            value_heads: config.linear_num_value_heads,
            key_dim: config.linear_key_head_dim,
            value_dim: config.linear_value_head_dim,
            conv_kernel_size: config.linear_conv_kernel_dim,
            rms_norm_eps: config.rms_norm_eps,
        };
        let hidden = config.hidden_size;
        let value_width = config.linear_value_width();
        let qkv_width = config.linear_qkv_width();
        let value_heads = config.linear_num_value_heads;
        let kernel = config.linear_conv_kernel_dim;
        for (label, tensor, expected) in [
            (
                "in_proj_qkv",
                &attention.in_proj_qkv_weight,
                vec![qkv_width, hidden],
            ),
            (
                "in_proj_z",
                &attention.in_proj_z_weight,
                vec![value_width, hidden],
            ),
            (
                "in_proj_a",
                &attention.in_proj_a_weight,
                vec![value_heads, hidden],
            ),
            (
                "in_proj_b",
                &attention.in_proj_b_weight,
                vec![value_heads, hidden],
            ),
            (
                "out_proj",
                &attention.out_proj_weight,
                vec![hidden, value_width],
            ),
            (
                "conv1d",
                &attention.conv1d_weight,
                vec![qkv_width, 1, kernel],
            ),
            ("A_log", &attention.a_log, vec![value_heads]),
            ("dt_bias", &attention.dt_bias, vec![value_heads]),
            (
                "norm",
                &attention.norm_weight,
                vec![config.linear_value_head_dim],
            ),
        ] {
            if tensor.shape().dims() != expected {
                return Err(Error::ShapeMismatch {
                    expected: format!("Metal W8 GDN {label} {:?}", expected),
                    got: tensor.shape().to_string(),
                });
            }
        }

        let qkv = f32_values(&attention.in_proj_qkv_weight, "Metal W8 GDN in_proj_qkv")?;
        let z = f32_values(&attention.in_proj_z_weight, "Metal W8 GDN in_proj_z")?;
        let a = f32_values(&attention.in_proj_a_weight, "Metal W8 GDN in_proj_a")?;
        let b = f32_values(&attention.in_proj_b_weight, "Metal W8 GDN in_proj_b")?;
        let input_elements = dimensions
            .input_projection_rows()
            .checked_mul(hidden)
            .ok_or_else(|| Error::Other("qwen3.5 Metal W8 GDN dimensions overflow".into()))?;
        let mut input_projection = Vec::with_capacity(input_elements);
        input_projection.extend_from_slice(&qkv);
        input_projection.extend_from_slice(&z);
        input_projection.extend_from_slice(&a);
        input_projection.extend_from_slice(&b);
        let output_projection = f32_values(&attention.out_proj_weight, "Metal W8 GDN out_proj")?;
        let conv_weight = f32_values(&attention.conv1d_weight, "Metal W8 GDN conv1d")?;
        let a_log = f32_values(&attention.a_log, "Metal W8 GDN A_log")?;
        let dt_bias = f32_values(&attention.dt_bias, "Metal W8 GDN dt_bias")?;
        let norm_weight = f32_values(&attention.norm_weight, "Metal W8 GDN norm")?;
        let packed = apxinf_metal::PackedW8GdnBlock::pack_f32(
            dimensions,
            apxinf_metal::GdnF32Weights {
                input_projection: &input_projection,
                output_projection: &output_projection,
                conv_weight: &conv_weight,
                a_log: &a_log,
                dt_bias: &dt_bias,
                norm_weight: &norm_weight,
            },
        )
        .map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 GDN layer {layer_index} packing failed: {error}"
            ))
        })?;
        let block = apxinf_metal::MetalW8GdnBlock::from_packed(&packed).map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 GDN layer {layer_index} construction failed: {error}"
            ))
        })?;
        Ok(Self {
            layer_index,
            dimensions,
            block,
            prefill_seed_calls: 0,
            block_elapsed_ns: 0,
            #[cfg(all(test, debug_assertions))]
            fail_next_decode_after_scratch: false,
        })
    }

    fn seed_after_cpu_prefill(&mut self, state: &apxinf_metal::GdnDecodeState) -> Result<()> {
        self.block.seed_decode_state(state).map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 GDN layer {} prefill state seed failed: {error}",
                self.layer_index
            ))
        })?;
        self.prefill_seed_calls = self.prefill_seed_calls.saturating_add(1);
        self.block_elapsed_ns = 0;
        Ok(())
    }

    fn owns_recurrent_state(&self) -> bool {
        self.block.stats().committed_state_version > 0
    }

    fn stats(&self) -> Qwen35MetalW8GdnStats {
        let stats = self.block.stats();
        Qwen35MetalW8GdnStats {
            layer_index: self.layer_index,
            prefill_seed_calls: self.prefill_seed_calls,
            decode_calls: stats.decode_calls,
            command_buffers: stats.command_buffers,
            waits: stats.waits,
            committed_state_version: stats.committed_state_version,
            block_elapsed_ns: self.block_elapsed_ns,
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.block.clear_decode_state().map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 GDN layer {} reset failed: {error}",
                self.layer_index
            ))
        })?;
        self.prefill_seed_calls = 0;
        self.block_elapsed_ns = 0;
        #[cfg(all(test, debug_assertions))]
        {
            self.fail_next_decode_after_scratch = false;
        }
        Ok(())
    }
}

#[cfg(feature = "metal-experiments")]
fn pack_qwen35_w8_linear_layer(
    weights: &Qwen35TextWeights,
    config: &Qwen35TextConfig,
    layer_index: usize,
    profile: Qwen35PackedW8LinearLayerReferenceProfile,
) -> Result<Qwen35PackedW8LinearLayerWeights> {
    let layer = weights.layers.get(layer_index).ok_or_else(|| {
        Error::Other(format!(
            "qwen3.5 Metal W8 linear layer {layer_index} is outside 0..{}",
            weights.layers.len()
        ))
    })?;
    if config.layer_types.get(layer_index) != Some(&Qwen35LayerType::LinearAttention) {
        return Err(Error::Other(format!(
            "qwen3.5 Metal W8 linear layer {layer_index} is not linear attention"
        )));
    }
    let Qwen35AttentionWeights::Linear(attention) = &layer.attention else {
        return Err(Error::Other(format!(
            "qwen3.5 Metal W8 linear layer {layer_index} checkpoint is not linear attention"
        )));
    };

    let dimensions = apxinf_metal::GdnDimensions {
        hidden_size: config.hidden_size,
        key_heads: config.linear_num_key_heads,
        value_heads: config.linear_num_value_heads,
        key_dim: config.linear_key_head_dim,
        value_dim: config.linear_value_head_dim,
        conv_kernel_size: config.linear_conv_kernel_dim,
        rms_norm_eps: config.rms_norm_eps,
    };
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let value_width = config.linear_value_width();
    let qkv_width = config.linear_qkv_width();
    let value_heads = config.linear_num_value_heads;
    let kernel = config.linear_conv_kernel_dim;
    for (label, tensor, expected) in [
        (
            "in_proj_qkv",
            &attention.in_proj_qkv_weight,
            vec![qkv_width, hidden],
        ),
        (
            "in_proj_z",
            &attention.in_proj_z_weight,
            vec![value_width, hidden],
        ),
        (
            "in_proj_a",
            &attention.in_proj_a_weight,
            vec![value_heads, hidden],
        ),
        (
            "in_proj_b",
            &attention.in_proj_b_weight,
            vec![value_heads, hidden],
        ),
        (
            "out_proj",
            &attention.out_proj_weight,
            vec![hidden, value_width],
        ),
        (
            "conv1d",
            &attention.conv1d_weight,
            vec![qkv_width, 1, kernel],
        ),
        ("A_log", &attention.a_log, vec![value_heads]),
        ("dt_bias", &attention.dt_bias, vec![value_heads]),
        (
            "GDN norm",
            &attention.norm_weight,
            vec![config.linear_value_head_dim],
        ),
        ("input RMSNorm", &layer.input_norm_weight, vec![hidden]),
        (
            "post-attention RMSNorm",
            &layer.post_attention_norm_weight,
            vec![hidden],
        ),
        (
            "MLP gate",
            &layer.mlp.gate_proj_weight,
            vec![intermediate, hidden],
        ),
        (
            "MLP up",
            &layer.mlp.up_proj_weight,
            vec![intermediate, hidden],
        ),
        (
            "MLP down",
            &layer.mlp.down_proj_weight,
            vec![hidden, intermediate],
        ),
    ] {
        if tensor.shape().dims() != expected {
            return Err(Error::ShapeMismatch {
                expected: format!("Metal W8 linear layer {label} {:?}", expected),
                got: tensor.shape().to_string(),
            });
        }
    }

    let qkv = f32_values(
        &attention.in_proj_qkv_weight,
        "Metal W8 linear layer in_proj_qkv",
    )?;
    let z = f32_values(
        &attention.in_proj_z_weight,
        "Metal W8 linear layer in_proj_z",
    )?;
    let a = f32_values(
        &attention.in_proj_a_weight,
        "Metal W8 linear layer in_proj_a",
    )?;
    let b = f32_values(
        &attention.in_proj_b_weight,
        "Metal W8 linear layer in_proj_b",
    )?;
    let input_elements = dimensions
        .input_projection_rows()
        .checked_mul(hidden)
        .ok_or_else(|| Error::Other("qwen3.5 Metal W8 linear layer dimensions overflow".into()))?;
    let mut input_projection = Vec::with_capacity(input_elements);
    input_projection.extend_from_slice(&qkv);
    input_projection.extend_from_slice(&z);
    input_projection.extend_from_slice(&a);
    input_projection.extend_from_slice(&b);
    let output_projection =
        f32_values(&attention.out_proj_weight, "Metal W8 linear layer out_proj")?;
    let conv_weight = f32_values(&attention.conv1d_weight, "Metal W8 linear layer conv1d")?;
    let a_log = f32_values(&attention.a_log, "Metal W8 linear layer A_log")?;
    let dt_bias = f32_values(&attention.dt_bias, "Metal W8 linear layer dt_bias")?;
    let gdn_norm = f32_values(&attention.norm_weight, "Metal W8 linear layer GDN norm")?;
    let packed_gdn = apxinf_metal::PackedW8GdnBlock::pack_f32_with_output_group_size(
        dimensions,
        apxinf_metal::GdnF32Weights {
            input_projection: &input_projection,
            output_projection: &output_projection,
            conv_weight: &conv_weight,
            a_log: &a_log,
            dt_bias: &dt_bias,
            norm_weight: &gdn_norm,
        },
        profile.gdn_output_group_size(),
    )
    .map_err(|error| {
        Error::Other(format!(
            "qwen3.5 Metal W8 linear layer {layer_index} GDN packing failed: {error}"
        ))
    })?;

    let gate = f32_values(
        &layer.mlp.gate_proj_weight,
        "Metal W8 linear layer MLP gate",
    )?;
    let up = f32_values(&layer.mlp.up_proj_weight, "Metal W8 linear layer MLP up")?;
    let down = f32_values(
        &layer.mlp.down_proj_weight,
        "Metal W8 linear layer MLP down",
    )?;
    let packed_mlp = apxinf_metal::PackedW8MlpBlock::pack_f32_with_down_group_size(
        &gate,
        &up,
        &down,
        hidden,
        intermediate,
        profile.mlp_down_group_size(),
    )
    .map_err(|error| {
        Error::Other(format!(
            "qwen3.5 Metal W8 linear layer {layer_index} MLP packing failed: {error}"
        ))
    })?;

    // Qwen3.5's two outer layer norms use the zero-centred Gemma
    // convention. Fold the runtime `+ 1` into the resident Metal weights;
    // the internal GDN norm above deliberately remains unshifted.
    let input_rms_weight = f32_values(
        &layer.input_norm_weight,
        "Metal W8 linear layer input RMSNorm",
    )?
    .iter()
    .map(|value| value + 1.0)
    .collect::<Vec<_>>();
    let post_attention_rms_weight = f32_values(
        &layer.post_attention_norm_weight,
        "Metal W8 linear layer post-attention RMSNorm",
    )?
    .iter()
    .map(|value| value + 1.0)
    .collect::<Vec<_>>();
    let packed = apxinf_metal::PackedW8LinearLayerBlock::new(
        packed_gdn,
        packed_mlp,
        &input_rms_weight,
        &post_attention_rms_weight,
        config.rms_norm_eps,
    )
    .map_err(|error| {
        Error::Other(format!(
            "qwen3.5 Metal W8 linear layer {layer_index} assembly failed: {error}"
        ))
    })?;
    Ok(Qwen35PackedW8LinearLayerWeights { dimensions, packed })
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8LinearLayer {
    fn pack(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        layer_index: usize,
    ) -> Result<Self> {
        let packed = pack_qwen35_w8_linear_layer(
            weights,
            config,
            layer_index,
            Qwen35PackedW8LinearLayerReferenceProfile::G64,
        )?;
        let block = apxinf_metal::MetalW8LinearLayerBlock::from_packed(&packed.packed).map_err(
            |error| {
                Error::Other(format!(
                    "qwen3.5 Metal W8 linear layer {layer_index} construction failed: {error}"
                ))
            },
        )?;
        Ok(Self {
            layer_index,
            dimensions: packed.dimensions,
            block,
            prefill_seed_calls: 0,
            block_elapsed_ns: 0,
            seeded: false,
            terminal_error: false,
            precision_profile: None,
            quantization: None,
            #[cfg(all(test, debug_assertions))]
            fail_next_decode_after_scratch: false,
        })
    }

    fn pack_precision_v2(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        layer_index: usize,
        profile: Qwen35MetalW8LinearLayerPrecisionProfile,
    ) -> Result<Self> {
        let packed_profile = match profile {
            Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2 => {
                Qwen35PackedW8LinearLayerReferenceProfile::GdnOutG32
            }
        };
        let packed = pack_qwen35_w8_linear_layer(weights, config, layer_index, packed_profile)?;
        let quantization = packed.packed.quantization_ledger().map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 precision-v2 linear layer {layer_index} ledger failed: {error}"
            ))
        })?;
        let block = apxinf_metal::MetalW8LinearLayerBlock::from_packed_gdn_out_g32(
            &packed.packed,
        )
        .map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 precision-v2 linear layer {layer_index} construction failed: {error}"
            ))
        })?;
        Ok(Self {
            layer_index,
            dimensions: packed.dimensions,
            block,
            prefill_seed_calls: 0,
            block_elapsed_ns: 0,
            seeded: false,
            terminal_error: false,
            precision_profile: Some(profile),
            quantization: Some(quantization),
            #[cfg(all(test, debug_assertions))]
            fail_next_decode_after_scratch: false,
        })
    }

    fn seed_after_cpu_prefill(&mut self, state: &apxinf_metal::GdnDecodeState) -> Result<()> {
        self.block.seed_decode_state(state).map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 linear layer {} prefill state seed failed: {error}",
                self.layer_index
            ))
        })?;
        self.prefill_seed_calls = self.prefill_seed_calls.saturating_add(1);
        self.block_elapsed_ns = 0;
        self.seeded = true;
        self.terminal_error = false;
        Ok(())
    }

    fn stats(&self) -> Qwen35MetalW8LinearLayerStats {
        let stats = self.block.stats();
        Qwen35MetalW8LinearLayerStats {
            layer_index: self.layer_index,
            prefill_seed_calls: self.prefill_seed_calls,
            decode_calls: stats.decode_calls,
            successful_decodes: stats.successful_decodes,
            failed_decodes: stats.failed_decodes,
            command_buffers: stats.command_buffers,
            compute_encoders: stats.compute_encoders,
            commits: stats.commits,
            waits: stats.waits,
            host_to_device_bytes: stats.host_to_device_bytes,
            device_to_host_bytes: stats.device_to_host_bytes,
            committed_state_version: stats.committed_state_version,
            terminal_error: self.terminal_error,
            block_elapsed_ns: self.block_elapsed_ns,
        }
    }

    fn precision_v2_stats(&self) -> Option<Qwen35MetalW8LinearLayerPrecisionV2Stats> {
        let profile = self.precision_profile?;
        Some(Qwen35MetalW8LinearLayerPrecisionV2Stats {
            profile,
            mechanism: profile.mechanism(),
            quantization: self.quantization?,
            execution: self.stats(),
        })
    }

    fn reset(&mut self) -> Result<()> {
        self.block.clear_decode_state().map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 linear layer {} reset failed: {error}",
                self.layer_index
            ))
        })?;
        self.prefill_seed_calls = 0;
        self.block_elapsed_ns = 0;
        self.seeded = false;
        self.terminal_error = false;
        #[cfg(all(test, debug_assertions))]
        {
            self.fail_next_decode_after_scratch = false;
        }
        Ok(())
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8AllLinearLayersPrecisionV2AggregateLedger {
    fn new(
        linear_layers: Vec<Qwen35MetalW8LinearLayerBufferLedger>,
        full_attention_mlp_layers: Vec<Qwen35MetalW8MlpBlockBufferLedger>,
    ) -> Self {
        macro_rules! total {
            ($field:ident) => {
                linear_layers
                    .iter()
                    .map(|entry| entry.ledger.$field)
                    .chain(
                        full_attention_mlp_layers
                            .iter()
                            .map(|entry| entry.ledger.$field),
                    )
                    .fold(0usize, usize::saturating_add)
            };
        }
        Self {
            scope: "resident-mtlbuffer-only",
            exclusions: "CPU F32 weights, host Vec allocations, Metal pipelines/libraries/queues, driver allocations, KV cache, and lm_head",
            includes_lm_head: false,
            total_persistent_mtlbuffer_bytes: total!(total_persistent_bytes),
            allocated_buffers: total!(allocated_buffers),
            shared_buffers: total!(shared_buffers),
            private_buffers: total!(private_buffers),
            host_to_device_bytes_per_decode: total!(host_input_bytes_per_decode),
            device_to_host_bytes_per_decode: total!(host_output_bytes_per_decode),
            state_host_transfer_bytes_per_decode: total!(state_host_transfer_bytes_per_decode),
            command_buffers_per_decode: total!(command_buffers_per_decode),
            compute_encoders_per_decode: total!(compute_encoders_per_decode),
            commits_per_decode: total!(commits_per_decode),
            waits_per_decode: total!(waits_per_decode),
            linear_layers,
            full_attention_mlp_layers,
        }
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8AllLinearLayersPrecisionV2 {
    fn pack(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        profile: Qwen35MetalW8LinearLayerPrecisionProfile,
    ) -> Result<Self> {
        if weights.layers.len() != config.layer_types.len() {
            return Err(Error::Other(format!(
                "qwen3.5 all-linear precision-v2 received {} weight layers for {} configured layer types",
                weights.layers.len(),
                config.layer_types.len()
            )));
        }
        let mut layers = (0..weights.layers.len()).map(|_| None).collect::<Vec<_>>();
        for (layer_index, layer_type) in config.layer_types.iter().enumerate() {
            if *layer_type == Qwen35LayerType::LinearAttention {
                layers[layer_index] = Some(Qwen35MetalW8LinearLayer::pack_precision_v2(
                    weights,
                    config,
                    layer_index,
                    profile,
                )?);
            }
        }
        if layers.iter().all(Option::is_none) {
            return Err(Error::Other(
                "qwen3.5 all-linear precision-v2 requires at least one linear-attention layer"
                    .into(),
            ));
        }
        Ok(Self {
            profile,
            layers,
            terminal_error: false,
        })
    }

    fn layer(&self, layer_index: usize) -> Option<&Qwen35MetalW8LinearLayer> {
        self.layers.get(layer_index).and_then(Option::as_ref)
    }

    fn layer_mut(&mut self, layer_index: usize) -> Option<&mut Qwen35MetalW8LinearLayer> {
        self.layers.get_mut(layer_index).and_then(Option::as_mut)
    }

    fn stats(
        &self,
        full_attention_mlp_layers: Vec<Qwen35MetalW8MlpBlockStats>,
    ) -> Qwen35MetalW8AllLinearLayersPrecisionV2Stats {
        Qwen35MetalW8AllLinearLayersPrecisionV2Stats {
            profile: self.profile,
            mechanism: "metal-w8-all-linear-layers-precision-v2",
            full_attention_mlp_mechanism: "metal-w8-mlp-block-g64",
            linear_layers: self
                .layers
                .iter()
                .flatten()
                .map(|layer| {
                    layer
                        .precision_v2_stats()
                        .expect("all-linear precision-v2 layer must expose its precision receipt")
                })
                .collect(),
            full_attention_mlp_layers,
            terminal_error: self.is_terminal(),
        }
    }

    fn buffer_ledgers(&self) -> Vec<Qwen35MetalW8LinearLayerBufferLedger> {
        self.layers
            .iter()
            .flatten()
            .map(|layer| Qwen35MetalW8LinearLayerBufferLedger {
                layer_index: layer.layer_index,
                ledger: layer.block.buffer_ledger(),
            })
            .collect()
    }

    fn is_terminal(&self) -> bool {
        self.terminal_error
            || self
                .layers
                .iter()
                .flatten()
                .any(|layer| layer.terminal_error)
    }

    fn latch_terminal(&mut self) {
        self.terminal_error = true;
    }

    fn reset(&mut self) -> Result<()> {
        for layer in self.layers.iter_mut().flatten() {
            if let Err(error) = layer.reset() {
                self.terminal_error = true;
                return Err(error);
            }
        }
        self.terminal_error = false;
        Ok(())
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8LinearLayerStack3V1 {
    fn pack(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        layer_indices: [usize; 3],
    ) -> Result<Self> {
        Self::pack_impl(
            weights,
            config,
            layer_indices,
            apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch,
            false,
        )
    }

    fn pack_with_gdn_core_profile_v1(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        layer_indices: [usize; 3],
        gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
    ) -> Result<Self> {
        validate_qwen35_production_gdn_core_profile_v1(gdn_core_profile)?;
        Self::pack_impl(weights, config, layer_indices, gdn_core_profile, true)
    }

    fn pack_impl(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        layer_indices: [usize; 3],
        gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
        use_production_profile_api: bool,
    ) -> Result<Self> {
        let packed = layer_indices
            .into_iter()
            .map(|layer_index| {
                pack_qwen35_w8_linear_layer(
                    weights,
                    config,
                    layer_index,
                    Qwen35PackedW8LinearLayerReferenceProfile::GdnOutG32,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let packed: [Qwen35PackedW8LinearLayerWeights; 3] = packed
            .try_into()
            .map_err(|_| Error::Other("qwen3.5 Metal W8 stack3-v1 packing depth changed".into()))?;
        let dimensions = packed[0].dimensions;
        let quantization = [
            packed[0].packed.quantization_ledger(),
            packed[1].packed.quantization_ledger(),
            packed[2].packed.quantization_ledger(),
        ]
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 stack3-v1 quantization ledger failed: {error}"
            ))
        })?
        .try_into()
        .map_err(|_| Error::Other("qwen3.5 Metal W8 stack3-v1 ledger depth changed".into()))?;
        let packed_refs = [&packed[0].packed, &packed[1].packed, &packed[2].packed];
        let block = match (use_production_profile_api, gdn_core_profile) {
            (false, apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch) => {
                apxinf_metal::MetalW8LinearLayerStack3::from_packed_gdn_out_g32_v1(packed_refs)
            }
            (
                true,
                apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch
                | apxinf_metal::GdnCoreProfileV1::Fused128,
            ) => apxinf_metal::MetalW8LinearLayerStack3::from_packed_gdn_out_g32_with_gdn_core_profile_v1(
                packed_refs,
                gdn_core_profile,
            ),
            _ => unreachable!("invalid Stack3 production-profile construction state"),
        }
        .map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 stack3-v1 layers {layer_indices:?} profile {} construction failed: {error}",
                qwen35_gdn_core_profile_v1_label(gdn_core_profile),
            ))
        })?;
        Ok(Self {
            layer_indices,
            gdn_core_profile,
            dimensions,
            block,
            quantization,
            pending_prefill_states: std::array::from_fn(|_| None),
            prefill_seed_calls: [0; 3],
            block_elapsed_ns: 0,
            seeded: false,
            terminal_error: false,
            #[cfg(all(test, debug_assertions))]
            fail_next_decode_after_scratch: false,
        })
    }

    fn seed_layer_after_cpu_prefill(
        &mut self,
        slot: usize,
        state: &apxinf_metal::GdnDecodeState,
    ) -> Result<()> {
        if self.terminal_error
            || self.seeded
            || slot >= 3
            || self.pending_prefill_states[slot].is_some()
        {
            self.terminal_error = true;
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 stack3-v1 {:?} received an invalid or duplicate prefill seed for slot {slot}; reset required",
                self.layer_indices
            )));
        }
        self.pending_prefill_states[slot] = Some(state.clone());
        self.prefill_seed_calls[slot] = self.prefill_seed_calls[slot].saturating_add(1);
        if self.pending_prefill_states.iter().all(Option::is_some) {
            let states = std::array::from_fn(|index| {
                self.pending_prefill_states[index]
                    .as_ref()
                    .expect("all three stack3 prefill states were checked")
                    .clone()
            });
            if let Err(error) = self.block.seed_decode_states(&states) {
                self.terminal_error = true;
                return Err(Error::Other(format!(
                    "qwen3.5 Metal W8 stack3-v1 {:?} prefill seed failed terminally: {error}; reset required",
                    self.layer_indices
                )));
            }
            self.pending_prefill_states = std::array::from_fn(|_| None);
            self.block_elapsed_ns = 0;
            self.seeded = true;
        }
        Ok(())
    }

    fn stats(&self) -> Qwen35MetalW8LinearLayerStack3V1Stats {
        let ledger = self.block.buffer_ledger();
        Qwen35MetalW8LinearLayerStack3V1Stats {
            layer_indices: self.layer_indices,
            mechanism: qwen35_stack3_mechanism_for_gdn_core_profile_v1(self.gdn_core_profile),
            gdn_core_profile: self.gdn_core_profile,
            gdn_function_chain: self.gdn_core_profile.expected_function_chain(),
            quantization: self.quantization,
            prefill_seed_calls: self.prefill_seed_calls,
            execution: self.block.stats(),
            last_gdn_core_receipt: self.block.last_gdn_core_receipt(),
            kernel_dispatches_per_decode: ledger.kernel_dispatches_per_decode,
            explicit_buffer_barriers_per_decode: ledger.explicit_buffer_barriers_per_decode,
            intermediate_host_finite_checks_per_decode: ledger
                .intermediate_host_finite_checks_per_decode,
            final_output_finite_checks_per_decode: ledger.final_output_finite_checks_per_decode,
            terminal_error: self.terminal_error,
            block_elapsed_ns: self.block_elapsed_ns,
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.block.clear_decode_states().map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 stack3-v1 {:?} reset failed: {error}",
                self.layer_indices
            ))
        })?;
        self.pending_prefill_states = std::array::from_fn(|_| None);
        self.prefill_seed_calls = [0; 3];
        self.block_elapsed_ns = 0;
        self.seeded = false;
        self.terminal_error = false;
        #[cfg(all(test, debug_assertions))]
        {
            self.fail_next_decode_after_scratch = false;
        }
        Ok(())
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8LinearLayerStacksV1AggregateLedger {
    fn new(
        stacks: Vec<Qwen35MetalW8LinearLayerStack3BufferLedger>,
        full_attention_mlp_layers: Vec<Qwen35MetalW8MlpBlockBufferLedger>,
    ) -> Self {
        macro_rules! total {
            ($field:ident) => {
                stacks
                    .iter()
                    .map(|entry| entry.ledger.$field)
                    .chain(
                        full_attention_mlp_layers
                            .iter()
                            .map(|entry| entry.ledger.$field),
                    )
                    .fold(0usize, usize::saturating_add)
            };
        }
        Self {
            scope: "resident-mtlbuffer-only",
            exclusions: "CPU F32 weights, host Vec allocations, Metal pipelines/libraries/queues, driver allocations, KV cache, and lm_head",
            includes_lm_head: false,
            total_persistent_mtlbuffer_bytes: total!(total_persistent_bytes),
            allocated_buffers: total!(allocated_buffers),
            shared_buffers: total!(shared_buffers),
            private_buffers: total!(private_buffers),
            host_to_device_bytes_per_decode: total!(host_input_bytes_per_decode),
            device_to_host_bytes_per_decode: total!(host_output_bytes_per_decode),
            state_host_transfer_bytes_per_decode: total!(state_host_transfer_bytes_per_decode),
            command_buffers_per_decode: total!(command_buffers_per_decode),
            compute_encoders_per_decode: total!(compute_encoders_per_decode),
            commits_per_decode: total!(commits_per_decode),
            waits_per_decode: total!(waits_per_decode),
            intermediate_host_finite_checks_per_decode: stacks
                .iter()
                .map(|entry| entry.ledger.intermediate_host_finite_checks_per_decode)
                .fold(0usize, usize::saturating_add),
            final_output_finite_checks_per_decode: stacks
                .iter()
                .map(|entry| entry.ledger.final_output_finite_checks_per_decode)
                .fold(0usize, usize::saturating_add),
            stacks,
            full_attention_mlp_layers,
        }
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8Stack3LmHeadV2AggregateLedger {
    fn new(
        body: Qwen35MetalW8LinearLayerStacksV1AggregateLedger,
        lm_head: apxinf_metal::LmHeadBufferLedger,
    ) -> Result<Self> {
        fn add(left: usize, right: usize, label: &str) -> Result<usize> {
            left.checked_add(right).ok_or_else(|| {
                Error::Other(format!("qwen3.5 Stack3 + lm_head v2 {label} overflow"))
            })
        }
        Ok(Self {
            scope: "resident-mtlbuffer-only",
            exclusions: "host F32 tied embedding and exact four-candidate F32 rerank, other CPU F32 weights, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, and KV cache",
            includes_lm_head: true,
            total_persistent_mtlbuffer_bytes: add(
                body.total_persistent_mtlbuffer_bytes,
                lm_head.total_persistent_bytes,
                "persistent bytes",
            )?,
            allocated_buffers: add(
                body.allocated_buffers,
                lm_head.allocated_buffers,
                "buffer count",
            )?,
            shared_buffers: add(
                body.shared_buffers,
                lm_head.shared_buffers,
                "shared buffer count",
            )?,
            private_buffers: add(
                body.private_buffers,
                lm_head.private_buffers,
                "private buffer count",
            )?,
            host_to_device_bytes_per_call: add(
                body.host_to_device_bytes_per_decode,
                lm_head.host_input_bytes_per_call,
                "host-to-device bytes",
            )?,
            device_to_host_bytes_per_call: add(
                body.device_to_host_bytes_per_decode,
                lm_head.host_output_bytes_per_call,
                "device-to-host bytes",
            )?,
            state_host_transfer_bytes_per_call: add(
                body.state_host_transfer_bytes_per_decode,
                lm_head.state_host_transfer_bytes_per_call,
                "state transfer bytes",
            )?,
            command_buffers_per_call: add(
                body.command_buffers_per_decode,
                lm_head.command_buffers_per_call,
                "command buffers",
            )?,
            compute_encoders_per_call: add(
                body.compute_encoders_per_decode,
                lm_head.compute_encoders_per_call,
                "compute encoders",
            )?,
            commits_per_call: add(
                body.commits_per_decode,
                lm_head.commits_per_call,
                "commits",
            )?,
            waits_per_call: add(
                body.waits_per_decode,
                lm_head.waits_per_call,
                "waits",
            )?,
            intermediate_host_finite_checks_per_call: body
                .intermediate_host_finite_checks_per_decode,
            final_output_finite_checks_per_call: body.final_output_finite_checks_per_decode,
            body,
            lm_head,
        })
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8LinearLayerStacksV1 {
    fn pack(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        first_layer_indices: &[usize],
        owns_full_attention_mlp_blocks: bool,
    ) -> Result<Self> {
        if weights.layers.len() != config.layer_types.len() || first_layer_indices.is_empty() {
            return Err(Error::Other(
                "qwen3.5 Metal W8 stack3-v1 requires matching weights and at least one stack"
                    .into(),
            ));
        }
        let mut layer_to_stack = vec![None; weights.layers.len()];
        let mut stacks = Vec::with_capacity(first_layer_indices.len());
        for &first in first_layer_indices {
            let layer_indices = [first, first.saturating_add(1), first.saturating_add(2)];
            for (slot, &layer_index) in layer_indices.iter().enumerate() {
                let Some(layer_type) = config.layer_types.get(layer_index) else {
                    return Err(Error::Other(format!(
                        "qwen3.5 Metal W8 stack3-v1 layers {layer_indices:?} exceed 0..{}",
                        config.layer_types.len()
                    )));
                };
                if *layer_type != Qwen35LayerType::LinearAttention {
                    return Err(Error::Other(format!(
                        "qwen3.5 Metal W8 stack3-v1 layer {layer_index} is not linear attention"
                    )));
                }
                if layer_to_stack[layer_index].is_some() {
                    return Err(Error::Other(format!(
                        "qwen3.5 Metal W8 stack3-v1 layer {layer_index} was selected more than once"
                    )));
                }
                layer_to_stack[layer_index] = Some((stacks.len(), slot));
            }
            stacks.push(Qwen35MetalW8LinearLayerStack3V1::pack(
                weights,
                config,
                layer_indices,
            )?);
        }
        Ok(Self {
            stacks,
            layer_to_stack,
            terminal_error: false,
            owns_full_attention_mlp_blocks,
        })
    }

    fn selected_slot(&self, layer_index: usize) -> Option<(usize, usize)> {
        self.layer_to_stack.get(layer_index).copied().flatten()
    }

    fn stack_start(&self, layer_index: usize) -> Option<usize> {
        self.selected_slot(layer_index)
            .and_then(|(stack_index, slot)| (slot == 0).then_some(stack_index))
    }

    fn seed_layer_after_cpu_prefill(
        &mut self,
        layer_index: usize,
        state: &apxinf_metal::GdnDecodeState,
    ) -> Result<()> {
        let (stack_index, slot) = self.selected_slot(layer_index).ok_or_else(|| {
            Error::Other(format!(
                "qwen3.5 Metal W8 stack3-v1 layer {layer_index} is not selected"
            ))
        })?;
        self.stacks[stack_index].seed_layer_after_cpu_prefill(slot, state)
    }

    fn stats(
        &self,
        full_attention_mlp_layers: Vec<Qwen35MetalW8MlpBlockStats>,
    ) -> Qwen35MetalW8LinearLayerStacksV1Stats {
        Qwen35MetalW8LinearLayerStacksV1Stats {
            mechanism: "metal-w8-linear-layer-stack3-v1",
            full_attention_mlp_mechanism: if self.owns_full_attention_mlp_blocks {
                "metal-w8-mlp-block-g64"
            } else {
                "none"
            },
            stacks: self.stacks.iter().map(|stack| stack.stats()).collect(),
            full_attention_mlp_layers: if self.owns_full_attention_mlp_blocks {
                full_attention_mlp_layers
            } else {
                Vec::new()
            },
            terminal_error: self.is_terminal(),
        }
    }

    fn buffer_ledgers(&self) -> Vec<Qwen35MetalW8LinearLayerStack3BufferLedger> {
        self.stacks
            .iter()
            .map(|stack| Qwen35MetalW8LinearLayerStack3BufferLedger {
                layer_indices: stack.layer_indices,
                ledger: stack.block.buffer_ledger(),
            })
            .collect()
    }

    fn is_terminal(&self) -> bool {
        self.terminal_error || self.stacks.iter().any(|stack| stack.terminal_error)
    }

    fn latch_terminal(&mut self) {
        self.terminal_error = true;
    }

    fn reset(&mut self) -> Result<()> {
        for stack in &mut self.stacks {
            if let Err(error) = stack.reset() {
                self.terminal_error = true;
                return Err(error);
            }
        }
        self.terminal_error = false;
        Ok(())
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8MlpStack3BoundaryRegionV1 {
    fn pack(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        boundary_mlp_layer_index: usize,
        stack_layer_indices: [usize; 3],
    ) -> Result<Self> {
        Self::pack_impl(
            weights,
            config,
            boundary_mlp_layer_index,
            stack_layer_indices,
            apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch,
            false,
        )
    }

    fn pack_with_gdn_core_profile_v1(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        boundary_mlp_layer_index: usize,
        stack_layer_indices: [usize; 3],
        gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
    ) -> Result<Self> {
        validate_qwen35_production_gdn_core_profile_v1(gdn_core_profile)?;
        Self::pack_impl(
            weights,
            config,
            boundary_mlp_layer_index,
            stack_layer_indices,
            gdn_core_profile,
            true,
        )
    }

    fn pack_impl(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        boundary_mlp_layer_index: usize,
        stack_layer_indices: [usize; 3],
        gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
        use_production_profile_api: bool,
    ) -> Result<Self> {
        if config.layer_types.get(boundary_mlp_layer_index) != Some(&Qwen35LayerType::FullAttention)
            || !matches!(
                weights
                    .layers
                    .get(boundary_mlp_layer_index)
                    .map(|layer| &layer.attention),
                Some(Qwen35AttentionWeights::Full(_))
            )
        {
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary v1 requires full-attention layer at index {boundary_mlp_layer_index}"
            )));
        }

        let packed_stack = stack_layer_indices
            .into_iter()
            .map(|layer_index| {
                pack_qwen35_w8_linear_layer(
                    weights,
                    config,
                    layer_index,
                    Qwen35PackedW8LinearLayerReferenceProfile::GdnOutG32,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let [stack0, stack1, stack2]: [Qwen35PackedW8LinearLayerWeights; 3] =
            packed_stack.try_into().map_err(|_| {
                Error::Other("qwen3.5 Metal W8 MLP→Stack3 boundary v1 packing depth changed".into())
            })?;
        let dimensions = stack0.dimensions;
        let quantization = [
            stack0.packed.quantization_ledger(),
            stack1.packed.quantization_ledger(),
            stack2.packed.quantization_ledger(),
        ]
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary v1 quantization ledger failed: {error}"
            ))
        })?
        .try_into()
        .map_err(|_| {
            Error::Other("qwen3.5 Metal W8 MLP→Stack3 boundary v1 ledger depth changed".into())
        })?;
        let boundary_mlp = pack_qwen35_w8_mlp_block(weights, config, boundary_mlp_layer_index)?;
        let boundary_layer = &weights.layers[boundary_mlp_layer_index];
        if boundary_layer.post_attention_norm_weight.shape().dims() != [config.hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: format!(
                    "Metal W8 MLP→Stack3 boundary v1 post-attention RMSNorm [{}]",
                    config.hidden_size
                ),
                got: boundary_layer
                    .post_attention_norm_weight
                    .shape()
                    .to_string(),
            });
        }
        // Qwen3.5 outer RMS weights use the zero-centred Gemma convention;
        // the primitive ABI accepts the already-folded multiplicative weight.
        let boundary_post_attention_rms_weight = f32_values(
            &boundary_layer.post_attention_norm_weight,
            "Metal W8 MLP→Stack3 boundary v1 post-attention RMSNorm",
        )?
        .iter()
        .map(|value| value + 1.0)
        .collect::<Vec<_>>();
        let packed = apxinf_metal::PackedW8MlpStack3BoundaryV1::new(
            boundary_mlp,
            &boundary_post_attention_rms_weight,
            config.rms_norm_eps,
            [stack0.packed, stack1.packed, stack2.packed],
        )
        .map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary v1 layer {boundary_mlp_layer_index} assembly failed: {error}"
            ))
        })?;
        let block = match (use_production_profile_api, gdn_core_profile) {
            (false, apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch) => {
                apxinf_metal::MetalW8MlpStack3BoundaryV1::from_packed(&packed)
            }
            (
                true,
                apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch
                | apxinf_metal::GdnCoreProfileV1::Fused128,
            ) => {
                apxinf_metal::MetalW8MlpStack3BoundaryV1::from_packed_with_gdn_core_profile_v1(
                    &packed,
                    gdn_core_profile,
                )
            }
            _ => unreachable!("invalid boundary production-profile construction state"),
        }
        .map_err(|error| {
                Error::Other(format!(
                    "qwen3.5 Metal W8 MLP→Stack3 boundary v1 layer {boundary_mlp_layer_index} profile {} construction failed: {error}",
                    qwen35_gdn_core_profile_v1_label(gdn_core_profile),
                ))
            })?;
        Ok(Self {
            boundary_mlp_layer_index,
            stack_layer_indices,
            gdn_core_profile,
            dimensions,
            block,
            quantization,
            pending_prefill_states: std::array::from_fn(|_| None),
            prefill_seed_calls: [0; 3],
            block_elapsed_ns: 0,
            seeded: false,
            terminal_error: false,
            #[cfg(all(test, debug_assertions))]
            fail_next_decode_after_scratch: false,
        })
    }

    fn stats(&self) -> Qwen35MetalW8MlpStack3BoundaryRegionV1Stats {
        let ledger = self.block.buffer_ledger();
        Qwen35MetalW8MlpStack3BoundaryRegionV1Stats {
            boundary_mlp_layer_index: self.boundary_mlp_layer_index,
            stack_layer_indices: self.stack_layer_indices,
            mechanism: qwen35_boundary_mechanism_for_gdn_core_profile_v1(self.gdn_core_profile),
            gdn_core_profile: self.gdn_core_profile,
            gdn_function_chain: self.gdn_core_profile.expected_function_chain(),
            quantization: self.quantization,
            prefill_seed_calls: self.prefill_seed_calls,
            execution: self.block.stats(),
            last_gdn_core_receipt: self.block.last_gdn_core_receipt(),
            kernel_dispatches_per_decode: ledger.kernel_dispatches_per_decode,
            explicit_buffer_barriers_per_decode: ledger.explicit_buffer_barriers_per_decode,
            terminal_error: self.terminal_error,
            block_elapsed_ns: self.block_elapsed_ns,
        }
    }

    fn seed_layer_after_cpu_prefill(
        &mut self,
        slot: usize,
        state: &apxinf_metal::GdnDecodeState,
    ) -> Result<()> {
        if self.terminal_error
            || self.seeded
            || slot >= 3
            || self.pending_prefill_states[slot].is_some()
        {
            self.terminal_error = true;
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary v1 layer {} received an invalid or duplicate prefill seed for slot {slot}; reset required",
                self.boundary_mlp_layer_index
            )));
        }
        self.pending_prefill_states[slot] = Some(state.clone());
        self.prefill_seed_calls[slot] = self.prefill_seed_calls[slot].saturating_add(1);
        if self.pending_prefill_states.iter().all(Option::is_some) {
            let states = std::array::from_fn(|index| {
                self.pending_prefill_states[index]
                    .as_ref()
                    .expect("all three boundary prefill states were checked")
                    .clone()
            });
            if let Err(error) = self.block.seed_decode_states(&states) {
                self.terminal_error = true;
                return Err(Error::Other(format!(
                    "qwen3.5 Metal W8 MLP→Stack3 boundary v1 layer {} prefill seed failed terminally: {error}; reset required",
                    self.boundary_mlp_layer_index
                )));
            }
            self.pending_prefill_states = std::array::from_fn(|_| None);
            self.block_elapsed_ns = 0;
            self.seeded = true;
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.block.clear_decode_states().map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary v1 layer {} reset failed: {error}",
                self.boundary_mlp_layer_index
            ))
        })?;
        self.pending_prefill_states = std::array::from_fn(|_| None);
        self.prefill_seed_calls = [0; 3];
        self.block_elapsed_ns = 0;
        self.seeded = false;
        self.terminal_error = false;
        #[cfg(all(test, debug_assertions))]
        {
            self.fail_next_decode_after_scratch = false;
        }
        Ok(())
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8MlpStack3BoundaryBodyV1 {
    fn pack(weights: &Qwen35TextWeights, config: &Qwen35TextConfig) -> Result<Self> {
        Self::validate_config_schedule(config)?;
        if weights.layers.len() != 24 {
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 requires exactly 24 weight layers, got {}",
                weights.layers.len()
            )));
        }
        for &layer_index in &QWEN35_MLP_STACK3_BOUNDARY_INITIAL_STACK_V1 {
            Self::require_layer_kind(
                weights,
                config,
                layer_index,
                Qwen35LayerType::LinearAttention,
            )?;
        }
        for &(boundary_mlp_layer_index, stack_layer_indices) in
            &QWEN35_MLP_STACK3_BOUNDARY_REGIONS_V1
        {
            Self::require_layer_kind(
                weights,
                config,
                boundary_mlp_layer_index,
                Qwen35LayerType::FullAttention,
            )?;
            for &layer_index in &stack_layer_indices {
                Self::require_layer_kind(
                    weights,
                    config,
                    layer_index,
                    Qwen35LayerType::LinearAttention,
                )?;
            }
        }
        Self::require_layer_kind(
            weights,
            config,
            QWEN35_MLP_STACK3_BOUNDARY_FINAL_MLP_V1,
            Qwen35LayerType::FullAttention,
        )?;

        let initial_stack = Qwen35MetalW8LinearLayerStack3V1::pack(
            weights,
            config,
            QWEN35_MLP_STACK3_BOUNDARY_INITIAL_STACK_V1,
        )?;
        let boundaries = QWEN35_MLP_STACK3_BOUNDARY_REGIONS_V1
            .into_iter()
            .map(|(boundary_mlp_layer_index, stack_layer_indices)| {
                Qwen35MetalW8MlpStack3BoundaryRegionV1::pack(
                    weights,
                    config,
                    boundary_mlp_layer_index,
                    stack_layer_indices,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let final_mlp = Qwen35MetalW8MlpBlockLayer::pack(
            weights,
            config,
            QWEN35_MLP_STACK3_BOUNDARY_FINAL_MLP_V1,
        )?;
        Ok(Self {
            initial_stack,
            boundaries,
            final_mlp,
            terminal_error: false,
        })
    }

    fn validate_config_schedule(config: &Qwen35TextConfig) -> Result<()> {
        if config.n_layers != 24 || config.layer_types.len() != 24 {
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 requires exactly 24 layers, got config={} and layer_types={}",
                config.n_layers,
                config.layer_types.len()
            )));
        }
        for &layer_index in &QWEN35_MLP_STACK3_BOUNDARY_INITIAL_STACK_V1 {
            if config.layer_types[layer_index] != Qwen35LayerType::LinearAttention {
                return Err(Error::Other(format!(
                    "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 requires linear-attention layer at index {layer_index}"
                )));
            }
        }
        for &(boundary_mlp_layer_index, stack_layer_indices) in
            &QWEN35_MLP_STACK3_BOUNDARY_REGIONS_V1
        {
            if config.layer_types[boundary_mlp_layer_index] != Qwen35LayerType::FullAttention {
                return Err(Error::Other(format!(
                    "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 requires full-attention layer at index {boundary_mlp_layer_index}"
                )));
            }
            for &layer_index in &stack_layer_indices {
                if config.layer_types[layer_index] != Qwen35LayerType::LinearAttention {
                    return Err(Error::Other(format!(
                        "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 requires linear-attention layer at index {layer_index}"
                    )));
                }
            }
        }
        if config.layer_types[QWEN35_MLP_STACK3_BOUNDARY_FINAL_MLP_V1]
            != Qwen35LayerType::FullAttention
        {
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 requires full-attention layer at index {}",
                QWEN35_MLP_STACK3_BOUNDARY_FINAL_MLP_V1
            )));
        }
        Ok(())
    }

    fn require_layer_kind(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        layer_index: usize,
        expected: Qwen35LayerType,
    ) -> Result<()> {
        let configured = config.layer_types.get(layer_index).copied();
        let checkpoint_matches = matches!(
            (
                expected,
                weights
                    .layers
                    .get(layer_index)
                    .map(|layer| &layer.attention)
            ),
            (
                Qwen35LayerType::LinearAttention,
                Some(Qwen35AttentionWeights::Linear(_))
            ) | (
                Qwen35LayerType::FullAttention,
                Some(Qwen35AttentionWeights::Full(_))
            )
        );
        if configured != Some(expected) || !checkpoint_matches {
            let expected_label = match expected {
                Qwen35LayerType::LinearAttention => "linear-attention",
                Qwen35LayerType::FullAttention => "full-attention",
            };
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 requires {expected_label} layer at index {layer_index}"
            )));
        }
        Ok(())
    }

    fn stats(&self) -> Qwen35MetalW8MlpStack3BoundaryBodyV1Stats {
        Qwen35MetalW8MlpStack3BoundaryBodyV1Stats {
            mechanism: "metal-w8-mlp-stack3-boundary-body-v1",
            initial_stack: self.initial_stack.stats(),
            boundaries: self
                .boundaries
                .iter()
                .map(|region| region.stats())
                .collect(),
            final_mlp: Qwen35MetalW8MlpBlockStats {
                layer_index: self.final_mlp.layer_index,
                decode_calls: self.final_mlp.decode_calls,
                block_elapsed_ns: self.final_mlp.block_elapsed_ns,
            },
            terminal_error: self.terminal_error
                || self.initial_stack.terminal_error
                || self.boundaries.iter().any(|region| region.terminal_error),
        }
    }

    fn selected_linear_slot(&self, layer_index: usize) -> Option<(Option<usize>, usize)> {
        if let Some(slot) = self
            .initial_stack
            .layer_indices
            .iter()
            .position(|&index| index == layer_index)
        {
            return Some((None, slot));
        }
        self.boundaries
            .iter()
            .enumerate()
            .find_map(|(region_index, region)| {
                region
                    .stack_layer_indices
                    .iter()
                    .position(|&index| index == layer_index)
                    .map(|slot| (Some(region_index), slot))
            })
    }

    fn seed_layer_after_cpu_prefill(
        &mut self,
        layer_index: usize,
        state: &apxinf_metal::GdnDecodeState,
    ) -> Result<()> {
        let (region_index, slot) = self.selected_linear_slot(layer_index).ok_or_else(|| {
            Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 layer {layer_index} is not owned"
            ))
        })?;
        match region_index {
            None => self.initial_stack.seed_layer_after_cpu_prefill(slot, state),
            Some(region_index) => {
                self.boundaries[region_index].seed_layer_after_cpu_prefill(slot, state)
            }
        }
    }

    fn boundary_index(&self, layer_index: usize) -> Option<usize> {
        self.boundaries
            .iter()
            .position(|region| region.boundary_mlp_layer_index == layer_index)
    }

    fn is_terminal(&self) -> bool {
        self.terminal_error
            || self.initial_stack.terminal_error
            || self.boundaries.iter().any(|region| region.terminal_error)
    }

    fn latch_terminal(&mut self) {
        self.terminal_error = true;
    }

    fn reset(&mut self) -> Result<()> {
        if let Err(error) = self.initial_stack.reset() {
            self.terminal_error = true;
            return Err(error);
        }
        for region in &mut self.boundaries {
            if let Err(error) = region.reset() {
                self.terminal_error = true;
                return Err(error);
            }
        }
        self.final_mlp.decode_calls = 0;
        self.final_mlp.block_elapsed_ns = 0;
        self.terminal_error = false;
        Ok(())
    }

    fn aggregate_ledger(&self) -> Result<Qwen35MetalW8MlpStack3BoundaryBodyV1AggregateLedger> {
        Qwen35MetalW8MlpStack3BoundaryBodyV1AggregateLedger::new(
            Qwen35MetalW8LinearLayerStack3BufferLedger {
                layer_indices: self.initial_stack.layer_indices,
                ledger: self.initial_stack.block.buffer_ledger(),
            },
            self.boundaries
                .iter()
                .map(
                    |region| Qwen35MetalW8MlpStack3BoundaryRegionBufferLedgerV1 {
                        boundary_mlp_layer_index: region.boundary_mlp_layer_index,
                        stack_layer_indices: region.stack_layer_indices,
                        ledger: region.block.buffer_ledger(),
                    },
                )
                .collect(),
            Qwen35MetalW8MlpBlockBufferLedger {
                layer_index: self.final_mlp.layer_index,
                ledger: self.final_mlp.mlp.buffer_ledger(),
            },
        )
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8MlpStack3BoundaryTailHeadV1 {
    fn pack_with_gdn_core_profile_v1(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        gdn_core_profile: apxinf_metal::GdnCoreProfileV1,
    ) -> Result<Self> {
        validate_qwen35_production_gdn_core_profile_v1(gdn_core_profile)?;
        if gdn_core_profile == apxinf_metal::GdnCoreProfileV1::Fused128 {
            validate_qwen35_gdn_core_fused_v1_shape(config)?;
        }
        Qwen35MetalW8MlpStack3BoundaryBodyV1::validate_config_schedule(config)?;
        if weights.layers.len() != 24 || weights.lm_head_weight.is_some() {
            return Err(Error::Other(
                "qwen3.5 Metal W8 boundary + tail-head v1 requires 24 layers and tied word embeddings"
                    .into(),
            ));
        }
        // Preserve the existing generic-shape Legacy test/diagnostic lane.
        // At the frozen Qwen3.5-0.8B shape, A and C both use the receipt-
        // bearing production profile API so the campaign observes the same
        // bridge contract in both arms.
        let use_production_profile_api = qwen35_matches_gdn_core_fused_v1_shape(config);
        let initial_stack = if use_production_profile_api {
            Qwen35MetalW8LinearLayerStack3V1::pack_with_gdn_core_profile_v1(
                weights,
                config,
                QWEN35_MLP_STACK3_BOUNDARY_INITIAL_STACK_V1,
                gdn_core_profile,
            )?
        } else {
            Qwen35MetalW8LinearLayerStack3V1::pack(
                weights,
                config,
                QWEN35_MLP_STACK3_BOUNDARY_INITIAL_STACK_V1,
            )?
        };
        let boundaries = QWEN35_MLP_STACK3_BOUNDARY_REGIONS_V1
            .into_iter()
            .map(|(boundary_mlp_layer_index, stack_layer_indices)| {
                if use_production_profile_api {
                    Qwen35MetalW8MlpStack3BoundaryRegionV1::pack_with_gdn_core_profile_v1(
                        weights,
                        config,
                        boundary_mlp_layer_index,
                        stack_layer_indices,
                        gdn_core_profile,
                    )
                } else {
                    Qwen35MetalW8MlpStack3BoundaryRegionV1::pack(
                        weights,
                        config,
                        boundary_mlp_layer_index,
                        stack_layer_indices,
                    )
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let tail_layer = weights
            .layers
            .get(QWEN35_MLP_STACK3_BOUNDARY_FINAL_MLP_V1)
            .ok_or_else(|| Error::Other("qwen3.5 tail-head v1 layer 23 is missing".into()))?;
        if tail_layer.post_attention_norm_weight.shape().dims() != [config.hidden_size]
            || weights.output_norm_weight.shape().dims() != [config.hidden_size]
            || weights.token_embedding.shape().dims() != [config.vocab_size, config.hidden_size]
        {
            return Err(Error::Other(
                "qwen3.5 Metal W8 tail-head v1 RMS or tied embedding shape changed".into(),
            ));
        }
        let post_attention_rms_weight = f32_values(
            &tail_layer.post_attention_norm_weight,
            "Metal W8 tail-head v1 post-attention RMSNorm",
        )?
        .iter()
        .map(|weight| weight + 1.0)
        .collect::<Vec<_>>();
        let final_rms_weight = f32_values(
            &weights.output_norm_weight,
            "Metal W8 tail-head v1 final RMSNorm",
        )?
        .iter()
        .map(|weight| weight + 1.0)
        .collect::<Vec<_>>();
        let mlp =
            pack_qwen35_w8_mlp_block(weights, config, QWEN35_MLP_STACK3_BOUNDARY_FINAL_MLP_V1)?;
        let embedding = f32_values(
            &weights.token_embedding,
            "Metal W8 tail-head v1 tied embedding",
        )?;
        let vocab =
            apxinf_metal::PackedW8Rows::pack_f32(&embedding, config.vocab_size, config.hidden_size)
                .map_err(|error| {
                    Error::Other(format!(
                        "qwen3.5 Metal W8 tail-head v1 tied embedding packing failed: {error}"
                    ))
                })?;
        let packed = apxinf_metal::PackedW8TailMlpHeadV1::new(
            mlp,
            &post_attention_rms_weight,
            &final_rms_weight,
            config.rms_norm_eps,
            vocab,
        )
        .map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 tail-head v1 assembly failed: {error}"
            ))
        })?;
        let tail = apxinf_metal::MetalW8TailMlpHeadV1::from_packed(&packed).map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 tail-head v1 construction failed: {error}"
            ))
        })?;
        Ok(Self {
            gdn_core_profile,
            initial_stack,
            boundaries,
            tail_layer_index: QWEN35_MLP_STACK3_BOUNDARY_FINAL_MLP_V1,
            tail,
            prefill_body_calls: 0,
            prefill_cpu_head_calls: 0,
            decode_calls: 0,
            teacher_calls: 0,
            rerank_elapsed_ns: 0,
            terminal_error: false,
        })
    }

    fn stats(&self) -> Qwen35MetalW8MlpStack3BoundaryTailHeadV1Stats {
        Qwen35MetalW8MlpStack3BoundaryTailHeadV1Stats {
            mechanism: qwen35_boundary_tail_head_mechanism_for_gdn_core_profile_v1(
                self.gdn_core_profile,
            ),
            gdn_core_profile: self.gdn_core_profile,
            gdn_function_chain: self.gdn_core_profile.expected_function_chain(),
            initial_stack: self.initial_stack.stats(),
            boundaries: self
                .boundaries
                .iter()
                .map(|region| region.stats())
                .collect(),
            tail_layer_index: self.tail_layer_index,
            tail: self.tail.stats(),
            prefill_body_calls: self.prefill_body_calls,
            prefill_cpu_head_calls: self.prefill_cpu_head_calls,
            decode_calls: self.decode_calls,
            teacher_calls: self.teacher_calls,
            rerank_elapsed_ns: self.rerank_elapsed_ns,
            terminal_error: self.terminal_error
                || self.initial_stack.terminal_error
                || self.boundaries.iter().any(|region| region.terminal_error)
                || self.tail.stats().terminal_error,
        }
    }

    fn selected_linear_slot(&self, layer_index: usize) -> Option<(Option<usize>, usize)> {
        if let Some(slot) = self
            .initial_stack
            .layer_indices
            .iter()
            .position(|&index| index == layer_index)
        {
            return Some((None, slot));
        }
        self.boundaries
            .iter()
            .enumerate()
            .find_map(|(region_index, region)| {
                region
                    .stack_layer_indices
                    .iter()
                    .position(|&index| index == layer_index)
                    .map(|slot| (Some(region_index), slot))
            })
    }

    fn seed_layer_after_cpu_prefill(
        &mut self,
        layer_index: usize,
        state: &apxinf_metal::GdnDecodeState,
    ) -> Result<()> {
        let (region_index, slot) = self.selected_linear_slot(layer_index).ok_or_else(|| {
            Error::Other(format!(
                "qwen3.5 Metal W8 boundary + tail-head v1 layer {layer_index} is not owned"
            ))
        })?;
        match region_index {
            None => self.initial_stack.seed_layer_after_cpu_prefill(slot, state),
            Some(region_index) => {
                self.boundaries[region_index].seed_layer_after_cpu_prefill(slot, state)
            }
        }
    }

    fn boundary_index(&self, layer_index: usize) -> Option<usize> {
        self.boundaries
            .iter()
            .position(|region| region.boundary_mlp_layer_index == layer_index)
    }

    fn is_terminal(&self) -> bool {
        self.terminal_error
            || self.initial_stack.terminal_error
            || self.boundaries.iter().any(|region| region.terminal_error)
            || self.tail.stats().terminal_error
    }

    fn latch_terminal(&mut self) {
        self.terminal_error = true;
    }

    fn reset(&mut self) -> Result<()> {
        if let Err(error) = self.initial_stack.reset() {
            self.terminal_error = true;
            return Err(error);
        }
        for region in &mut self.boundaries {
            if let Err(error) = region.reset() {
                self.terminal_error = true;
                return Err(error);
            }
        }
        self.tail.reset().map_err(|error| {
            self.terminal_error = true;
            Error::Other(format!(
                "qwen3.5 Metal W8 boundary + tail-head v1 reset failed: {error}"
            ))
        })?;
        self.prefill_body_calls = 0;
        self.prefill_cpu_head_calls = 0;
        self.decode_calls = 0;
        self.teacher_calls = 0;
        self.rerank_elapsed_ns = 0;
        self.terminal_error = false;
        Ok(())
    }

    fn aggregate_ledger(&self) -> Result<Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger> {
        Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger::new(
            Qwen35MetalW8LinearLayerStack3BufferLedger {
                layer_indices: self.initial_stack.layer_indices,
                ledger: self.initial_stack.block.buffer_ledger(),
            },
            self.boundaries
                .iter()
                .map(
                    |region| Qwen35MetalW8MlpStack3BoundaryRegionBufferLedgerV1 {
                        boundary_mlp_layer_index: region.boundary_mlp_layer_index,
                        stack_layer_indices: region.stack_layer_indices,
                        ledger: region.block.buffer_ledger(),
                    },
                )
                .collect(),
            self.tail_layer_index,
            self.tail.buffer_ledger(),
        )
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger {
    fn new(
        initial_stack: Qwen35MetalW8LinearLayerStack3BufferLedger,
        boundaries: Vec<Qwen35MetalW8MlpStack3BoundaryRegionBufferLedgerV1>,
        tail_layer_index: usize,
        tail: apxinf_metal::TailMlpHeadBufferLedgerV1,
    ) -> Result<Self> {
        fn add(left: usize, right: usize, label: &str) -> Result<usize> {
            left.checked_add(right).ok_or_else(|| {
                Error::Other(format!(
                    "qwen3.5 Metal W8 boundary + tail-head v1 {label} overflow"
                ))
            })
        }
        macro_rules! total {
            ($field:ident, $tail_field:ident) => {{
                let mut sum = initial_stack.ledger.$field;
                for boundary in &boundaries {
                    sum = add(sum, boundary.ledger.$field, stringify!($field))?;
                }
                add(sum, tail.$tail_field, stringify!($field))?
            }};
        }
        validate_qwen35_production_gdn_core_profile_v1(initial_stack.ledger.gdn_core_profile)?;
        let gdn_core_profile = initial_stack.ledger.gdn_core_profile;
        let gdn_function_chain = gdn_core_profile.expected_function_chain();
        if initial_stack.ledger.gdn_function_chain != gdn_function_chain
            || boundaries.len() != QWEN35_MLP_STACK3_BOUNDARY_REGIONS_V1.len()
            || boundaries.iter().any(|boundary| {
                boundary.ledger.gdn_core_profile != gdn_core_profile
                    || boundary.ledger.gdn_function_chain != gdn_function_chain
            })
        {
            return Err(Error::Other(
                "qwen3.5 Metal W8 boundary + tail-head v1 requires one consistent production GDN core profile across the initial stack and all five boundaries"
                    .into(),
            ));
        }
        let kernel_dispatches_per_decode =
            total!(kernel_dispatches_per_decode, kernel_dispatches_per_decode);
        let explicit_buffer_barriers_per_decode = total!(
            explicit_buffer_barriers_per_decode,
            buffer_barriers_per_decode
        );
        let expected_topology = match gdn_core_profile {
            apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch => (267, 243),
            apxinf_metal::GdnCoreProfileV1::Fused128 => (213, 189),
            apxinf_metal::GdnCoreProfileV1::QkStagedFourDispatch => unreachable!(
                "qk-staged production profile was rejected before aggregate construction"
            ),
        };
        if (
            kernel_dispatches_per_decode,
            explicit_buffer_barriers_per_decode,
        ) != expected_topology
        {
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 boundary + tail-head v1 profile {} topology is {kernel_dispatches_per_decode} dispatches/{explicit_buffer_barriers_per_decode} barriers, expected {}/{}",
                qwen35_gdn_core_profile_v1_label(gdn_core_profile),
                expected_topology.0,
                expected_topology.1,
            )));
        }
        let command_buffers_per_decode =
            total!(command_buffers_per_decode, command_buffers_per_decode);
        let compute_encoders_per_decode =
            total!(compute_encoders_per_decode, compute_encoders_per_decode);
        let commits_per_decode = total!(commits_per_decode, commits_per_decode);
        let waits_per_decode = total!(waits_per_decode, waits_per_decode);
        if (
            command_buffers_per_decode,
            compute_encoders_per_decode,
            commits_per_decode,
            waits_per_decode,
        ) != (7, 24, 7, 7)
        {
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 boundary + tail-head v1 submission topology is {command_buffers_per_decode} command buffers/{compute_encoders_per_decode} encoders/{commits_per_decode} commits/{waits_per_decode} waits, expected 7/24/7/7"
            )));
        }
        Ok(Self {
            scope: "resident-mtlbuffer-only",
            exclusions: "CPU F32 weights, host F32 tied embedding and exact four-candidate rerank, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, full attention/KV, model loader, and prefill CPU head",
            includes_lm_head: true,
            gdn_core_profile,
            gdn_function_chain,
            total_persistent_mtlbuffer_bytes: total!(total_persistent_bytes, total_persistent_bytes),
            allocated_buffers: total!(allocated_buffers, allocated_buffers),
            shared_buffers: total!(shared_buffers, shared_buffers),
            private_buffers: total!(private_buffers, private_buffers),
            host_to_device_bytes_per_decode: total!(
                host_input_bytes_per_decode,
                host_input_bytes_per_decode
            ),
            device_to_host_bytes_per_decode: total!(
                host_output_bytes_per_decode,
                host_output_bytes_per_decode
            ),
            state_host_transfer_bytes_per_decode: total!(
                state_host_transfer_bytes_per_decode,
                state_host_transfer_bytes_per_decode
            ),
            command_buffers_per_decode,
            compute_encoders_per_decode,
            kernel_dispatches_per_decode,
            explicit_buffer_barriers_per_decode,
            commits_per_decode,
            waits_per_decode,
            initial_stack,
            boundaries,
            tail_layer_index,
            tail,
        })
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8MlpStack3BoundaryBodyV1AggregateLedger {
    fn new(
        initial_stack: Qwen35MetalW8LinearLayerStack3BufferLedger,
        boundaries: Vec<Qwen35MetalW8MlpStack3BoundaryRegionBufferLedgerV1>,
        final_mlp: Qwen35MetalW8MlpBlockBufferLedger,
    ) -> Result<Self> {
        fn add(left: usize, right: usize, label: &str) -> Result<usize> {
            left.checked_add(right).ok_or_else(|| {
                Error::Other(format!(
                    "qwen3.5 Metal W8 MLP→Stack3 boundary body v1 {label} overflow"
                ))
            })
        }
        macro_rules! total {
            ($field:ident) => {{
                let mut sum = initial_stack.ledger.$field;
                for entry in &boundaries {
                    sum = add(sum, entry.ledger.$field, stringify!($field))?;
                }
                add(sum, final_mlp.ledger.$field, stringify!($field))?
            }};
        }
        let mut kernel_dispatches_per_decode = initial_stack.ledger.kernel_dispatches_per_decode;
        let mut intermediate_host_finite_checks_per_decode = initial_stack
            .ledger
            .intermediate_host_finite_checks_per_decode;
        let mut final_output_finite_checks_per_decode =
            initial_stack.ledger.final_output_finite_checks_per_decode;
        for entry in &boundaries {
            kernel_dispatches_per_decode = add(
                kernel_dispatches_per_decode,
                entry.ledger.kernel_dispatches_per_decode,
                "kernel dispatch count",
            )?;
            intermediate_host_finite_checks_per_decode = add(
                intermediate_host_finite_checks_per_decode,
                entry.ledger.intermediate_host_finite_checks_per_decode,
                "intermediate finite-check count",
            )?;
            final_output_finite_checks_per_decode = add(
                final_output_finite_checks_per_decode,
                entry.ledger.final_output_finite_checks_per_decode,
                "final finite-check count",
            )?;
        }
        // The standalone MLP contains three kernels in its single compute
        // transaction. Its legacy ledger predates the explicit dispatch field.
        kernel_dispatches_per_decode =
            add(kernel_dispatches_per_decode, 3, "kernel dispatch count")?;
        Ok(Self {
            scope: "resident-mtlbuffer-only",
            exclusions: "CPU F32 weights, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, full attention/KV, output norm, and lm_head",
            includes_lm_head: false,
            total_persistent_mtlbuffer_bytes: total!(total_persistent_bytes),
            allocated_buffers: total!(allocated_buffers),
            shared_buffers: total!(shared_buffers),
            private_buffers: total!(private_buffers),
            host_to_device_bytes_per_decode: total!(host_input_bytes_per_decode),
            device_to_host_bytes_per_decode: total!(host_output_bytes_per_decode),
            state_host_transfer_bytes_per_decode: total!(state_host_transfer_bytes_per_decode),
            command_buffers_per_decode: total!(command_buffers_per_decode),
            compute_encoders_per_decode: total!(compute_encoders_per_decode),
            kernel_dispatches_per_decode,
            commits_per_decode: total!(commits_per_decode),
            waits_per_decode: total!(waits_per_decode),
            intermediate_host_finite_checks_per_decode,
            final_output_finite_checks_per_decode,
            initial_stack,
            boundaries,
            final_mlp,
        })
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35PackedW8LinearLayerReference {
    fn pack(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        layer_index: usize,
        profile: Qwen35PackedW8LinearLayerReferenceProfile,
    ) -> Result<Self> {
        let packed = pack_qwen35_w8_linear_layer(weights, config, layer_index, profile)?;
        let quantization = packed.packed.quantization_ledger().map_err(|error| {
            Error::Other(format!(
                "qwen3.5 packed W8 linear layer {layer_index} precision ledger failed: {error}"
            ))
        })?;
        Ok(Self {
            layer_index,
            profile,
            quantization,
            dimensions: packed.dimensions,
            packed: packed.packed,
            state: None,
            prefill_seed_calls: 0,
            decode_calls: 0,
            successful_decodes: 0,
            failed_decodes: 0,
            committed_state_version: 0,
            terminal_error: false,
            block_elapsed_ns: 0,
            #[cfg(all(test, debug_assertions))]
            fail_next_decode_after_reference: false,
        })
    }

    fn seed_after_cpu_prefill(&mut self, state: &apxinf_metal::GdnDecodeState) {
        self.state = Some(state.clone());
        self.prefill_seed_calls = self.prefill_seed_calls.saturating_add(1);
        self.decode_calls = 0;
        self.successful_decodes = 0;
        self.failed_decodes = 0;
        self.committed_state_version = 0;
        self.terminal_error = false;
        self.block_elapsed_ns = 0;
    }

    fn stats(&self) -> Qwen35PackedW8LinearLayerReferenceStats {
        Qwen35PackedW8LinearLayerReferenceStats {
            layer_index: self.layer_index,
            profile: self.profile,
            quantization: self.quantization,
            prefill_seed_calls: self.prefill_seed_calls,
            decode_calls: self.decode_calls,
            successful_decodes: self.successful_decodes,
            failed_decodes: self.failed_decodes,
            committed_state_version: self.committed_state_version,
            terminal_error: self.terminal_error,
            block_elapsed_ns: self.block_elapsed_ns,
        }
    }

    fn reset(&mut self) {
        self.state = None;
        self.prefill_seed_calls = 0;
        self.decode_calls = 0;
        self.successful_decodes = 0;
        self.failed_decodes = 0;
        self.committed_state_version = 0;
        self.terminal_error = false;
        self.block_elapsed_ns = 0;
        #[cfg(all(test, debug_assertions))]
        {
            self.fail_next_decode_after_reference = false;
        }
    }
}

#[cfg(feature = "metal-experiments")]
fn gdn_decode_state_from_cpu(
    config: &Qwen35TextConfig,
    state: &Qwen35LinearState,
) -> Result<apxinf_metal::GdnDecodeState> {
    let [query, key, value] = state.convolution_suffixes();
    let tensor_values = |tensor: Option<&Tensor>, label: &str| -> Result<Vec<f32>> {
        let tensor = tensor.ok_or_else(|| {
            Error::Other(format!(
                "qwen3.5 Metal W8 GDN CPU prefill did not produce {label} state"
            ))
        })?;
        Ok(tensor.as_f32()?.to_vec())
    };
    let dimensions = apxinf_metal::GdnDimensions {
        hidden_size: config.hidden_size,
        key_heads: config.linear_num_key_heads,
        value_heads: config.linear_num_value_heads,
        key_dim: config.linear_key_head_dim,
        value_dim: config.linear_value_head_dim,
        conv_kernel_size: config.linear_conv_kernel_dim,
        rms_norm_eps: config.rms_norm_eps,
    };
    apxinf_metal::GdnDecodeState::from_parts(
        dimensions,
        tensor_values(query, "query convolution")?,
        tensor_values(key, "key convolution")?,
        tensor_values(value, "value convolution")?,
        tensor_values(state.recurrent(), "recurrent")?,
    )
    .map_err(|error| Error::Other(format!("qwen3.5 Metal W8 GDN state seed: {error}")))
}

#[cfg(feature = "metal-experiments")]
fn run_linear_attention_with_metal_w8_gdn(
    hidden: &Tensor,
    gdn: &mut Qwen35MetalW8GdnLayer,
) -> Result<Tensor> {
    if hidden.shape().dims() != [1, gdn.dimensions.hidden_size] {
        return Err(Error::ShapeMismatch {
            expected: format!("[1, {}]", gdn.dimensions.hidden_size),
            got: hidden.shape().to_string(),
        });
    }
    let input = hidden.as_f32()?;
    #[cfg(all(test, debug_assertions))]
    if std::mem::take(&mut gdn.fail_next_decode_after_scratch) {
        return match gdn
            .block
            .inject_failure_after_scratch_execution_for_testing(input)
        {
            Err(error) => Err(Error::Other(format!(
                "qwen3.5 Metal W8 GDN layer {} failed terminally: {error}",
                gdn.layer_index
            ))),
            Ok(()) => Err(Error::Other(format!(
                "qwen3.5 Metal W8 GDN layer {} fault injection unexpectedly succeeded",
                gdn.layer_index
            ))),
        };
    }
    let started = std::time::Instant::now();
    let output = gdn
        .block
        .decode(input)
        .map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 GDN layer {} failed terminally: {error}",
                gdn.layer_index
            ))
        })?
        .to_vec();
    let tensor = Tensor::from_f32(vec![1, gdn.dimensions.hidden_size], &output)?;
    gdn.block_elapsed_ns = gdn
        .block_elapsed_ns
        .saturating_add(started.elapsed().as_nanos());
    Ok(tensor)
}

#[cfg(feature = "metal-experiments")]
fn run_linear_layer_with_metal_w8(
    hidden: &Tensor,
    linear_layer: &mut Qwen35MetalW8LinearLayer,
) -> Result<Tensor> {
    if linear_layer.terminal_error {
        return Err(Error::Other(format!(
            "qwen3.5 Metal W8 linear layer {} is terminal after a decode error; reset required",
            linear_layer.layer_index
        )));
    }
    if hidden.shape().dims() != [1, linear_layer.dimensions.hidden_size] {
        return Err(Error::ShapeMismatch {
            expected: format!("[1, {}]", linear_layer.dimensions.hidden_size),
            got: hidden.shape().to_string(),
        });
    }
    if !linear_layer.seeded {
        linear_layer.terminal_error = true;
        return Err(Error::Other(format!(
            "qwen3.5 Metal W8 linear layer {} decode state was not seeded by CPU prefill; reset required",
            linear_layer.layer_index
        )));
    }
    let input = hidden.as_f32()?;
    let started = std::time::Instant::now();

    #[cfg(all(test, debug_assertions))]
    if std::mem::take(&mut linear_layer.fail_next_decode_after_scratch) {
        let execution = linear_layer
            .block
            .inject_failure_after_scratch_execution_for_testing(input);
        linear_layer.block_elapsed_ns = linear_layer
            .block_elapsed_ns
            .saturating_add(started.elapsed().as_nanos());
        linear_layer.terminal_error = true;
        return match execution {
            Err(error) => Err(Error::Other(format!(
                "qwen3.5 Metal W8 linear layer {} failed terminally: {error}; reset required",
                linear_layer.layer_index
            ))),
            Ok(()) => Err(Error::Other(format!(
                "qwen3.5 Metal W8 linear layer {} fault injection unexpectedly succeeded; reset required",
                linear_layer.layer_index
            ))),
        };
    }

    let output = match linear_layer.block.decode(input) {
        Ok(output) => output.to_vec(),
        Err(error) => {
            linear_layer.block_elapsed_ns = linear_layer
                .block_elapsed_ns
                .saturating_add(started.elapsed().as_nanos());
            linear_layer.terminal_error = true;
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 linear layer {} failed terminally: {error}; reset required",
                linear_layer.layer_index
            )));
        }
    };
    let tensor = Tensor::from_f32(vec![1, linear_layer.dimensions.hidden_size], &output).map_err(
        |error| {
            linear_layer.terminal_error = true;
            Error::Other(format!(
                "qwen3.5 Metal W8 linear layer {} produced an invalid output terminally: {error}; reset required",
                linear_layer.layer_index
            ))
        },
    )?;
    linear_layer.block_elapsed_ns = linear_layer
        .block_elapsed_ns
        .saturating_add(started.elapsed().as_nanos());
    Ok(tensor)
}

#[cfg(feature = "metal-experiments")]
fn run_linear_layer_stack3_with_metal_w8(
    hidden: &Tensor,
    stack: &mut Qwen35MetalW8LinearLayerStack3V1,
) -> Result<Tensor> {
    if stack.terminal_error {
        return Err(Error::Other(format!(
            "qwen3.5 Metal W8 stack3-v1 {:?} is terminal after a decode error; reset required",
            stack.layer_indices
        )));
    }
    if hidden.shape().dims() != [1, stack.dimensions.hidden_size] {
        return Err(Error::ShapeMismatch {
            expected: format!("[1, {}]", stack.dimensions.hidden_size),
            got: hidden.shape().to_string(),
        });
    }
    if !stack.seeded {
        stack.terminal_error = true;
        return Err(Error::Other(format!(
            "qwen3.5 Metal W8 stack3-v1 {:?} was not fully seeded by CPU prefill; reset required",
            stack.layer_indices
        )));
    }
    let input = hidden.as_f32()?;
    let started = std::time::Instant::now();

    #[cfg(all(test, debug_assertions))]
    if std::mem::take(&mut stack.fail_next_decode_after_scratch) {
        let execution = stack
            .block
            .inject_failure_after_scratch_execution_for_testing(input);
        stack.block_elapsed_ns = stack
            .block_elapsed_ns
            .saturating_add(started.elapsed().as_nanos());
        stack.terminal_error = true;
        return match execution {
            Err(error) => Err(Error::Other(format!(
                "qwen3.5 Metal W8 stack3-v1 {:?} failed terminally: {error}; reset required",
                stack.layer_indices
            ))),
            Ok(()) => Err(Error::Other(format!(
                "qwen3.5 Metal W8 stack3-v1 {:?} fault injection unexpectedly succeeded; reset required",
                stack.layer_indices
            ))),
        };
    }

    let output = match stack.block.decode(input) {
        Ok(output) => output.to_vec(),
        Err(error) => {
            stack.block_elapsed_ns = stack
                .block_elapsed_ns
                .saturating_add(started.elapsed().as_nanos());
            stack.terminal_error = true;
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 stack3-v1 {:?} failed terminally: {error}; reset required",
                stack.layer_indices
            )));
        }
    };
    let tensor = Tensor::from_f32(vec![1, stack.dimensions.hidden_size], &output).map_err(
        |error| {
            stack.terminal_error = true;
            Error::Other(format!(
                "qwen3.5 Metal W8 stack3-v1 {:?} produced an invalid final output terminally: {error}; reset required",
                stack.layer_indices
            ))
        },
    )?;
    stack.block_elapsed_ns = stack
        .block_elapsed_ns
        .saturating_add(started.elapsed().as_nanos());
    Ok(tensor)
}

#[cfg(feature = "metal-experiments")]
fn run_mlp_stack3_boundary_with_metal_w8(
    hidden: &Tensor,
    region: &mut Qwen35MetalW8MlpStack3BoundaryRegionV1,
) -> Result<Tensor> {
    if region.terminal_error {
        return Err(Error::Other(format!(
            "qwen3.5 Metal W8 MLP→Stack3 boundary v1 layer {} is terminal after a decode error; reset required",
            region.boundary_mlp_layer_index
        )));
    }
    if hidden.shape().dims() != [1, region.dimensions.hidden_size] {
        return Err(Error::ShapeMismatch {
            expected: format!("[1, {}]", region.dimensions.hidden_size),
            got: hidden.shape().to_string(),
        });
    }
    if !region.seeded {
        region.terminal_error = true;
        return Err(Error::Other(format!(
            "qwen3.5 Metal W8 MLP→Stack3 boundary v1 layer {} was not fully seeded by CPU prefill; reset required",
            region.boundary_mlp_layer_index
        )));
    }
    let input = hidden.as_f32()?;
    let started = std::time::Instant::now();

    #[cfg(all(test, debug_assertions))]
    if std::mem::take(&mut region.fail_next_decode_after_scratch) {
        let execution = region
            .block
            .inject_failure_after_scratch_execution_for_testing(input);
        region.block_elapsed_ns = region
            .block_elapsed_ns
            .saturating_add(started.elapsed().as_nanos());
        region.terminal_error = true;
        return match execution {
            Err(error) => Err(Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary v1 layer {} failed terminally: {error}; reset required",
                region.boundary_mlp_layer_index
            ))),
            Ok(()) => Err(Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary v1 layer {} fault injection unexpectedly succeeded; reset required",
                region.boundary_mlp_layer_index
            ))),
        };
    }

    let output = match region.block.decode(input) {
        Ok(output) => output.to_vec(),
        Err(error) => {
            region.block_elapsed_ns = region
                .block_elapsed_ns
                .saturating_add(started.elapsed().as_nanos());
            region.terminal_error = true;
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary v1 layer {} failed terminally: {error}; reset required",
                region.boundary_mlp_layer_index
            )));
        }
    };
    let tensor = Tensor::from_f32(vec![1, region.dimensions.hidden_size], &output).map_err(
        |error| {
            region.terminal_error = true;
            Error::Other(format!(
                "qwen3.5 Metal W8 MLP→Stack3 boundary v1 layer {} produced an invalid final output terminally: {error}; reset required",
                region.boundary_mlp_layer_index
            ))
        },
    )?;
    region.block_elapsed_ns = region
        .block_elapsed_ns
        .saturating_add(started.elapsed().as_nanos());
    Ok(tensor)
}

#[cfg(feature = "metal-experiments")]
fn run_linear_layer_with_packed_w8_reference(
    hidden: &Tensor,
    reference: &mut Qwen35PackedW8LinearLayerReference,
) -> Result<Tensor> {
    if reference.terminal_error {
        return Err(Error::Other(format!(
            "qwen3.5 packed W8 linear-layer reference {} is terminal after a decode error; reset required",
            reference.layer_index
        )));
    }
    if hidden.shape().dims() != [1, reference.dimensions.hidden_size] {
        return Err(Error::ShapeMismatch {
            expected: format!("[1, {}]", reference.dimensions.hidden_size),
            got: hidden.shape().to_string(),
        });
    }
    let Some(state) = reference.state.as_ref() else {
        reference.terminal_error = true;
        return Err(Error::Other(format!(
            "qwen3.5 packed W8 linear-layer reference {} was not seeded by CPU prefill; reset required",
            reference.layer_index
        )));
    };
    let input = hidden.as_f32()?;
    let started = std::time::Instant::now();
    reference.decode_calls = reference.decode_calls.saturating_add(1);
    let decoded = match reference.packed.decode_reference(input, state) {
        Ok(decoded) => decoded,
        Err(error) => {
            reference.failed_decodes = reference.failed_decodes.saturating_add(1);
            reference.terminal_error = true;
            reference.block_elapsed_ns = reference
                .block_elapsed_ns
                .saturating_add(started.elapsed().as_nanos());
            return Err(Error::Other(format!(
                "qwen3.5 packed W8 linear-layer reference {} failed terminally: {error}; reset required",
                reference.layer_index
            )));
        }
    };

    #[cfg(all(test, debug_assertions))]
    if std::mem::take(&mut reference.fail_next_decode_after_reference) {
        reference.failed_decodes = reference.failed_decodes.saturating_add(1);
        reference.terminal_error = true;
        reference.block_elapsed_ns = reference
            .block_elapsed_ns
            .saturating_add(started.elapsed().as_nanos());
        return Err(Error::Other(format!(
            "qwen3.5 packed W8 linear-layer reference {} injected failure after scratch execution; reset required",
            reference.layer_index
        )));
    }

    let tensor = match Tensor::from_f32(vec![1, reference.dimensions.hidden_size], &decoded.output)
    {
        Ok(tensor) => tensor,
        Err(error) => {
            reference.failed_decodes = reference.failed_decodes.saturating_add(1);
            reference.terminal_error = true;
            reference.block_elapsed_ns = reference
                .block_elapsed_ns
                .saturating_add(started.elapsed().as_nanos());
            return Err(Error::Other(format!(
                "qwen3.5 packed W8 linear-layer reference {} produced an invalid output terminally: {error}; reset required",
                reference.layer_index
            )));
        }
    };
    reference.state = Some(decoded.state);
    reference.successful_decodes = reference.successful_decodes.saturating_add(1);
    reference.committed_state_version = reference.committed_state_version.saturating_add(1);
    reference.block_elapsed_ns = reference
        .block_elapsed_ns
        .saturating_add(started.elapsed().as_nanos());
    Ok(tensor)
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8Body {
    fn pack(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        layer_indices: &[usize],
    ) -> Result<Self> {
        if layer_indices.is_empty() {
            return Err(Error::Other(
                "qwen3.5 Metal W8 body requires at least one selected layer".into(),
            ));
        }
        let mut selected = vec![false; weights.layers.len()];
        for &layer_index in layer_indices {
            let Some(slot) = selected.get_mut(layer_index) else {
                return Err(Error::Other(format!(
                    "qwen3.5 Metal W8 body layer {layer_index} is outside 0..{}",
                    weights.layers.len()
                )));
            };
            if *slot {
                return Err(Error::Other(format!(
                    "qwen3.5 Metal W8 body layer {layer_index} was selected more than once"
                )));
            }
            *slot = true;
        }
        let mut layers = (0..weights.layers.len()).map(|_| None).collect::<Vec<_>>();
        for (layer_index, selected) in selected.into_iter().enumerate() {
            if selected {
                layers[layer_index] =
                    Some(Qwen35MetalW8BodyLayer::pack(weights, config, layer_index)?);
            }
        }
        Ok(Self { layers })
    }

    fn layer_mut(&mut self, layer_index: usize) -> Option<&mut Qwen35MetalW8BodyLayer> {
        self.layers.get_mut(layer_index).and_then(Option::as_mut)
    }

    fn stats(&self) -> Vec<Qwen35MetalW8BodyStats> {
        self.layers
            .iter()
            .flatten()
            .map(|layer| Qwen35MetalW8BodyStats {
                layer_index: layer.layer_index,
                decode_calls: layer.decode_calls,
                projection_elapsed_ns: layer.projection_elapsed_ns,
            })
            .collect()
    }
}

#[cfg(feature = "metal-experiments")]
impl Qwen35MetalW8BodyLayer {
    fn pack(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        layer_index: usize,
    ) -> Result<Self> {
        let layer = weights.layers.get(layer_index).ok_or_else(|| {
            Error::Other(format!(
                "qwen3.5 Metal W8 body layer {layer_index} is outside 0..{}",
                weights.layers.len()
            ))
        })?;
        let expected = [config.intermediate_size, config.hidden_size];
        if layer.mlp.gate_proj_weight.shape().dims() != expected
            || layer.mlp.up_proj_weight.shape().dims() != expected
        {
            return Err(Error::Other(format!(
                "qwen3.5 Metal W8 body layer {layer_index} requires gate/up shape [{}, {}]",
                config.intermediate_size, config.hidden_size
            )));
        }
        let gate = f32_values(&layer.mlp.gate_proj_weight, "Metal W8 MLP gate")?;
        let up = f32_values(&layer.mlp.up_proj_weight, "Metal W8 MLP up")?;
        let element_count = config
            .intermediate_size
            .checked_mul(config.hidden_size)
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| Error::Other("qwen3.5 Metal W8 gate+up dimensions overflow".into()))?;
        let mut stacked = Vec::with_capacity(element_count);
        stacked.extend_from_slice(&gate);
        stacked.extend_from_slice(&up);
        let mlp_gate_up = apxinf_metal::MetalW8MatVec::from_f32_rows(
            &stacked,
            2 * config.intermediate_size,
            config.hidden_size,
        )
        .map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 body layer {layer_index} gate+up packing failed: {error}"
            ))
        })?;
        Ok(Self {
            layer_index,
            mlp_gate_up,
            decode_calls: 0,
            projection_elapsed_ns: 0,
        })
    }
}

#[cfg(feature = "metal-w8-head-mlp")]
impl Qwen35MetalW8MlpBlocks {
    fn pack(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        layer_indices: &[usize],
    ) -> Result<Self> {
        if layer_indices.is_empty() {
            return Err(Error::Other(
                "qwen3.5 Metal W8 MLP block requires at least one selected layer".into(),
            ));
        }
        let mut selected = vec![false; weights.layers.len()];
        for &layer_index in layer_indices {
            let Some(slot) = selected.get_mut(layer_index) else {
                return Err(Error::Other(format!(
                    "qwen3.5 Metal W8 MLP block layer {layer_index} is outside 0..{}",
                    weights.layers.len()
                )));
            };
            if *slot {
                return Err(Error::Other(format!(
                    "qwen3.5 Metal W8 MLP block layer {layer_index} was selected more than once"
                )));
            }
            *slot = true;
        }
        let mut layers = (0..weights.layers.len()).map(|_| None).collect::<Vec<_>>();
        for (layer_index, selected) in selected.into_iter().enumerate() {
            if selected {
                layers[layer_index] = Some(Qwen35MetalW8MlpBlockLayer::pack(
                    weights,
                    config,
                    layer_index,
                )?);
            }
        }
        Ok(Self { layers })
    }

    fn layer_mut(&mut self, layer_index: usize) -> Option<&mut Qwen35MetalW8MlpBlockLayer> {
        self.layers.get_mut(layer_index).and_then(Option::as_mut)
    }

    fn stats(&self) -> Vec<Qwen35MetalW8MlpBlockStats> {
        self.layers
            .iter()
            .flatten()
            .map(|layer| Qwen35MetalW8MlpBlockStats {
                layer_index: layer.layer_index,
                decode_calls: layer.decode_calls,
                block_elapsed_ns: layer.block_elapsed_ns,
            })
            .collect()
    }

    #[cfg(feature = "metal-experiments")]
    fn buffer_ledgers(&self) -> Vec<Qwen35MetalW8MlpBlockBufferLedger> {
        self.layers
            .iter()
            .flatten()
            .map(|layer| Qwen35MetalW8MlpBlockBufferLedger {
                layer_index: layer.layer_index,
                ledger: layer.mlp.buffer_ledger(),
            })
            .collect()
    }

    #[cfg(feature = "metal-experiments")]
    fn reset_stats(&mut self) {
        for layer in self.layers.iter_mut().flatten() {
            layer.decode_calls = 0;
            layer.block_elapsed_ns = 0;
        }
    }
}

#[cfg(feature = "metal-w8-head-mlp")]
fn pack_qwen35_w8_mlp_block(
    weights: &Qwen35TextWeights,
    config: &Qwen35TextConfig,
    layer_index: usize,
) -> Result<apxinf_metal::PackedW8MlpBlock> {
    let layer = weights.layers.get(layer_index).ok_or_else(|| {
        Error::Other(format!(
            "qwen3.5 Metal W8 MLP block layer {layer_index} is outside 0..{}",
            weights.layers.len()
        ))
    })?;
    let gate_up_expected = [config.intermediate_size, config.hidden_size];
    let down_expected = [config.hidden_size, config.intermediate_size];
    if layer.mlp.gate_proj_weight.shape().dims() != gate_up_expected
        || layer.mlp.up_proj_weight.shape().dims() != gate_up_expected
        || layer.mlp.down_proj_weight.shape().dims() != down_expected
    {
        return Err(Error::Other(format!(
            "qwen3.5 Metal W8 MLP block layer {layer_index} requires gate/up [{}, {}] and down [{}, {}]",
            config.intermediate_size,
            config.hidden_size,
            config.hidden_size,
            config.intermediate_size
        )));
    }
    let gate = f32_values(&layer.mlp.gate_proj_weight, "Metal W8 MLP block gate")?;
    let up = f32_values(&layer.mlp.up_proj_weight, "Metal W8 MLP block up")?;
    let down = f32_values(&layer.mlp.down_proj_weight, "Metal W8 MLP block down")?;
    apxinf_metal::PackedW8MlpBlock::pack_f32(
        &gate,
        &up,
        &down,
        config.hidden_size,
        config.intermediate_size,
    )
    .map_err(|error| {
        Error::Other(format!(
            "qwen3.5 Metal W8 MLP block layer {layer_index} packing failed: {error}"
        ))
    })
}

#[cfg(feature = "metal-w8-head-mlp")]
impl Qwen35MetalW8MlpBlockLayer {
    fn pack(
        weights: &Qwen35TextWeights,
        config: &Qwen35TextConfig,
        layer_index: usize,
    ) -> Result<Self> {
        let packed = pack_qwen35_w8_mlp_block(weights, config, layer_index)?;
        let mlp = apxinf_metal::MetalW8MlpBlock::from_packed(&packed).map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 MLP block layer {layer_index} construction failed: {error}"
            ))
        })?;
        Ok(Self {
            layer_index,
            mlp,
            decode_calls: 0,
            block_elapsed_ns: 0,
        })
    }
}

fn run_mlp(backend: &dyn Backend, hidden: &Tensor, weights: &RuntimeMlpWeights) -> Result<Tensor> {
    let gate = backend.silu(&backend.matmul(hidden, &weights.gate_projection)?)?;
    let up = backend.matmul(hidden, &weights.up_projection)?;
    let activated = backend.mul(&gate, &up)?;
    backend.matmul(&activated, &weights.down_projection)
}

#[cfg(feature = "metal-experiments")]
fn run_mlp_with_metal_w8(
    backend: &dyn Backend,
    hidden: &Tensor,
    weights: &RuntimeMlpWeights,
    body: &mut Qwen35MetalW8BodyLayer,
) -> Result<Tensor> {
    let dims = hidden.shape().dims();
    if dims.len() != 2 || dims[0] != 1 || dims[1] != body.mlp_gate_up.columns() {
        return Err(Error::ShapeMismatch {
            expected: format!("[1, {}]", body.mlp_gate_up.columns()),
            got: hidden.shape().to_string(),
        });
    }
    let projection_started = std::time::Instant::now();
    let projected = body
        .mlp_gate_up
        .multiply(hidden.as_f32()?)
        .map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 body layer {} gate+up failed: {error}",
                body.layer_index
            ))
        })?;
    body.projection_elapsed_ns = body
        .projection_elapsed_ns
        .saturating_add(projection_started.elapsed().as_nanos());
    let intermediate = projected.len() / 2;
    if intermediate == 0 || projected.len() != 2 * intermediate {
        return Err(Error::Other(format!(
            "qwen3.5 Metal W8 body layer {} returned invalid gate+up width {}",
            body.layer_index,
            projected.len()
        )));
    }
    body.decode_calls += 1;
    let gate = Tensor::from_f32(vec![1, intermediate], &projected[..intermediate])?;
    let up = Tensor::from_f32(vec![1, intermediate], &projected[intermediate..])?;
    let activated = backend.mul(&backend.silu(&gate)?, &up)?;
    backend.matmul(&activated, &weights.down_projection)
}

#[cfg(feature = "metal-w8-head-mlp")]
fn run_mlp_with_metal_w8_block(
    hidden: &Tensor,
    block: &mut Qwen35MetalW8MlpBlockLayer,
) -> Result<Tensor> {
    let dims = hidden.shape().dims();
    if dims.len() != 2 || dims[0] != 1 || dims[1] != block.mlp.hidden_size() {
        return Err(Error::ShapeMismatch {
            expected: format!("[1, {}]", block.mlp.hidden_size()),
            got: hidden.shape().to_string(),
        });
    }
    let block_started = std::time::Instant::now();
    let hidden_size = block.mlp.hidden_size();
    let output = {
        let projected = block.mlp.forward(hidden.as_f32()?).map_err(|error| {
            Error::Other(format!(
                "qwen3.5 Metal W8 MLP block layer {} failed: {error}",
                block.layer_index
            ))
        })?;
        Tensor::from_f32(vec![1, hidden_size], projected)?
    };
    block.block_elapsed_ns = block
        .block_elapsed_ns
        .saturating_add(block_started.elapsed().as_nanos());
    block.decode_calls += 1;
    Ok(output)
}

struct RuntimeWeights {
    token_embedding: Tensor,
    layers: Vec<RuntimeLayerWeights>,
    output_norm_weight: Tensor,
    /// Untied output projection packed as `[hidden, vocab]`. `None` means the
    /// checkpoint ties its output projection to `token_embedding`, which stays
    /// in `[vocab, hidden]` form and is consumed via `matmul_rhs_transposed`.
    lm_head: Option<Tensor>,
}

impl RuntimeWeights {
    fn pack(weights: Qwen35TextWeights, config: &Qwen35TextConfig) -> Result<Self> {
        let Qwen35TextWeights {
            token_embedding,
            layers,
            output_norm_weight,
            lm_head_weight,
        } = weights;

        let token_embedding = into_cpu_f32(token_embedding, "token embedding")?;
        let lm_head = match lm_head_weight {
            Some(weight) => Some(transpose_rows(
                &weight,
                0,
                config.vocab_size,
                config.hidden_size,
            )?),
            None => None,
        };
        let output_norm_weight = into_cpu_f32(output_norm_weight, "output norm")?;

        let mut runtime_layers = Vec::with_capacity(layers.len());
        for (index, (layer, layer_type)) in layers
            .into_iter()
            .zip(config.layer_types.iter().copied())
            .enumerate()
        {
            runtime_layers.push(RuntimeLayerWeights::pack(layer, layer_type, config, index)?);
        }
        Ok(Self {
            token_embedding,
            layers: runtime_layers,
            output_norm_weight,
            lm_head,
        })
    }
}

struct RuntimeLayerWeights {
    input_norm_weight: Tensor,
    attention: RuntimeAttentionWeights,
    post_attention_norm_weight: Tensor,
    mlp: RuntimeMlpWeights,
}

impl RuntimeLayerWeights {
    fn pack(
        layer: Qwen35LayerWeights,
        layer_type: Qwen35LayerType,
        config: &Qwen35TextConfig,
        index: usize,
    ) -> Result<Self> {
        let Qwen35LayerWeights {
            input_norm_weight,
            attention,
            post_attention_norm_weight,
            mlp,
        } = layer;
        let attention = match (attention, layer_type) {
            (Qwen35AttentionWeights::Linear(weights), Qwen35LayerType::LinearAttention) => {
                RuntimeAttentionWeights::Linear(RuntimeLinearWeights::pack(weights, config)?)
            }
            (Qwen35AttentionWeights::Full(weights), Qwen35LayerType::FullAttention) => {
                RuntimeAttentionWeights::Full(RuntimeFullWeights::pack(weights, config)?)
            }
            _ => {
                return Err(Error::Other(format!(
                    "qwen3.5: checkpoint/config attention mismatch at layer {index}"
                )));
            }
        };
        Ok(Self {
            input_norm_weight: into_cpu_f32(input_norm_weight, "input norm")?,
            attention,
            post_attention_norm_weight: into_cpu_f32(
                post_attention_norm_weight,
                "post-attention norm",
            )?,
            mlp: RuntimeMlpWeights::pack(mlp, config)?,
        })
    }
}

enum RuntimeAttentionWeights {
    Linear(RuntimeLinearWeights),
    Full(RuntimeFullWeights),
}

struct RuntimeLinearWeights {
    a_log: Tensor,
    dt_bias: Tensor,
    query_projection: Tensor,
    key_projection: Tensor,
    value_projection: Tensor,
    z_projection: Tensor,
    a_projection: Tensor,
    b_projection: Tensor,
    query_conv_weight: Tensor,
    key_conv_weight: Tensor,
    value_conv_weight: Tensor,
    norm_weight: Tensor,
    output_projection: Tensor,
}

impl RuntimeLinearWeights {
    fn pack(weights: Qwen35LinearAttentionWeights, config: &Qwen35TextConfig) -> Result<Self> {
        let Qwen35LinearAttentionWeights {
            a_log,
            conv1d_weight,
            dt_bias,
            in_proj_a_weight,
            in_proj_b_weight,
            in_proj_qkv_weight,
            in_proj_z_weight,
            norm_weight,
            out_proj_weight,
        } = weights;
        let key_width = config.linear_key_width();
        let value_width = config.linear_value_width();
        Ok(Self {
            a_log: into_cpu_f32(a_log, "GDN A_log")?,
            dt_bias: into_cpu_f32(dt_bias, "GDN dt_bias")?,
            query_projection: transpose_rows(
                &in_proj_qkv_weight,
                0,
                key_width,
                config.hidden_size,
            )?,
            key_projection: transpose_rows(
                &in_proj_qkv_weight,
                key_width,
                key_width,
                config.hidden_size,
            )?,
            value_projection: transpose_rows(
                &in_proj_qkv_weight,
                2 * key_width,
                value_width,
                config.hidden_size,
            )?,
            z_projection: transpose_rows(&in_proj_z_weight, 0, value_width, config.hidden_size)?,
            a_projection: transpose_rows(
                &in_proj_a_weight,
                0,
                config.linear_num_value_heads,
                config.hidden_size,
            )?,
            b_projection: transpose_rows(
                &in_proj_b_weight,
                0,
                config.linear_num_value_heads,
                config.hidden_size,
            )?,
            query_conv_weight: slice_conv_rows(
                &conv1d_weight,
                0,
                key_width,
                config.linear_conv_kernel_dim,
            )?,
            key_conv_weight: slice_conv_rows(
                &conv1d_weight,
                key_width,
                key_width,
                config.linear_conv_kernel_dim,
            )?,
            value_conv_weight: slice_conv_rows(
                &conv1d_weight,
                2 * key_width,
                value_width,
                config.linear_conv_kernel_dim,
            )?,
            norm_weight: into_cpu_f32(norm_weight, "GDN output norm")?,
            output_projection: transpose_rows(
                &out_proj_weight,
                0,
                config.hidden_size,
                value_width,
            )?,
        })
    }
}

struct RuntimeFullWeights {
    query_projection: Tensor,
    gate_projection: Option<Tensor>,
    key_projection: Tensor,
    value_projection: Tensor,
    output_projection: Tensor,
    query_norm_weight: Tensor,
    key_norm_weight: Tensor,
}

impl RuntimeFullWeights {
    fn pack(weights: Qwen35FullAttentionWeights, config: &Qwen35TextConfig) -> Result<Self> {
        let Qwen35FullAttentionWeights {
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            o_proj_weight,
            q_norm_weight,
            k_norm_weight,
        } = weights;
        let (query_projection, gate_projection) = if config.attn_output_gate {
            let (query, gate) = transpose_interleaved_q_gate(
                &q_proj_weight,
                config.n_attention_heads,
                config.head_dim,
                config.hidden_size,
            )?;
            (query, Some(gate))
        } else {
            (
                transpose_rows(
                    &q_proj_weight,
                    0,
                    config.full_query_width(),
                    config.hidden_size,
                )?,
                None,
            )
        };
        Ok(Self {
            query_projection,
            gate_projection,
            key_projection: transpose_rows(
                &k_proj_weight,
                0,
                config.full_kv_width(),
                config.hidden_size,
            )?,
            value_projection: transpose_rows(
                &v_proj_weight,
                0,
                config.full_kv_width(),
                config.hidden_size,
            )?,
            output_projection: transpose_rows(
                &o_proj_weight,
                0,
                config.hidden_size,
                config.full_query_width(),
            )?,
            query_norm_weight: into_cpu_f32(q_norm_weight, "attention Q norm")?,
            key_norm_weight: into_cpu_f32(k_norm_weight, "attention K norm")?,
        })
    }
}

struct RuntimeMlpWeights {
    gate_projection: Tensor,
    up_projection: Tensor,
    down_projection: Tensor,
}

impl RuntimeMlpWeights {
    fn pack(weights: Qwen35MlpWeights, config: &Qwen35TextConfig) -> Result<Self> {
        Ok(Self {
            gate_projection: transpose_rows(
                &weights.gate_proj_weight,
                0,
                config.intermediate_size,
                config.hidden_size,
            )?,
            up_projection: transpose_rows(
                &weights.up_proj_weight,
                0,
                config.intermediate_size,
                config.hidden_size,
            )?,
            down_projection: transpose_rows(
                &weights.down_proj_weight,
                0,
                config.hidden_size,
                config.intermediate_size,
            )?,
        })
    }
}

fn f32_values<'a>(tensor: &'a Tensor, label: &str) -> Result<Cow<'a, [f32]>> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other(format!(
            "qwen3.5 {label}: checkpoint tensor must be on CPU before packing"
        )));
    }
    match tensor.dtype() {
        DType::F32 => Ok(Cow::Borrowed(tensor.as_f32()?)),
        DType::F16 | DType::BF16 => Ok(Cow::Owned(tensor.to_f32_vec()?)),
        dtype => Err(Error::Other(format!(
            "qwen3.5 {label}: unsupported dtype {dtype}"
        ))),
    }
}

fn into_cpu_f32(tensor: Tensor, label: &str) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other(format!(
            "qwen3.5 {label}: checkpoint tensor must be on CPU"
        )));
    }
    if tensor.dtype() == DType::F32 {
        return Ok(tensor);
    }
    let shape = tensor.shape().dims().to_vec();
    let values = tensor.to_f32_vec()?;
    Tensor::from_f32(shape, &values)
}

/// Select a contiguous range of HF output rows and physically transpose it to
/// `[in, selected_out]` for the backend matmul contract.
fn transpose_rows(
    tensor: &Tensor,
    row_start: usize,
    row_count: usize,
    expected_input: usize,
) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2
        || dims[1] != expected_input
        || row_start.checked_add(row_count).is_none()
        || row_start + row_count > dims[0]
    {
        return Err(Error::ShapeMismatch {
            expected: format!(
                "[at least {}, {expected_input}] with rows {row_start}..{}",
                row_start.saturating_add(row_count),
                row_start.saturating_add(row_count)
            ),
            got: tensor.shape().to_string(),
        });
    }
    let source = f32_values(tensor, "linear weight")?;
    let mut output = vec![0.0f32; expected_input * row_count];
    for input in 0..expected_input {
        for output_index in 0..row_count {
            output[input * row_count + output_index] =
                source[(row_start + output_index) * expected_input + input];
        }
    }
    Tensor::from_f32(vec![expected_input, row_count], &output)
}

/// Qwen3.5 stores `q_proj` rows per head as `[query_head, gate_head]`.
/// Deinterleave those blocks before transposing; a global half split is wrong.
fn transpose_interleaved_q_gate(
    tensor: &Tensor,
    heads: usize,
    head_dim: usize,
    hidden_size: usize,
) -> Result<(Tensor, Tensor)> {
    let expected_rows = heads
        .checked_mul(head_dim)
        .and_then(|width| width.checked_mul(2))
        .ok_or_else(|| Error::Other("qwen3.5 q_proj dimensions overflow".into()))?;
    if tensor.shape().dims() != [expected_rows, hidden_size] {
        return Err(Error::ShapeMismatch {
            expected: format!("[{expected_rows}, {hidden_size}]"),
            got: tensor.shape().to_string(),
        });
    }
    let source = f32_values(tensor, "gated q_proj")?;
    let query_width = heads * head_dim;
    let mut query = vec![0.0f32; hidden_size * query_width];
    let mut gate = vec![0.0f32; hidden_size * query_width];
    for head in 0..heads {
        let source_head = head * 2 * head_dim;
        let target_head = head * head_dim;
        for dim in 0..head_dim {
            for input in 0..hidden_size {
                query[input * query_width + target_head + dim] =
                    source[(source_head + dim) * hidden_size + input];
                gate[input * query_width + target_head + dim] =
                    source[(source_head + head_dim + dim) * hidden_size + input];
            }
        }
    }
    Ok((
        Tensor::from_f32(vec![hidden_size, query_width], &query)?,
        Tensor::from_f32(vec![hidden_size, query_width], &gate)?,
    ))
}

fn slice_conv_rows(
    tensor: &Tensor,
    row_start: usize,
    row_count: usize,
    kernel_size: usize,
) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 3
        || dims[1] != 1
        || dims[2] != kernel_size
        || row_start.checked_add(row_count).is_none()
        || row_start + row_count > dims[0]
    {
        return Err(Error::ShapeMismatch {
            expected: format!(
                "[at least {}, 1, {kernel_size}] with rows {row_start}..{}",
                row_start.saturating_add(row_count),
                row_start.saturating_add(row_count)
            ),
            got: tensor.shape().to_string(),
        });
    }
    let source = f32_values(tensor, "depthwise convolution weight")?;
    let start = row_start * kernel_size;
    let end = start + row_count * kernel_size;
    Tensor::from_f32(vec![row_count, 1, kernel_size], &source[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::config::tests::MINI_CONFIG;
    use crate::qwen35::Qwen35WeightSchema;

    fn tensor_values(name: &str, count: usize) -> Vec<f32> {
        if name.ends_with("linear_attn.A_log") {
            return vec![-1.25; count];
        }
        if name.ends_with("linear_attn.dt_bias") {
            return vec![-0.5; count];
        }
        if name.ends_with("linear_attn.norm.weight") {
            return vec![0.9; count];
        }
        let salt = name
            .bytes()
            .fold(0u32, |sum, byte| sum.wrapping_add(byte as u32));
        (0..count)
            .map(|index| {
                let phase = ((index as u32).wrapping_mul(17).wrapping_add(salt) % 29) as f32;
                (phase - 14.0) * 0.0025
            })
            .collect()
    }

    fn fixture() -> (Qwen35Config, HashMap<String, Tensor>) {
        let config = Qwen35Config::from_json_str(MINI_CONFIG).unwrap();
        let schema = Qwen35WeightSchema::new(&config).unwrap();
        let tensors = schema
            .specs()
            .iter()
            .map(|spec| {
                let count = spec.shape.iter().product();
                (
                    spec.name.clone(),
                    Tensor::from_f32(spec.shape.clone(), &tensor_values(&spec.name, count))
                        .unwrap(),
                )
            })
            .collect();
        (config, tensors)
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    fn metal_width_fixture() -> (Qwen35Config, HashMap<String, Tensor>) {
        let raw = MINI_CONFIG
            .replacen("\"hidden_size\": 8", "\"hidden_size\": 64", 1)
            .replacen("\"out_hidden_size\": 8", "\"out_hidden_size\": 64", 1);
        let config = Qwen35Config::from_json_str(&raw).unwrap();
        let schema = Qwen35WeightSchema::new(&config).unwrap();
        let tensors = schema
            .specs()
            .iter()
            .map(|spec| {
                let count = spec.shape.iter().product();
                (
                    spec.name.clone(),
                    Tensor::from_f32(spec.shape.clone(), &tensor_values(&spec.name, count))
                        .unwrap(),
                )
            })
            .collect();
        (config, tensors)
    }

    #[cfg(all(feature = "metal-w8-head-mlp", target_os = "macos"))]
    fn metal_mlp_block_fixture() -> (Qwen35Config, HashMap<String, Tensor>) {
        let raw = MINI_CONFIG
            .replacen("\"hidden_size\": 8", "\"hidden_size\": 64", 1)
            .replacen("\"intermediate_size\": 12", "\"intermediate_size\": 64", 1)
            .replacen("\"out_hidden_size\": 8", "\"out_hidden_size\": 64", 1);
        let config = Qwen35Config::from_json_str(&raw).unwrap();
        let schema = Qwen35WeightSchema::new(&config).unwrap();
        let tensors = schema
            .specs()
            .iter()
            .map(|spec| {
                let count = spec.shape.iter().product();
                (
                    spec.name.clone(),
                    Tensor::from_f32(spec.shape.clone(), &tensor_values(&spec.name, count))
                        .unwrap(),
                )
            })
            .collect();
        (config, tensors)
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    fn metal_gdn_fixture() -> (Qwen35Config, HashMap<String, Tensor>) {
        let raw = MINI_CONFIG
            .replacen("\"hidden_size\": 8", "\"hidden_size\": 64", 1)
            .replacen(
                "\"linear_key_head_dim\": 4",
                "\"linear_key_head_dim\": 32",
                1,
            )
            .replacen(
                "\"linear_value_head_dim\": 4",
                "\"linear_value_head_dim\": 32",
                1,
            )
            .replacen("\"out_hidden_size\": 8", "\"out_hidden_size\": 64", 1);
        let config = Qwen35Config::from_json_str(&raw).unwrap();
        let schema = Qwen35WeightSchema::new(&config).unwrap();
        let tensors = schema
            .specs()
            .iter()
            .map(|spec| {
                let count = spec.shape.iter().product();
                (
                    spec.name.clone(),
                    Tensor::from_f32(spec.shape.clone(), &tensor_values(&spec.name, count))
                        .unwrap(),
                )
            })
            .collect();
        (config, tensors)
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    fn metal_linear_layer_fixture() -> (Qwen35Config, HashMap<String, Tensor>) {
        let raw = MINI_CONFIG
            .replacen("\"hidden_size\": 8", "\"hidden_size\": 64", 1)
            .replacen("\"intermediate_size\": 12", "\"intermediate_size\": 64", 1)
            .replacen(
                "\"linear_key_head_dim\": 4",
                "\"linear_key_head_dim\": 32",
                1,
            )
            .replacen(
                "\"linear_value_head_dim\": 4",
                "\"linear_value_head_dim\": 32",
                1,
            )
            .replacen("\"out_hidden_size\": 8", "\"out_hidden_size\": 64", 1);
        let config = Qwen35Config::from_json_str(&raw).unwrap();
        let schema = Qwen35WeightSchema::new(&config).unwrap();
        let tensors = schema
            .specs()
            .iter()
            .map(|spec| {
                let count = spec.shape.iter().product();
                (
                    spec.name.clone(),
                    Tensor::from_f32(spec.shape.clone(), &tensor_values(&spec.name, count))
                        .unwrap(),
                )
            })
            .collect();
        (config, tensors)
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    fn metal_all_linear_layers_fixture() -> (Qwen35Config, HashMap<String, Tensor>) {
        let layer_types = (0..24)
            .map(|layer_index| {
                if layer_index % 4 == 3 {
                    "\"full_attention\""
                } else {
                    "\"linear_attention\""
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let raw = MINI_CONFIG
            .replacen("\"hidden_size\": 8", "\"hidden_size\": 64", 1)
            .replacen("\"intermediate_size\": 12", "\"intermediate_size\": 64", 1)
            .replacen(
                "\"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\"",
                &layer_types,
                1,
            )
            .replacen("\"linear_key_head_dim\": 4", "\"linear_key_head_dim\": 32", 1)
            .replacen(
                "\"linear_value_head_dim\": 4",
                "\"linear_value_head_dim\": 32",
                1,
            )
            .replacen("\"num_hidden_layers\": 4", "\"num_hidden_layers\": 24", 1)
            .replacen("\"out_hidden_size\": 8", "\"out_hidden_size\": 64", 1);
        let config = Qwen35Config::from_json_str(&raw).unwrap();
        let schema = Qwen35WeightSchema::new(&config).unwrap();
        let tensors = schema
            .specs()
            .iter()
            .map(|spec| {
                let count = spec.shape.iter().product();
                (
                    spec.name.clone(),
                    Tensor::from_f32(spec.shape.clone(), &tensor_values(&spec.name, count))
                        .unwrap(),
                )
            })
            .collect();
        (config, tensors)
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) {
        assert_eq!(actual.shape(), expected.shape());
        let actual = actual.as_f32().unwrap();
        let expected = expected.as_f32().unwrap();
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= tolerance,
                "element {index}: actual={actual}, expected={expected}, error={error}"
            );
        }
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    fn assert_gdn_seed_matches_cpu(model: &GeneralQwen35, layer_index: usize) {
        let cpu = model.state.linear_state(layer_index).unwrap();
        let metal = model
            .metal_w8_gdn
            .as_ref()
            .unwrap()
            .block
            .state_snapshot()
            .unwrap();
        let [query, key, value] = cpu.convolution_suffixes();
        assert_eq!(metal.query_conv(), query.unwrap().as_f32().unwrap());
        assert_eq!(metal.key_conv(), key.unwrap().as_f32().unwrap());
        assert_eq!(metal.value_conv(), value.unwrap().as_f32().unwrap());
        assert_eq!(
            metal.recurrent(),
            cpu.recurrent().unwrap().as_f32().unwrap()
        );
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_w8_gdn_is_explicit_seeds_exactly_and_tracks_two_decode_steps() {
        let (config, tensors) = metal_gdn_fixture();
        let mut cpu =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        assert!(cpu.metal_w8_gdn_stats().is_none());
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_gdn_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();

        let cpu_prefill = cpu.forward_hidden(&[1, 2], 0).unwrap();
        let diagnostic_prefill = diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        assert_close(&diagnostic_prefill, &cpu_prefill, 1.0e-7);
        assert_gdn_seed_matches_cpu(&diagnostic, 0);
        let prefill_stats = diagnostic.metal_w8_gdn_stats().unwrap();
        assert_eq!(prefill_stats.prefill_seed_calls, 1);
        assert_eq!(prefill_stats.decode_calls, 0);
        assert_eq!(prefill_stats.command_buffers, 0);
        assert_eq!(prefill_stats.waits, 0);

        for (token, position) in [(3, 2), (4, 3)] {
            let cpu_hidden = cpu.forward_hidden(&[token], position).unwrap();
            let diagnostic_hidden = diagnostic.forward_hidden(&[token], position).unwrap();
            assert_close(&diagnostic_hidden, &cpu_hidden, 3.0e-3);
        }
        let stats = diagnostic.metal_w8_gdn_stats().unwrap();
        assert_eq!(stats.layer_index, 0);
        assert_eq!(stats.prefill_seed_calls, 1);
        assert_eq!(stats.decode_calls, 2);
        assert_eq!(stats.command_buffers, 2);
        assert_eq!(stats.waits, 2);
        assert_eq!(stats.committed_state_version, 2);
        assert!(stats.block_elapsed_ns > 0);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_w8_gdn_reset_clears_state_and_requires_a_new_prefill_seed() {
        let (config, tensors) = metal_gdn_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_gdn_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[3], 2).unwrap();
        assert_eq!(diagnostic.state.position(), 3);

        diagnostic.reset();

        assert_eq!(diagnostic.state.position(), 0);
        assert!(diagnostic
            .metal_w8_gdn
            .as_ref()
            .unwrap()
            .block
            .state_snapshot()
            .is_err());
        assert_eq!(
            diagnostic.metal_w8_gdn_stats().unwrap(),
            Qwen35MetalW8GdnStats {
                layer_index: 0,
                prefill_seed_calls: 0,
                decode_calls: 0,
                command_buffers: 0,
                waits: 0,
                committed_state_version: 0,
                block_elapsed_ns: 0,
            }
        );
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        assert_gdn_seed_matches_cpu(&diagnostic, 0);
        assert_eq!(
            diagnostic.metal_w8_gdn_stats().unwrap().prefill_seed_calls,
            1
        );
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_w8_gdn_error_is_terminal_and_does_not_commit_or_advance() {
        let (config, tensors) = metal_gdn_fixture();
        let mut cpu =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_gdn_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();
        cpu.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        let before_state = diagnostic
            .metal_w8_gdn
            .as_ref()
            .unwrap()
            .block
            .state_snapshot()
            .unwrap();
        let before_stats = diagnostic.metal_w8_gdn_stats().unwrap();
        diagnostic.inject_metal_w8_gdn_failure_once_for_test();

        let error = diagnostic.forward_hidden(&[3], 2).unwrap_err();

        assert!(error.to_string().contains("injected"));
        assert_eq!(diagnostic.state.position(), 2);
        assert_eq!(diagnostic.metal_w8_gdn_stats().unwrap(), before_stats);
        assert_eq!(
            diagnostic
                .metal_w8_gdn
                .as_ref()
                .unwrap()
                .block
                .state_snapshot()
                .unwrap(),
            before_state
        );

        let cpu_hidden = cpu.forward_hidden(&[3], 2).unwrap();
        let diagnostic_hidden = diagnostic.forward_hidden(&[3], 2).unwrap();
        assert_close(&diagnostic_hidden, &cpu_hidden, 3.0e-3);
        assert_eq!(diagnostic.state.position(), 3);
        assert_eq!(diagnostic.metal_w8_gdn_stats().unwrap().decode_calls, 1);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_w8_gdn_constructor_rejects_full_attention_selection() {
        let (config, tensors) = metal_gdn_fixture();
        let error = GeneralQwen35::from_weights_with_metal_w8_gdn_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            3,
        )
        .err()
        .expect("a full-attention layer cannot silently become a GDN lane");
        assert!(error.to_string().contains("not linear attention"));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_w8_linear_layer_is_explicit_seeds_prefill_and_owns_two_decode_steps() {
        let (config, tensors) = metal_linear_layer_fixture();
        let mut cpu =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        assert!(cpu.metal_w8_linear_layer_stats().is_none());
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_linear_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();

        let cpu_prefill = cpu.forward_hidden(&[1, 2], 0).unwrap();
        let diagnostic_prefill = diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        assert_close(&diagnostic_prefill, &cpu_prefill, 1.0e-7);
        let cpu_state_after_prefill = gdn_decode_state_from_cpu(
            &diagnostic.config.text,
            diagnostic.state.linear_state(0).unwrap(),
        )
        .unwrap();
        assert_eq!(
            diagnostic
                .metal_w8_linear_layer
                .as_ref()
                .unwrap()
                .block
                .state_snapshot()
                .unwrap(),
            cpu_state_after_prefill
        );
        let prefill = diagnostic.metal_w8_linear_layer_stats().unwrap();
        assert_eq!(prefill.prefill_seed_calls, 1);
        assert_eq!(prefill.decode_calls, 0);
        assert_eq!(prefill.command_buffers, 0);
        assert_eq!(prefill.waits, 0);

        for (token, position) in [(3, 2), (4, 3)] {
            let cpu_hidden = cpu.forward_hidden(&[token], position).unwrap();
            let diagnostic_hidden = diagnostic.forward_hidden(&[token], position).unwrap();
            assert_close(&diagnostic_hidden, &cpu_hidden, 1.0e-2);
        }

        assert_eq!(
            gdn_decode_state_from_cpu(
                &diagnostic.config.text,
                diagnostic.state.linear_state(0).unwrap(),
            )
            .unwrap(),
            cpu_state_after_prefill,
            "selected CPU layer state must remain frozen after Metal takes ownership"
        );
        let stats = diagnostic.metal_w8_linear_layer_stats().unwrap();
        assert_eq!(stats.layer_index, 0);
        assert_eq!(stats.prefill_seed_calls, 1);
        assert_eq!(stats.decode_calls, 2);
        assert_eq!(stats.successful_decodes, 2);
        assert_eq!(stats.failed_decodes, 0);
        assert_eq!(stats.command_buffers, 2);
        assert_eq!(stats.compute_encoders, 2);
        assert_eq!(stats.commits, 2);
        assert_eq!(stats.waits, 2);
        assert_eq!(stats.host_to_device_bytes, 2 * 64 * 4);
        assert_eq!(stats.device_to_host_bytes, 2 * 64 * 4);
        assert_eq!(stats.committed_state_version, 2);
        assert!(!stats.terminal_error);
        assert!(stats.block_elapsed_ns > 0);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_precision_v2_gdn_out_g32_is_explicit_and_reports_its_exact_mechanism() {
        let (config, tensors) = metal_linear_layer_fixture();
        let mut cpu =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        assert!(cpu.metal_w8_linear_layer_precision_v2_stats().is_none());
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_linear_layer_precision_v2(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
            Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2,
        )
        .unwrap();

        let cpu_prefill = cpu.forward_hidden(&[1, 2], 0).unwrap();
        let diagnostic_prefill = diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        assert_close(&diagnostic_prefill, &cpu_prefill, 1.0e-2);
        let prefill = diagnostic
            .metal_w8_linear_layer_precision_v2_stats()
            .unwrap();
        assert_eq!(
            prefill.profile,
            Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2
        );
        assert_eq!(prefill.mechanism, "metal-w8-linear-layer-precision-v2");
        assert_eq!(
            prefill.quantization.gdn_input_group_size,
            apxinf_metal::W8GroupSize::G64
        );
        assert_eq!(
            prefill.quantization.gdn_output_group_size,
            apxinf_metal::W8GroupSize::G32
        );
        assert_eq!(
            prefill.quantization.mlp_down_group_size,
            apxinf_metal::W8GroupSize::G64
        );
        assert_eq!(prefill.execution.prefill_seed_calls, 1);
        assert_eq!(prefill.execution.decode_calls, 0);

        let cpu_hidden = cpu.forward_hidden(&[3], 2).unwrap();
        let diagnostic_hidden = diagnostic.forward_hidden(&[3], 2).unwrap();
        assert_close(&diagnostic_hidden, &cpu_hidden, 1.0e-2);
        let decoded = diagnostic
            .metal_w8_linear_layer_precision_v2_stats()
            .unwrap();
        assert_eq!(decoded.execution.decode_calls, 1);
        assert_eq!(decoded.execution.successful_decodes, 1);
        assert_eq!(decoded.execution.command_buffers, 1);
        assert_eq!(decoded.execution.compute_encoders, 1);
        assert_eq!(decoded.execution.commits, 1);
        assert_eq!(decoded.execution.waits, 1);
        assert_eq!(decoded.execution.committed_state_version, 1);
        assert!(!decoded.execution.terminal_error);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_precision_v2_all_linear_layers_hit_18_without_duplicate_mlp_selection() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let ordinary =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        assert!(ordinary
            .metal_w8_all_linear_layers_precision_v2_stats()
            .is_none());
        assert!(ordinary.metal_w8_mlp_block_layer_stats().is_empty());
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_all_linear_layers_precision_v2(
                config,
                tensors,
                Device::Cpu,
                16,
                Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2,
            )
            .unwrap();
        let linear_indices = (0..24)
            .filter(|layer_index| layer_index % 4 != 3)
            .collect::<Vec<_>>();
        let full_indices = (0..24)
            .filter(|layer_index| layer_index % 4 == 3)
            .collect::<Vec<_>>();

        let initial = diagnostic
            .metal_w8_all_linear_layers_precision_v2_stats()
            .unwrap();
        assert_eq!(
            initial.profile,
            Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2
        );
        assert_eq!(initial.mechanism, "metal-w8-all-linear-layers-precision-v2");
        assert_eq!(
            initial.full_attention_mlp_mechanism,
            "metal-w8-mlp-block-g64"
        );
        assert_eq!(initial.linear_layers.len(), 18);
        assert_eq!(
            initial
                .linear_layers
                .iter()
                .map(|stats| stats.execution.layer_index)
                .collect::<Vec<_>>(),
            linear_indices
        );
        assert_eq!(
            initial
                .full_attention_mlp_layers
                .iter()
                .map(|stats| stats.layer_index)
                .collect::<Vec<_>>(),
            full_indices
        );
        assert!(initial
            .linear_layers
            .iter()
            .all(|stats| stats.execution.decode_calls == 0
                && stats.quantization.gdn_input_group_size == apxinf_metal::W8GroupSize::G64
                && stats.quantization.gdn_output_group_size == apxinf_metal::W8GroupSize::G32
                && stats.quantization.mlp_gate_group_size == apxinf_metal::W8GroupSize::G64
                && stats.quantization.mlp_up_group_size == apxinf_metal::W8GroupSize::G64
                && stats.quantization.mlp_down_group_size == apxinf_metal::W8GroupSize::G64));
        assert!(initial
            .full_attention_mlp_layers
            .iter()
            .all(|stats| stats.decode_calls == 0));
        assert!(!initial.terminal_error);
        let generation_receipt = diagnostic.generation_path_receipt().unwrap();
        assert_eq!(
            generation_receipt["format"],
            "apxinf-qwen35-all-linear-layers-generation-path-v2"
        );
        assert_eq!(generation_receipt["profile"], "gdn-out-g32-v2");
        assert_eq!(
            generation_receipt["linear_layers"]
                .as_array()
                .unwrap()
                .len(),
            18
        );
        assert_eq!(
            generation_receipt["full_attention_mlp_layers"]
                .as_array()
                .unwrap()
                .len(),
            6
        );
        assert_eq!(generation_receipt["terminal_error"], false);
        let ledgers = diagnostic
            .metal_w8_all_linear_layers_precision_v2_buffer_ledgers()
            .unwrap();
        assert_eq!(ledgers.len(), 18);
        assert_eq!(
            ledgers
                .iter()
                .map(|entry| entry.layer_index)
                .collect::<Vec<_>>(),
            linear_indices
        );
        assert!(ledgers.iter().all(|entry| {
            entry.ledger.allocated_buffers == 32
                && entry.ledger.command_buffers_per_decode == 1
                && entry.ledger.compute_encoders_per_decode == 1
                && entry.ledger.commits_per_decode == 1
                && entry.ledger.waits_per_decode == 1
                && entry.ledger.state_host_transfer_bytes_per_decode == 0
        }));
        let aggregate = diagnostic
            .metal_w8_all_linear_layers_precision_v2_aggregate_ledger()
            .unwrap();
        assert_eq!(aggregate.scope, "resident-mtlbuffer-only");
        assert!(!aggregate.includes_lm_head);
        assert_eq!(aggregate.linear_layers.len(), 18);
        assert_eq!(aggregate.full_attention_mlp_layers.len(), 6);
        assert_eq!(aggregate.allocated_buffers, 624);
        assert_eq!(aggregate.shared_buffers, 468);
        assert_eq!(aggregate.private_buffers, 156);
        assert_eq!(aggregate.host_to_device_bytes_per_decode, 6_144);
        assert_eq!(aggregate.device_to_host_bytes_per_decode, 6_144);
        assert_eq!(aggregate.state_host_transfer_bytes_per_decode, 0);
        assert_eq!(aggregate.command_buffers_per_decode, 24);
        assert_eq!(aggregate.compute_encoders_per_decode, 36);
        assert_eq!(aggregate.commits_per_decode, 24);
        assert_eq!(aggregate.waits_per_decode, 24);
        assert_eq!(
            aggregate.total_persistent_mtlbuffer_bytes,
            aggregate
                .linear_layers
                .iter()
                .map(|entry| entry.ledger.total_persistent_bytes)
                .sum::<usize>()
                + aggregate
                    .full_attention_mlp_layers
                    .iter()
                    .map(|entry| entry.ledger.total_persistent_bytes)
                    .sum::<usize>()
        );

        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        let seeded = diagnostic
            .metal_w8_all_linear_layers_precision_v2_stats()
            .unwrap();
        assert!(seeded.linear_layers.iter().all(|stats| {
            stats.execution.prefill_seed_calls == 1 && stats.execution.decode_calls == 0
        }));
        assert!(seeded
            .full_attention_mlp_layers
            .iter()
            .all(|stats| stats.decode_calls == 0));

        let output = diagnostic.forward_hidden(&[3], 2).unwrap();
        assert_eq!(output.shape().dims(), [1, 64]);
        assert!(output
            .as_f32()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        let decoded = diagnostic
            .metal_w8_all_linear_layers_precision_v2_stats()
            .unwrap();
        assert!(decoded.linear_layers.iter().all(|stats| {
            let execution = stats.execution;
            execution.prefill_seed_calls == 1
                && execution.decode_calls == 1
                && execution.successful_decodes == 1
                && execution.failed_decodes == 0
                && execution.command_buffers == 1
                && execution.compute_encoders == 1
                && execution.commits == 1
                && execution.waits == 1
                && execution.committed_state_version == 1
        }));
        assert!(decoded
            .full_attention_mlp_layers
            .iter()
            .all(|stats| stats.decode_calls == 1));
        assert!(!decoded.terminal_error);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_stack3_v1_explicit_slice_runs_layers_zero_one_two_as_one_transaction() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let ordinary =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        assert!(ordinary.metal_w8_linear_layer_stacks_v1_stats().is_none());
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_linear_layer_stack3_v1(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();

        let initial = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert_eq!(initial.mechanism, "metal-w8-linear-layer-stack3-v1");
        assert_eq!(initial.stacks.len(), 1);
        assert_eq!(initial.stacks[0].layer_indices, [0, 1, 2]);
        assert_eq!(initial.stacks[0].prefill_seed_calls, [0, 0, 0]);
        assert_eq!(initial.stacks[0].execution.decode_calls, 0);
        assert_eq!(
            initial.stacks[0].intermediate_host_finite_checks_per_decode,
            0
        );
        assert_eq!(initial.stacks[0].final_output_finite_checks_per_decode, 1);
        assert!(initial.full_attention_mlp_layers.is_empty());
        assert!(!initial.terminal_error);

        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        let seeded = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert_eq!(seeded.stacks[0].prefill_seed_calls, [1, 1, 1]);
        assert_eq!(seeded.stacks[0].execution.decode_calls, 0);

        let output = diagnostic.forward_hidden(&[3], 2).unwrap();
        assert_eq!(output.shape().dims(), [1, 64]);
        assert!(output
            .as_f32()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        let decoded = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        let execution = decoded.stacks[0].execution;
        assert_eq!(execution.decode_calls, 1);
        assert_eq!(execution.successful_decodes, 1);
        assert_eq!(execution.failed_decodes, 0);
        assert_eq!(execution.command_buffers, 1);
        assert_eq!(execution.compute_encoders, 3);
        assert_eq!(execution.commits, 1);
        assert_eq!(execution.waits, 1);
        assert_eq!(execution.state_commits, 3);
        assert_eq!(execution.last_state_commit_mask, 0b111);
        assert_eq!(execution.committed_stack_version, 1);
        assert!(!decoded.terminal_error);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_mlp_stack3_boundary_body_v1_is_explicit_and_owns_the_exact_schedule() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let ordinary =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        assert!(ordinary
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .is_none());
        assert!(ordinary
            .metal_w8_mlp_stack3_boundary_body_v1_aggregate_ledger()
            .is_none());

        let diagnostic = GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_body_v1(
            config,
            tensors,
            Device::Cpu,
            16,
        )
        .unwrap();

        let stats = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        assert_eq!(stats.mechanism, "metal-w8-mlp-stack3-boundary-body-v1");
        assert_eq!(stats.initial_stack.layer_indices, [0, 1, 2]);
        assert_eq!(
            stats
                .boundaries
                .iter()
                .map(|region| (region.boundary_mlp_layer_index, region.stack_layer_indices))
                .collect::<Vec<_>>(),
            vec![
                (3, [4, 5, 6]),
                (7, [8, 9, 10]),
                (11, [12, 13, 14]),
                (15, [16, 17, 18]),
                (19, [20, 21, 22]),
            ]
        );
        assert_eq!(stats.final_mlp.layer_index, 23);
        assert_eq!(stats.initial_stack.execution.decode_calls, 0);
        assert!(stats
            .boundaries
            .iter()
            .all(|region| region.execution.decode_calls == 0));
        assert_eq!(stats.final_mlp.decode_calls, 0);
        assert!(!stats.terminal_error);

        assert!(diagnostic.metal_w8_linear_layer_stacks_v1_stats().is_none());
        assert!(diagnostic.metal_w8_mlp_block_layer_stats().is_empty());
        assert!(diagnostic
            .metal_w8_all_linear_layers_precision_v2_stats()
            .is_none());
        assert!(diagnostic.metal_w8_lm_head_stats().is_none());
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_boundary_tail_head_v1_is_explicit_tied_only_and_owns_layer_23_exclusively() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let ordinary =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        assert!(ordinary
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .is_none());

        let diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
                config.clone(),
                tensors.clone(),
                Device::Cpu,
                16,
            )
            .unwrap();
        let stats = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert_eq!(stats.mechanism, "metal-w8-mlp-stack3-boundary-tail-head-v1");
        assert_eq!(
            stats.gdn_core_profile,
            apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch
        );
        assert_eq!(
            stats.gdn_function_chain,
            apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch.expected_function_chain()
        );
        assert_eq!(
            stats.initial_stack.mechanism,
            "metal-w8-linear-layer-stack3-v1"
        );
        assert_eq!(stats.initial_stack.kernel_dispatches_per_decode, 39);
        assert_eq!(stats.initial_stack.explicit_buffer_barriers_per_decode, 36);
        assert!(stats.boundaries.iter().all(|region| {
            region.mechanism == "metal-w8-mlp-stack3-boundary-v1"
                && region.gdn_core_profile == apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch
                && region.kernel_dispatches_per_decode == 44
                && region.explicit_buffer_barriers_per_decode == 40
        }));
        assert_eq!(stats.initial_stack.layer_indices, [0, 1, 2]);
        assert_eq!(stats.boundaries.len(), 5);
        assert_eq!(stats.tail_layer_index, 23);
        assert_eq!(stats.tail, Default::default());
        assert!(diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .is_none());
        assert!(diagnostic.metal_w8_linear_layer_stacks_v1_stats().is_none());
        assert!(diagnostic.metal_w8_mlp_block_layer_stats().is_empty());
        assert!(diagnostic.metal_w8_lm_head_stats().is_none());

        let mut untied_config = config;
        untied_config.text.tie_word_embeddings = false;
        let mut untied_tensors = tensors;
        untied_tensors.insert(
            "lm_head.weight".into(),
            untied_tensors["model.language_model.embed_tokens.weight"].clone(),
        );
        let error = GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
            untied_config,
            untied_tensors,
            Device::Cpu,
            16,
        )
        .err()
        .expect("tail-head v1 must reject untied output weights");
        assert!(error.to_string().contains("tied word embeddings"));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_boundary_tail_head_v1_uses_cpu_prefill_then_two_exact_reranked_decode_steps() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut cpu =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();

        let cpu_prefill = cpu.prefill_for_generation(LlmInput::text(&[1, 2])).unwrap();
        let diagnostic_prefill = diagnostic
            .prefill_for_generation(LlmInput::text(&[1, 2]))
            .unwrap();
        assert_close(&diagnostic_prefill, &cpu_prefill, 1.0e-7);
        let seeded = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert_eq!(seeded.prefill_body_calls, 1);
        assert_eq!(seeded.prefill_cpu_head_calls, 1);
        assert_eq!(seeded.initial_stack.prefill_seed_calls, [1, 1, 1]);
        assert!(seeded
            .boundaries
            .iter()
            .all(|region| region.prefill_seed_calls == [1, 1, 1]));
        assert_eq!(seeded.tail.decode_calls, 0);

        for (token, position) in [(3, 2), (4, 3)] {
            let cpu_hidden = cpu.forward_hidden(&[token], position).unwrap();
            let cpu_logits = cpu.project_logits(&cpu_hidden).unwrap();
            let expected = argmax_f32_row(&cpu_logits, cpu.config.text.vocab_size).unwrap();
            let comparison = diagnostic
                .teacher_forced_decode_candidates(token, position)
                .unwrap();
            assert_eq!(
                comparison.accelerator_candidate_elapsed_ns(),
                comparison.topk_elapsed_ns
            );
            assert!(comparison.accelerator_candidate_elapsed_ns() > 0);
            assert_eq!(comparison.cpu_token, expected);
            assert!(comparison.w8_candidates.contains(&comparison.cpu_token));
            assert_eq!(comparison.reranked_token, comparison.cpu_token);
        }

        let decoded = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert_eq!(decoded.initial_stack.execution.successful_decodes, 2);
        assert!(decoded
            .boundaries
            .iter()
            .all(|region| region.execution.successful_decodes == 2));
        assert_eq!(decoded.tail.decode_calls, 2);
        assert_eq!(decoded.tail.successful_decodes, 2);
        assert_eq!(decoded.tail.failed_decodes, 0);
        assert_eq!(decoded.decode_calls, 0);
        assert_eq!(decoded.teacher_calls, 2);
        assert!(!decoded.terminal_error);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_boundary_tail_head_v1_generation_uses_cpu_prefill_and_exactly_two_tail_decodes() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut cpu =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();

        let cpu_tokens = cpu
            .generate_streaming(LlmInput::text(&[1, 2]), 3, |_| {}, None)
            .unwrap()
            .0;
        let diagnostic_tokens = diagnostic
            .generate_streaming(LlmInput::text(&[1, 2]), 3, |_| {}, None)
            .unwrap()
            .0;

        assert_eq!(diagnostic_tokens, cpu_tokens);
        let stats = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert_eq!(stats.prefill_body_calls, 1);
        assert_eq!(stats.prefill_cpu_head_calls, 1);
        assert_eq!(stats.decode_calls, 2);
        assert_eq!(stats.teacher_calls, 0);
        assert_eq!(stats.tail.decode_calls, 2);
        assert_eq!(stats.tail.successful_decodes, 2);
        assert_eq!(stats.tail.output_commits, 4);
        assert!(!stats.terminal_error);
        let receipt = diagnostic.generation_path_receipt().unwrap();
        assert_eq!(receipt["prefill_body_calls"], 1);
        assert_eq!(receipt["prefill_head"]["calls"], 1);
        assert_eq!(receipt["prefill_head"]["tail_transactions"], 0);
        assert_eq!(receipt["decode_head"]["calls"], 2);
        assert_eq!(receipt["decode_head"]["teacher_calls"], 0);
        assert_eq!(receipt["decode_head"]["tail_transactions"], 2);
        assert_eq!(receipt["initial_stack"]["last_state_commit_mask"], 0b111);
        assert!(receipt["boundaries"]
            .as_array()
            .unwrap()
            .iter()
            .all(|region| region["last_state_commit_mask"] == 0b111));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_boundary_tail_head_v1_reports_an_independent_exact_ledger_and_receipt() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();

        let aggregate = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_aggregate_ledger()
            .unwrap();
        assert!(aggregate.includes_lm_head);
        assert_eq!(aggregate.initial_stack.layer_indices, [0, 1, 2]);
        assert_eq!(aggregate.boundaries.len(), 5);
        assert_eq!(aggregate.tail_layer_index, 23);
        assert_eq!(aggregate.allocated_buffers, 494);
        assert_eq!(aggregate.shared_buffers, 443);
        assert_eq!(aggregate.private_buffers, 51);
        assert_eq!(aggregate.host_to_device_bytes_per_decode, 7 * 64 * 4);
        assert_eq!(
            aggregate.device_to_host_bytes_per_decode,
            7 * 64 * 4 + 4 * std::mem::size_of::<u32>()
        );
        assert_eq!(aggregate.state_host_transfer_bytes_per_decode, 0);
        assert_eq!(aggregate.command_buffers_per_decode, 7);
        assert_eq!(aggregate.compute_encoders_per_decode, 24);
        assert_eq!(aggregate.kernel_dispatches_per_decode, 267);
        assert_eq!(aggregate.explicit_buffer_barriers_per_decode, 243);
        assert_eq!(
            aggregate.gdn_core_profile,
            apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch
        );
        assert_eq!(aggregate.commits_per_decode, 7);
        assert_eq!(aggregate.waits_per_decode, 7);

        let receipt = diagnostic.generation_path_receipt().unwrap();
        assert_eq!(
            receipt["format"],
            "apxinf-qwen35-mlp-stack3-boundary-tail-head-generation-path-v1"
        );
        assert_eq!(
            receipt["mechanism"],
            "metal-w8-mlp-stack3-boundary-tail-head-v1"
        );
        assert_eq!(receipt["gdn_core_profile"], "legacy-four-dispatch");
        assert_eq!(
            receipt["gdn_function_chain"],
            apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch.expected_function_chain()
        );
        assert_eq!(receipt["cpu_prefill_all_24_layers"], true);
        assert_eq!(receipt["prefill_head"]["mechanism"], "cpu-f32-tied");
        assert_eq!(receipt["decode_head"]["mechanism"], "metal-w8-tail-v1");
        assert_eq!(receipt["decode_head"]["calls"], 0);
        assert_eq!(receipt["aggregate"]["allocated_buffers"], 494);
        assert_eq!(receipt["aggregate"]["compute_encoders_per_decode"], 24);
        assert_eq!(receipt["aggregate"]["kernel_dispatches_per_decode"], 267);
        assert_eq!(
            receipt["aggregate"]["explicit_buffer_barriers_per_decode"],
            243
        );
        assert_eq!(receipt["terminal_error"], false);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_boundary_tail_head_v1_body_fault_submits_zero_tail_work_and_reset_recovers() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.inject_metal_w8_boundary_tail_head_initial_failure_once_for_test();

        let error = diagnostic.decode_token(3, 2).unwrap().unwrap_err();
        assert!(error.to_string().contains("terminal"));
        let failed = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert!(failed.terminal_error);
        assert_eq!(failed.initial_stack.execution.failed_decodes, 1);
        assert!(failed
            .boundaries
            .iter()
            .all(|region| region.execution.decode_calls == 0));
        assert_eq!(failed.tail.decode_calls, 0);
        let failed_receipt = diagnostic.generation_path_receipt().unwrap();
        assert_eq!(failed_receipt["initial_stack"]["last_state_commit_mask"], 0);
        assert!(failed_receipt["boundaries"]
            .as_array()
            .unwrap()
            .iter()
            .all(|region| region["last_state_commit_mask"] == 0));

        let retry = diagnostic.decode_token(3, 2).unwrap().unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(
            diagnostic
                .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
                .unwrap(),
            failed,
            "terminal retry must submit no body or tail work"
        );

        diagnostic.reset();
        let reset = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert!(!reset.terminal_error);
        assert_eq!(reset.initial_stack.execution, Default::default());
        assert!(reset
            .boundaries
            .iter()
            .all(|region| region.execution == Default::default()));
        assert_eq!(reset.tail, Default::default());
        let reset_receipt = diagnostic.generation_path_receipt().unwrap();
        assert_eq!(reset_receipt["initial_stack"]["last_state_commit_mask"], 0);
        assert!(reset_receipt["boundaries"]
            .as_array()
            .unwrap()
            .iter()
            .all(|region| region["last_state_commit_mask"] == 0));
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.decode_token(3, 2).unwrap().unwrap();
        let recovered = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert_eq!(recovered.tail.successful_decodes, 1);
        assert!(!recovered.terminal_error);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_boundary_tail_head_v1_tail_post_execution_fault_is_terminal_and_retry_is_zero_work() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.inject_metal_w8_boundary_tail_head_tail_post_execution_failure_once_for_test();

        let error = diagnostic.decode_token(3, 2).unwrap().unwrap_err();
        assert!(error.to_string().contains("terminal"));
        let failed = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert!(failed.terminal_error);
        assert_eq!(failed.initial_stack.execution.successful_decodes, 1);
        assert!(failed
            .boundaries
            .iter()
            .all(|region| region.execution.successful_decodes == 1));
        assert_eq!(failed.tail.decode_calls, 1);
        assert_eq!(failed.tail.successful_decodes, 0);
        assert_eq!(failed.tail.failed_decodes, 1);
        assert_eq!(failed.tail.device_to_host_bytes, 0);
        assert_eq!(failed.tail.output_commits, 0);
        assert!(failed.tail.terminal_error);
        let failed_receipt = diagnostic.generation_path_receipt().unwrap();
        assert_eq!(
            failed_receipt["initial_stack"]["last_state_commit_mask"],
            0b111
        );
        assert!(failed_receipt["boundaries"]
            .as_array()
            .unwrap()
            .iter()
            .all(|region| region["last_state_commit_mask"] == 0b111));

        let retry = diagnostic.decode_token(3, 2).unwrap().unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(
            diagnostic
                .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
                .unwrap(),
            failed
        );

        diagnostic.reset();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.decode_token(3, 2).unwrap().unwrap();
        let recovered = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert_eq!(recovered.tail.successful_decodes, 1);
        assert!(!recovered.terminal_error);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_boundary_tail_head_v1_malformed_tail_outputs_are_terminal_retry_zero_and_resettable() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        for fault in [
            Qwen35BoundaryTailHeadFaultV1ForTest::TailNonFiniteOutput,
            Qwen35BoundaryTailHeadFaultV1ForTest::TailDuplicateCandidate,
            Qwen35BoundaryTailHeadFaultV1ForTest::TailOutOfRangeCandidate,
        ] {
            let mut diagnostic =
                GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
                    config.clone(),
                    tensors.clone(),
                    Device::Cpu,
                    16,
                )
                .unwrap();
            diagnostic.forward_hidden(&[1, 2], 0).unwrap();
            diagnostic.inject_metal_w8_boundary_tail_head_malformed_once_for_test(fault);

            let error = diagnostic.decode_token(3, 2).unwrap().unwrap_err();
            assert!(error.to_string().contains("terminal"));
            let failed = diagnostic
                .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
                .unwrap();
            assert!(failed.terminal_error);
            assert_eq!(failed.tail.decode_calls, 1);
            assert_eq!(failed.tail.failed_decodes, 1);
            assert_eq!(failed.tail.device_to_host_bytes, 0);
            assert_eq!(failed.tail.output_commits, 0);
            assert!(failed.tail.terminal_error);

            let retry = diagnostic.decode_token(3, 2).unwrap().unwrap_err();
            assert!(retry.to_string().contains("reset required"));
            assert_eq!(
                diagnostic
                    .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
                    .unwrap(),
                failed
            );

            diagnostic.reset();
            diagnostic.forward_hidden(&[1, 2], 0).unwrap();
            diagnostic.decode_token(3, 2).unwrap().unwrap();
            assert!(
                !diagnostic
                    .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
                    .unwrap()
                    .terminal_error
            );
        }
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_boundary_tail_head_v1_rerank_fault_latches_after_tail_and_reset_recovers() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.inject_metal_w8_boundary_tail_head_rerank_failure_once_for_test();

        let error = diagnostic.decode_token(3, 2).unwrap().unwrap_err();
        assert!(error.to_string().contains("rerank"));
        assert!(error.to_string().contains("terminal"));
        let failed = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert!(failed.terminal_error);
        assert_eq!(failed.tail.decode_calls, 1);
        assert_eq!(failed.tail.successful_decodes, 1);
        assert_eq!(failed.tail.output_commits, 2);
        assert!(!failed.tail.terminal_error);
        assert_eq!(failed.decode_calls, 0);
        assert_eq!(failed.rerank_elapsed_ns, 0);

        let retry = diagnostic.decode_token(3, 2).unwrap().unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(
            diagnostic
                .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
                .unwrap(),
            failed
        );

        diagnostic.reset();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.decode_token(3, 2).unwrap().unwrap();
        let recovered = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert_eq!(recovered.decode_calls, 1);
        assert_eq!(recovered.tail.successful_decodes, 1);
        assert!(!recovered.terminal_error);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_boundary_tail_head_v1_partial_prefill_is_terminal_retry_zero_and_resettable() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_v1(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();
        diagnostic.inject_failure_after_layer_once_for_test(1);

        let error = diagnostic.forward_hidden(&[1, 2], 0).unwrap_err();
        assert!(error.to_string().contains("injected after layer 1"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 0);
        let failed = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert!(failed.terminal_error);
        assert_eq!(failed.initial_stack.prefill_seed_calls, [1, 1, 0]);
        assert_eq!(failed.prefill_body_calls, 0);
        assert_eq!(failed.tail.decode_calls, 0);

        let retry = diagnostic.forward_hidden(&[1, 2], 0).unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(
            diagnostic
                .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
                .unwrap(),
            failed
        );

        diagnostic.reset();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        let recovered = diagnostic
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .unwrap();
        assert!(!recovered.terminal_error);
        assert_eq!(recovered.initial_stack.prefill_seed_calls, [1, 1, 1]);
        assert_eq!(recovered.prefill_body_calls, 1);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_boundary_tail_head_v1_keeps_legacy_lanes_and_receipt_formats_isolated() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let legacy_boundary =
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_body_v1(
                config.clone(),
                tensors.clone(),
                Device::Cpu,
                16,
            )
            .unwrap();
        assert!(legacy_boundary
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .is_none());
        assert_eq!(
            legacy_boundary.generation_path_receipt().unwrap()["format"],
            "apxinf-qwen35-mlp-stack3-boundary-body-generation-path-v1"
        );

        let legacy_stack_head =
            GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_and_lm_head_v2(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();
        assert!(legacy_stack_head
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .is_none());
        assert_eq!(
            legacy_stack_head.generation_path_receipt().unwrap()["format"],
            "apxinf-qwen35-stack3-lm-head-generation-path-v2"
        );
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_mlp_stack3_boundary_body_v1_prefills_once_and_runs_two_exact_decode_schedules() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut cpu =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_body_v1(
            config,
            tensors,
            Device::Cpu,
            16,
        )
        .unwrap();

        let cpu_prefill = cpu.forward_hidden(&[1, 2], 0).unwrap();
        let diagnostic_prefill = diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        assert_close(&diagnostic_prefill, &cpu_prefill, 1.0e-7);
        let seeded = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        assert_eq!(seeded.initial_stack.prefill_seed_calls, [1, 1, 1]);
        assert!(seeded
            .boundaries
            .iter()
            .all(|region| region.prefill_seed_calls == [1, 1, 1]));
        assert_eq!(seeded.initial_stack.execution.decode_calls, 0);
        assert!(seeded
            .boundaries
            .iter()
            .all(|region| region.execution.decode_calls == 0));
        assert_eq!(seeded.final_mlp.decode_calls, 0);
        let owned_stack_indices = [
            [0, 1, 2],
            [4, 5, 6],
            [8, 9, 10],
            [12, 13, 14],
            [16, 17, 18],
            [20, 21, 22],
        ];
        let cpu_seed_states = owned_stack_indices.map(|indices| {
            indices.map(|layer_index| {
                gdn_decode_state_from_cpu(
                    &diagnostic.config.text,
                    diagnostic.state.linear_state(layer_index).unwrap(),
                )
                .unwrap()
            })
        });
        let boundary_body = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1
            .as_ref()
            .unwrap();
        assert_eq!(
            boundary_body.initial_stack.block.state_snapshots().unwrap(),
            cpu_seed_states[0]
        );
        for (region, expected_states) in boundary_body
            .boundaries
            .iter()
            .zip(cpu_seed_states[1..].iter())
        {
            assert_eq!(&region.block.state_snapshots().unwrap(), expected_states);
        }
        let before_invalid_decode = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        let invalid_decode = diagnostic.forward_hidden(&[3, 4], 2).unwrap_err();
        assert!(invalid_decode.to_string().contains("single-token decode"));
        assert_eq!(diagnostic.state.position(), 2);
        assert_eq!(
            diagnostic
                .metal_w8_mlp_stack3_boundary_body_v1_stats()
                .unwrap(),
            before_invalid_decode,
            "multi-token decode must be rejected before any transaction"
        );

        for (token, position) in [(3, 2), (4, 3)] {
            let cpu_hidden = cpu.forward_hidden(&[token], position).unwrap();
            let diagnostic_hidden = diagnostic.forward_hidden(&[token], position).unwrap();
            assert_close(&diagnostic_hidden, &cpu_hidden, 3.0e-2);
        }

        let decoded = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        assert_eq!(decoded.initial_stack.execution.decode_calls, 2);
        assert_eq!(decoded.initial_stack.execution.successful_decodes, 2);
        assert_eq!(decoded.initial_stack.execution.committed_stack_version, 2);
        assert_eq!(
            decoded.initial_stack.execution.host_to_device_bytes,
            2 * 64 * 4
        );
        assert_eq!(
            decoded.initial_stack.execution.device_to_host_bytes,
            2 * 64 * 4
        );
        assert!(decoded.boundaries.iter().all(|region| {
            region.execution.decode_calls == 2
                && region.execution.successful_decodes == 2
                && region.execution.failed_decodes == 0
                && region.execution.command_buffers == 2
                && region.execution.compute_encoders == 8
                && region.execution.commits == 2
                && region.execution.waits == 2
                && region.execution.state_commits == 6
                && region.execution.last_state_commit_mask == 0b111
                && region.execution.committed_stack_version == 2
                && region.execution.host_to_device_bytes == 2 * 64 * 4
                && region.execution.device_to_host_bytes == 2 * 64 * 4
        }));
        assert_eq!(decoded.final_mlp.decode_calls, 2);
        assert!(!decoded.terminal_error);
        for (indices, expected_states) in owned_stack_indices.iter().zip(cpu_seed_states.iter()) {
            let frozen = indices.map(|layer_index| {
                gdn_decode_state_from_cpu(
                    &diagnostic.config.text,
                    diagnostic.state.linear_state(layer_index).unwrap(),
                )
                .unwrap()
            });
            assert_eq!(
                &frozen, expected_states,
                "CPU recurrent state for Metal-owned layers must stay frozen after prefill"
            );
        }
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_mlp_stack3_boundary_body_v1_reports_one_exact_aggregate_and_receipt() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let diagnostic = GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_body_v1(
            config,
            tensors,
            Device::Cpu,
            16,
        )
        .unwrap();

        let aggregate = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_aggregate_ledger()
            .unwrap();
        assert_eq!(aggregate.scope, "resident-mtlbuffer-only");
        assert!(!aggregate.includes_lm_head);
        assert_eq!(aggregate.initial_stack.layer_indices, [0, 1, 2]);
        assert_eq!(aggregate.boundaries.len(), 5);
        assert_eq!(aggregate.final_mlp.layer_index, 23);
        assert_eq!(aggregate.allocated_buffers, 489);
        assert_eq!(aggregate.shared_buffers, 439);
        assert_eq!(aggregate.private_buffers, 50);
        assert_eq!(aggregate.host_to_device_bytes_per_decode, 7 * 64 * 4);
        assert_eq!(aggregate.device_to_host_bytes_per_decode, 7 * 64 * 4);
        assert_eq!(aggregate.state_host_transfer_bytes_per_decode, 0);
        assert_eq!(aggregate.command_buffers_per_decode, 7);
        assert_eq!(aggregate.compute_encoders_per_decode, 26);
        assert_eq!(aggregate.kernel_dispatches_per_decode, 262);
        assert_eq!(aggregate.commits_per_decode, 7);
        assert_eq!(aggregate.waits_per_decode, 7);
        assert_eq!(aggregate.intermediate_host_finite_checks_per_decode, 0);
        assert_eq!(aggregate.final_output_finite_checks_per_decode, 6);
        assert_eq!(
            aggregate.total_persistent_mtlbuffer_bytes,
            aggregate.initial_stack.ledger.total_persistent_bytes
                + aggregate
                    .boundaries
                    .iter()
                    .map(|entry| entry.ledger.total_persistent_bytes)
                    .sum::<usize>()
                + aggregate.final_mlp.ledger.total_persistent_bytes
        );

        let receipt = diagnostic.generation_path_receipt().unwrap();
        assert_eq!(
            receipt["format"],
            "apxinf-qwen35-mlp-stack3-boundary-body-generation-path-v1"
        );
        assert_eq!(receipt["mechanism"], "metal-w8-mlp-stack3-boundary-body-v1");
        assert_eq!(
            receipt["initial_stack"]["layer_indices"],
            serde_json::json!([0, 1, 2])
        );
        assert_eq!(receipt["boundaries"].as_array().unwrap().len(), 5);
        assert_eq!(receipt["final_mlp"]["layer_index"], 23);
        assert_eq!(receipt["terminal_error"], false);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_mlp_stack3_boundary_body_v1_latches_the_whole_lane_after_a_later_fault() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_body_v1(
            config,
            tensors,
            Device::Cpu,
            16,
        )
        .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.inject_metal_w8_mlp_stack3_boundary_failure_once_for_test(0);

        let error = diagnostic.forward_hidden(&[3], 2).unwrap_err();
        assert!(error.to_string().contains("lane is terminal"));
        let failed = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        assert!(failed.terminal_error);
        assert_eq!(failed.initial_stack.execution.successful_decodes, 1);
        assert_eq!(failed.initial_stack.execution.committed_stack_version, 1);
        assert_eq!(failed.boundaries[0].execution.decode_calls, 1);
        assert_eq!(failed.boundaries[0].execution.successful_decodes, 0);
        assert_eq!(failed.boundaries[0].execution.failed_decodes, 1);
        assert_eq!(failed.boundaries[0].execution.state_commits, 0);
        assert_eq!(failed.boundaries[0].execution.device_to_host_bytes, 0);
        assert_eq!(failed.boundaries[0].execution.committed_stack_version, 0);
        assert!(failed.boundaries[0].terminal_error);
        assert!(failed.boundaries[1..]
            .iter()
            .all(|region| region.execution.decode_calls == 0));
        assert_eq!(failed.final_mlp.decode_calls, 0);

        let retry = diagnostic.forward_hidden(&[3], 2).unwrap_err();
        assert!(retry.to_string().contains("terminal"));
        assert_eq!(
            diagnostic
                .metal_w8_mlp_stack3_boundary_body_v1_stats()
                .unwrap(),
            failed,
            "terminal retry must submit no additional work"
        );

        diagnostic.reset();
        let reset = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        assert!(!reset.terminal_error);
        assert_eq!(reset.initial_stack.prefill_seed_calls, [0, 0, 0]);
        assert_eq!(reset.initial_stack.execution.decode_calls, 0);
        assert!(reset.boundaries.iter().all(|region| {
            region.prefill_seed_calls == [0, 0, 0] && region.execution.decode_calls == 0
        }));
        assert_eq!(reset.final_mlp.decode_calls, 0);

        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[3], 2).unwrap();
        let recovered = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        assert!(!recovered.terminal_error);
        assert_eq!(recovered.initial_stack.execution.successful_decodes, 1);
        assert!(recovered
            .boundaries
            .iter()
            .all(|region| region.execution.successful_decodes == 1));
        assert_eq!(recovered.final_mlp.decode_calls, 1);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_mlp_stack3_boundary_body_v1_every_region_fault_is_terminal_and_retry_is_zero_work() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        for fault_region in 0..QWEN35_MLP_STACK3_BOUNDARY_REGIONS_V1.len() {
            let mut diagnostic =
                GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_body_v1(
                    config.clone(),
                    tensors.clone(),
                    Device::Cpu,
                    16,
                )
                .unwrap();
            diagnostic.forward_hidden(&[1, 2], 0).unwrap();
            diagnostic.inject_metal_w8_mlp_stack3_boundary_failure_once_for_test(fault_region);

            let error = diagnostic.forward_hidden(&[3], 2).unwrap_err();
            assert!(error.to_string().contains("entire lane is terminal"));
            let failed = diagnostic
                .metal_w8_mlp_stack3_boundary_body_v1_stats()
                .unwrap();
            assert!(failed.terminal_error);
            assert_eq!(failed.initial_stack.execution.successful_decodes, 1);
            for (region_index, region) in failed.boundaries.iter().enumerate() {
                if region_index < fault_region {
                    assert_eq!(region.execution.successful_decodes, 1);
                    assert_eq!(region.execution.committed_stack_version, 1);
                } else if region_index == fault_region {
                    assert_eq!(region.execution.decode_calls, 1);
                    assert_eq!(region.execution.failed_decodes, 1);
                    assert_eq!(region.execution.state_commits, 0);
                    assert_eq!(region.execution.device_to_host_bytes, 0);
                    assert_eq!(region.execution.committed_stack_version, 0);
                } else {
                    assert_eq!(region.execution.decode_calls, 0);
                }
            }
            assert_eq!(failed.final_mlp.decode_calls, 0);

            let retry = diagnostic.forward_hidden(&[3], 2).unwrap_err();
            assert!(retry.to_string().contains("reset required"));
            assert_eq!(
                diagnostic
                    .metal_w8_mlp_stack3_boundary_body_v1_stats()
                    .unwrap(),
                failed,
                "region {fault_region} terminal retry must submit no work"
            );

            diagnostic.reset();
            let reset = diagnostic
                .metal_w8_mlp_stack3_boundary_body_v1_stats()
                .unwrap();
            assert!(!reset.terminal_error);
            assert_eq!(reset.initial_stack.execution, Default::default());
            assert!(reset
                .boundaries
                .iter()
                .all(|region| region.execution == Default::default()));
            assert_eq!(reset.final_mlp.decode_calls, 0);
        }
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_mlp_stack3_boundary_body_v1_initial_stack_fault_latches_retry_and_resets() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_body_v1(
            config,
            tensors,
            Device::Cpu,
            16,
        )
        .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.inject_metal_w8_mlp_stack3_boundary_initial_failure_once_for_test();

        let error = diagnostic.forward_hidden(&[3], 2).unwrap_err();
        assert!(error.to_string().contains("entire lane is terminal"));
        let failed = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        assert!(failed.terminal_error);
        assert_eq!(failed.initial_stack.execution.decode_calls, 1);
        assert_eq!(failed.initial_stack.execution.successful_decodes, 0);
        assert_eq!(failed.initial_stack.execution.failed_decodes, 1);
        assert_eq!(failed.initial_stack.execution.state_commits, 0);
        assert_eq!(failed.initial_stack.execution.device_to_host_bytes, 0);
        assert_eq!(failed.initial_stack.execution.committed_stack_version, 0);
        assert!(failed.initial_stack.terminal_error);
        assert!(failed
            .boundaries
            .iter()
            .all(|region| region.execution.decode_calls == 0));
        assert_eq!(failed.final_mlp.decode_calls, 0);

        let retry = diagnostic.forward_hidden(&[3], 2).unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(
            diagnostic
                .metal_w8_mlp_stack3_boundary_body_v1_stats()
                .unwrap(),
            failed,
            "initial Stack3 terminal retry must submit no work"
        );

        diagnostic.reset();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[3], 2).unwrap();
        let recovered = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        assert!(!recovered.terminal_error);
        assert_eq!(recovered.initial_stack.execution.successful_decodes, 1);
        assert!(recovered
            .boundaries
            .iter()
            .all(|region| region.execution.successful_decodes == 1));
        assert_eq!(recovered.final_mlp.decode_calls, 1);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_mlp_stack3_boundary_body_v1_final_mlp_fault_latches_retry_and_resets() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_body_v1(
            config,
            tensors,
            Device::Cpu,
            16,
        )
        .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.inject_metal_w8_mlp_stack3_boundary_final_mlp_failure_once_for_test();

        let error = diagnostic.forward_hidden(&[3], 2).unwrap_err();
        assert!(error.to_string().contains("final MLP"));
        assert!(error.to_string().contains("entire lane is terminal"));
        let failed = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        assert!(failed.terminal_error);
        assert_eq!(failed.initial_stack.execution.successful_decodes, 1);
        assert!(failed
            .boundaries
            .iter()
            .all(|region| region.execution.successful_decodes == 1));
        assert_eq!(failed.final_mlp.decode_calls, 1);

        let retry = diagnostic.forward_hidden(&[3], 2).unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(
            diagnostic
                .metal_w8_mlp_stack3_boundary_body_v1_stats()
                .unwrap(),
            failed,
            "final MLP terminal retry must submit no work"
        );

        diagnostic.reset();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[3], 2).unwrap();
        let recovered = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        assert!(!recovered.terminal_error);
        assert_eq!(recovered.initial_stack.execution.successful_decodes, 1);
        assert!(recovered
            .boundaries
            .iter()
            .all(|region| region.execution.successful_decodes == 1));
        assert_eq!(recovered.final_mlp.decode_calls, 1);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_mlp_stack3_boundary_body_v1_partial_prefill_is_terminal_until_reset() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_body_v1(
            config,
            tensors,
            Device::Cpu,
            16,
        )
        .unwrap();
        diagnostic.inject_failure_after_layer_once_for_test(1);

        let error = diagnostic.forward_hidden(&[1, 2], 0).unwrap_err();

        assert!(error.to_string().contains("injected after layer 1"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 0);
        let failed = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        assert!(failed.terminal_error);
        assert_eq!(failed.initial_stack.prefill_seed_calls, [1, 1, 0]);
        assert_eq!(failed.initial_stack.execution.decode_calls, 0);
        assert!(failed.boundaries.iter().all(|region| {
            region.prefill_seed_calls == [0, 0, 0] && region.execution.decode_calls == 0
        }));
        assert_eq!(failed.final_mlp.decode_calls, 0);

        let retry = diagnostic.forward_hidden(&[1, 2], 0).unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(
            diagnostic
                .metal_w8_mlp_stack3_boundary_body_v1_stats()
                .unwrap(),
            failed,
            "dirty partial prefill retry must submit no work"
        );

        diagnostic.reset();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        let recovered = diagnostic
            .metal_w8_mlp_stack3_boundary_body_v1_stats()
            .unwrap();
        assert!(!recovered.terminal_error);
        assert_eq!(recovered.initial_stack.prefill_seed_calls, [1, 1, 1]);
        assert!(recovered
            .boundaries
            .iter()
            .all(|region| region.prefill_seed_calls == [1, 1, 1]));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_mlp_stack3_boundary_body_v1_rejects_a_changed_full_attention_index() {
        let (mut config, tensors) = metal_all_linear_layers_fixture();
        config.text.layer_types[3] = Qwen35LayerType::LinearAttention;

        let error = match GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_body_v1(
            config,
            tensors,
            Device::Cpu,
            16,
        ) {
            Ok(_) => panic!("changed full-attention index must fail closed"),
            Err(error) => error,
        };

        assert!(error
            .to_string()
            .contains("requires full-attention layer at index 3"));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_stack3_v1_all_six_runs_have_the_exact_body_ledger_and_no_duplicate_mlp() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_v1(
            config,
            tensors,
            Device::Cpu,
            16,
        )
        .unwrap();
        let expected_stacks = [
            [0, 1, 2],
            [4, 5, 6],
            [8, 9, 10],
            [12, 13, 14],
            [16, 17, 18],
            [20, 21, 22],
        ];
        let expected_full = [3, 7, 11, 15, 19, 23];

        let initial = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert_eq!(initial.stacks.len(), 6);
        assert_eq!(
            initial
                .stacks
                .iter()
                .map(|stack| stack.layer_indices)
                .collect::<Vec<_>>(),
            expected_stacks
        );
        assert_eq!(
            initial
                .full_attention_mlp_layers
                .iter()
                .map(|stats| stats.layer_index)
                .collect::<Vec<_>>(),
            expected_full
        );
        let ledgers = diagnostic
            .metal_w8_linear_layer_stack3_v1_buffer_ledgers()
            .unwrap();
        assert_eq!(ledgers.len(), 6);
        assert!(ledgers.iter().all(|entry| {
            entry.ledger.allocated_buffers == 76
                && entry.ledger.shared_buffers == 68
                && entry.ledger.private_buffers == 8
                && entry.ledger.command_buffers_per_decode == 1
                && entry.ledger.compute_encoders_per_decode == 3
                && entry.ledger.commits_per_decode == 1
                && entry.ledger.waits_per_decode == 1
                && entry.ledger.intermediate_host_finite_checks_per_decode == 0
                && entry.ledger.final_output_finite_checks_per_decode == 1
        }));
        let aggregate = diagnostic
            .metal_w8_linear_layer_stacks_v1_aggregate_ledger()
            .unwrap();
        assert_eq!(aggregate.scope, "resident-mtlbuffer-only");
        assert!(!aggregate.includes_lm_head);
        assert_eq!(aggregate.stacks.len(), 6);
        assert_eq!(aggregate.full_attention_mlp_layers.len(), 6);
        assert_eq!(aggregate.allocated_buffers, 504);
        assert_eq!(aggregate.shared_buffers, 444);
        assert_eq!(aggregate.private_buffers, 60);
        assert_eq!(aggregate.host_to_device_bytes_per_decode, 3_072);
        assert_eq!(aggregate.device_to_host_bytes_per_decode, 3_072);
        assert_eq!(aggregate.state_host_transfer_bytes_per_decode, 0);
        assert_eq!(aggregate.command_buffers_per_decode, 12);
        assert_eq!(aggregate.compute_encoders_per_decode, 36);
        assert_eq!(aggregate.commits_per_decode, 12);
        assert_eq!(aggregate.waits_per_decode, 12);
        assert_eq!(aggregate.intermediate_host_finite_checks_per_decode, 0);
        assert_eq!(aggregate.final_output_finite_checks_per_decode, 6);
        assert_eq!(
            aggregate.total_persistent_mtlbuffer_bytes,
            aggregate
                .stacks
                .iter()
                .map(|entry| entry.ledger.total_persistent_bytes)
                .sum::<usize>()
                + aggregate
                    .full_attention_mlp_layers
                    .iter()
                    .map(|entry| entry.ledger.total_persistent_bytes)
                    .sum::<usize>()
        );
        let receipt = diagnostic.generation_path_receipt().unwrap();
        assert_eq!(
            receipt["format"],
            "apxinf-qwen35-linear-layer-stacks-generation-path-v1"
        );
        assert_eq!(receipt["stacks"].as_array().unwrap().len(), 6);
        assert_eq!(
            receipt["full_attention_mlp_layers"]
                .as_array()
                .unwrap()
                .len(),
            6
        );

        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[3], 2).unwrap();
        let decoded = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert!(decoded.stacks.iter().all(|stack| {
            stack.prefill_seed_calls == [1, 1, 1]
                && stack.execution.decode_calls == 1
                && stack.execution.successful_decodes == 1
                && stack.execution.state_commits == 3
                && stack.execution.last_state_commit_mask == 0b111
        }));
        assert!(decoded
            .full_attention_mlp_layers
            .iter()
            .all(|stats| stats.decode_calls == 1));
        assert!(!decoded.terminal_error);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_stack3_head_v2_constructor_is_an_explicit_composite_kill_switch() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let legacy = GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_v1(
            config.clone(),
            tensors.clone(),
            Device::Cpu,
            16,
        )
        .unwrap();
        assert!(legacy.metal_w8_lm_head_stats().is_none());

        let diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_and_lm_head_v2(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();

        let body = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert_eq!(body.stacks.len(), 6);
        assert_eq!(body.full_attention_mlp_layers.len(), 6);
        assert!(diagnostic.metal_w8_lm_head_stats().is_some());
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_stack3_head_v2_model_exposes_body_plus_head_ledger_components() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_and_lm_head_v2(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();

        let aggregate = diagnostic
            .metal_w8_stack3_lm_head_v2_aggregate_ledger()
            .unwrap();

        assert_eq!(aggregate.body.stacks.len(), 6);
        assert_eq!(aggregate.body.full_attention_mlp_layers.len(), 6);
        assert_eq!(aggregate.lm_head.allocated_buffers, 5);
        assert_eq!(
            aggregate.total_persistent_mtlbuffer_bytes,
            aggregate.body.total_persistent_mtlbuffer_bytes
                + aggregate.lm_head.total_persistent_bytes
        );
        assert_eq!(
            aggregate.command_buffers_per_call,
            aggregate.body.command_buffers_per_decode + 1
        );
        assert_eq!(
            aggregate.compute_encoders_per_call,
            aggregate.body.compute_encoders_per_decode + 2
        );
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_stack3_head_v2_uses_a_new_generation_receipt_with_head_stats() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_and_lm_head_v2(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();

        let receipt = diagnostic.generation_path_receipt().unwrap();

        assert_eq!(
            receipt["format"],
            "apxinf-qwen35-stack3-lm-head-generation-path-v2"
        );
        assert_eq!(receipt["mechanism"], "metal-w8-stack3-lm-head-v2");
        assert_eq!(receipt["metal_w8_complete_linear_layer_stacks"], true);
        assert_eq!(receipt["metal_w8_full_attention_mlp_blocks"], true);
        assert_eq!(receipt["metal_w8_tied_lm_head_topk4_f32_rerank"], true);
        assert_eq!(receipt["stacks"].as_array().unwrap().len(), 6);
        assert_eq!(
            receipt["full_attention_mlp_layers"]
                .as_array()
                .unwrap()
                .len(),
            6
        );
        assert_eq!(receipt["lm_head"]["mechanism"], "metal-w8-top4-f32-rerank");
        assert_eq!(receipt["lm_head"]["prefill_calls"], 0);
        assert_eq!(receipt["lm_head"]["decode_calls"], 0);
        assert_eq!(receipt["lm_head"]["teacher_calls"], 0);
        assert_eq!(receipt["terminal_error"], false);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_stack3_head_v2_head_failure_latches_retry_and_reset_clears_every_lane() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_and_lm_head_v2(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();
        diagnostic
            .prefill_for_generation(LlmInput::text(&[1, 2]))
            .unwrap();
        assert_eq!(diagnostic.state.position(), 2);
        assert_eq!(
            diagnostic.metal_w8_lm_head_stats().unwrap(),
            Qwen35MetalW8LmHeadStats::default()
        );
        diagnostic.inject_metal_w8_stack3_lm_head_failure_once_for_test();

        let error = diagnostic
            .teacher_forced_decode_candidates(3, 2)
            .err()
            .expect("the injected post-body head failure must be returned");

        assert!(error.to_string().contains("Stack3 + lm_head v2"));
        assert!(error.to_string().contains("terminal"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 3);
        let failed_body = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert!(failed_body.terminal_error);
        assert!(failed_body
            .stacks
            .iter()
            .all(|stack| stack.execution.decode_calls == 1));
        assert!(failed_body
            .full_attention_mlp_layers
            .iter()
            .all(|layer| layer.decode_calls == 1));
        let failed_head = diagnostic.metal_w8_lm_head_stats().unwrap();
        assert_eq!(failed_head, Qwen35MetalW8LmHeadStats::default());
        assert_eq!(
            diagnostic.generation_path_receipt().unwrap()["terminal_error"],
            true
        );

        let retry_error = diagnostic
            .teacher_forced_decode_candidates(4, 3)
            .err()
            .expect("a terminal composite retry must fail before body work");
        assert!(retry_error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 3);
        assert_eq!(
            diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap(),
            failed_body
        );
        assert_eq!(diagnostic.metal_w8_lm_head_stats().unwrap(), failed_head);

        diagnostic.reset();

        assert_eq!(diagnostic.state.position(), 0);
        let reset_body = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert!(!reset_body.terminal_error);
        assert!(reset_body.stacks.iter().all(|stack| {
            stack.prefill_seed_calls == [0, 0, 0]
                && stack.execution.decode_calls == 0
                && !stack.terminal_error
        }));
        assert!(reset_body
            .full_attention_mlp_layers
            .iter()
            .all(|layer| layer.decode_calls == 0));
        assert_eq!(
            diagnostic.metal_w8_lm_head_stats().unwrap(),
            Qwen35MetalW8LmHeadStats::default()
        );
        assert_eq!(
            diagnostic.generation_path_receipt().unwrap()["terminal_error"],
            false
        );
        diagnostic
            .prefill_token_for_generation(LlmInput::text(&[1, 2]))
            .expect("the reset composite must still own prefill")
            .unwrap();
        assert_eq!(
            diagnostic.metal_w8_lm_head_stats().unwrap().prefill_calls,
            1
        );
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_stack3_head_v2_body_failure_never_reaches_the_head() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_and_lm_head_v2(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();
        diagnostic
            .prefill_for_generation(LlmInput::text(&[1, 2]))
            .unwrap();
        diagnostic.inject_metal_w8_stack3_failure_once_for_test(0);

        let error = diagnostic
            .teacher_forced_decode_candidates(3, 2)
            .err()
            .expect("the injected body failure must be returned");

        assert!(error.to_string().contains("Stack3 + lm_head v2"));
        assert!(error.to_string().contains("terminal"));
        assert_eq!(
            diagnostic.metal_w8_lm_head_stats().unwrap(),
            Qwen35MetalW8LmHeadStats::default()
        );
        assert_eq!(
            diagnostic.generation_path_receipt().unwrap()["terminal_error"],
            true
        );
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_stack3_head_v2_teacher_step_binds_candidate_f32_top4_and_rerank() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_and_lm_head_v2(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();
        diagnostic
            .prefill_for_generation(LlmInput::text(&[1, 2]))
            .unwrap();

        let comparison = diagnostic.teacher_forced_decode_candidates(3, 2).unwrap();

        assert_eq!(comparison.reranked_token, comparison.cpu_token);
        assert!(comparison.w8_candidates.contains(&comparison.cpu_token));
        let head = diagnostic.metal_w8_lm_head_stats().unwrap();
        assert_eq!(head.prefill_calls, 0);
        assert_eq!(head.decode_calls, 0);
        assert_eq!(head.teacher_calls, 1);
        let body = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert!(body
            .stacks
            .iter()
            .all(|stack| stack.execution.decode_calls == 1));
        assert!(body
            .full_attention_mlp_layers
            .iter()
            .all(|layer| layer.decode_calls == 1));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_stack3_head_v2_free_run_uses_the_shared_head_fast_path() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut cpu =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_and_lm_head_v2(
                config,
                tensors,
                Device::Cpu,
                16,
            )
            .unwrap();

        let cpu_tokens = cpu
            .generate_streaming(LlmInput::text(&[1, 2]), 3, |_| {}, None)
            .unwrap()
            .0;
        let diagnostic_tokens = diagnostic
            .generate_streaming(LlmInput::text(&[1, 2]), 3, |_| {}, None)
            .unwrap()
            .0;

        assert_eq!(diagnostic_tokens, cpu_tokens);
        assert_eq!(diagnostic_tokens.len(), 3);
        let head = diagnostic.metal_w8_lm_head_stats().unwrap();
        assert_eq!(head.prefill_calls, 1);
        assert_eq!(head.decode_calls, 2);
        assert_eq!(head.teacher_calls, 0);
        let body = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert!(body.stacks.iter().all(|stack| {
            stack.prefill_seed_calls == [1, 1, 1] && stack.execution.decode_calls == 2
        }));
        assert!(body
            .full_attention_mlp_layers
            .iter()
            .all(|layer| layer.decode_calls == 2));
    }

    #[cfg(feature = "metal-experiments")]
    #[test]
    fn metal_stack3_v1_qwen35_08b_static_ledger_closes_to_the_frozen_target() {
        let stack_ledger = apxinf_metal::LinearLayerStack3BufferLedger {
            gdn_core_profile: apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch,
            gdn_function_chain: apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch
                .expected_function_chain(),
            allocated_buffers: 76,
            shared_buffers: 68,
            private_buffers: 8,
            packed_weight_bytes: 64_585_728,
            packed_scale_bytes: 4_429_824,
            f32_parameter_bytes: 321_408,
            active_state_bytes: 3_440_640,
            scratch_state_bytes: 3_440_640,
            activation_bytes: 133_248,
            total_persistent_bytes: 76_351_488,
            host_input_bytes_per_decode: 4_096,
            host_output_bytes_per_decode: 4_096,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 1,
            compute_encoders_per_decode: 3,
            kernel_dispatches_per_decode: 39,
            explicit_buffer_barriers_per_decode: 36,
            gdn_core_seams_per_decode: 3,
            gdn_core_kernel_dispatches_per_decode: 12,
            gdn_core_explicit_buffer_barriers_per_decode: 12,
            gdn_core_recurrent_or_fused_threads_per_threadgroup: 256,
            gdn_core_threadgroups_per_decode: 126,
            gdn_core_launched_threads_per_decode: 30_864,
            gdn_core_source_declared_threadgroup_memory_bytes: 0,
            gdn_core_expected_pipeline_static_threadgroup_memory_bytes: 0,
            gdn_core_internal_threadgroup_barrier_sites_per_threadgroup: 0,
            commits_per_decode: 1,
            waits_per_decode: 1,
            intermediate_host_finite_checks_per_decode: 0,
            final_output_finite_checks_per_decode: 1,
        };
        let mlp_ledger = apxinf_metal::MlpBlockBufferLedger {
            scope: "resident-mtlbuffer-only",
            allocated_buffers: 8,
            shared_buffers: 6,
            private_buffers: 2,
            packed_weight_bytes: 11_010_048,
            packed_scale_bytes: 688_128,
            activation_bytes: 51_200,
            total_persistent_bytes: 11_749_376,
            host_input_bytes_per_decode: 4_096,
            host_output_bytes_per_decode: 4_096,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 1,
            compute_encoders_per_decode: 3,
            commits_per_decode: 1,
            waits_per_decode: 1,
        };
        let stack_indices = [
            [0, 1, 2],
            [4, 5, 6],
            [8, 9, 10],
            [12, 13, 14],
            [16, 17, 18],
            [20, 21, 22],
        ];
        let full_indices = [3, 7, 11, 15, 19, 23];
        let aggregate = Qwen35MetalW8LinearLayerStacksV1AggregateLedger::new(
            stack_indices
                .map(|layer_indices| Qwen35MetalW8LinearLayerStack3BufferLedger {
                    layer_indices,
                    ledger: stack_ledger,
                })
                .to_vec(),
            full_indices
                .map(|layer_index| Qwen35MetalW8MlpBlockBufferLedger {
                    layer_index,
                    ledger: mlp_ledger,
                })
                .to_vec(),
        );

        assert_eq!(aggregate.total_persistent_mtlbuffer_bytes, 528_605_184);
        assert_eq!(aggregate.allocated_buffers, 504);
        assert_eq!(aggregate.shared_buffers, 444);
        assert_eq!(aggregate.private_buffers, 60);
        assert_eq!(aggregate.host_to_device_bytes_per_decode, 49_152);
        assert_eq!(aggregate.device_to_host_bytes_per_decode, 49_152);
        assert_eq!(aggregate.command_buffers_per_decode, 12);
        assert_eq!(aggregate.compute_encoders_per_decode, 36);
        assert_eq!(aggregate.commits_per_decode, 12);
        assert_eq!(aggregate.waits_per_decode, 12);
        assert_eq!(aggregate.intermediate_host_finite_checks_per_decode, 0);
        assert_eq!(aggregate.final_output_finite_checks_per_decode, 6);
    }

    #[cfg(feature = "metal-experiments")]
    #[test]
    fn metal_mlp_stack3_boundary_body_v1_qwen35_08b_static_ledger_is_exact() {
        let initial_stack = Qwen35MetalW8LinearLayerStack3BufferLedger {
            layer_indices: [0, 1, 2],
            ledger: apxinf_metal::LinearLayerStack3BufferLedger {
                gdn_core_profile: apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch,
                gdn_function_chain: apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch
                    .expected_function_chain(),
                allocated_buffers: 76,
                shared_buffers: 68,
                private_buffers: 8,
                packed_weight_bytes: 64_585_728,
                packed_scale_bytes: 4_429_824,
                f32_parameter_bytes: 321_408,
                active_state_bytes: 3_440_640,
                scratch_state_bytes: 3_440_640,
                activation_bytes: 133_248,
                total_persistent_bytes: 76_351_488,
                host_input_bytes_per_decode: 4_096,
                host_output_bytes_per_decode: 4_096,
                state_host_transfer_bytes_per_decode: 0,
                command_buffers_per_decode: 1,
                compute_encoders_per_decode: 3,
                kernel_dispatches_per_decode: 39,
                explicit_buffer_barriers_per_decode: 36,
                gdn_core_seams_per_decode: 3,
                gdn_core_kernel_dispatches_per_decode: 12,
                gdn_core_explicit_buffer_barriers_per_decode: 12,
                gdn_core_recurrent_or_fused_threads_per_threadgroup: 256,
                gdn_core_threadgroups_per_decode: 126,
                gdn_core_launched_threads_per_decode: 30_864,
                gdn_core_source_declared_threadgroup_memory_bytes: 0,
                gdn_core_expected_pipeline_static_threadgroup_memory_bytes: 0,
                gdn_core_internal_threadgroup_barrier_sites_per_threadgroup: 0,
                commits_per_decode: 1,
                waits_per_decode: 1,
                intermediate_host_finite_checks_per_decode: 0,
                final_output_finite_checks_per_decode: 1,
            },
        };
        let boundary_ledger = apxinf_metal::MlpStack3BoundaryBufferLedgerV1 {
            gdn_core_profile: apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch,
            gdn_function_chain: apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch
                .expected_function_chain(),
            scope: "resident-mtlbuffer-only",
            exclusions: "boundary exclusions",
            abi_version: 1,
            stack_depth: 3,
            allocated_buffers: 81,
            shared_buffers: 73,
            private_buffers: 8,
            packed_weight_bytes: 75_595_776,
            packed_scale_bytes: 5_117_952,
            f32_parameter_bytes: 325_504,
            active_state_bytes: 3_440_640,
            scratch_state_bytes: 3_440_640,
            activation_bytes: 133_248,
            total_persistent_bytes: 88_053_760,
            host_input_bytes_per_decode: 4_096,
            host_output_bytes_per_decode: 4_096,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 1,
            compute_encoders_per_decode: 4,
            kernel_dispatches_per_decode: 44,
            explicit_buffer_barriers_per_decode: 40,
            gdn_core_seams_per_decode: 3,
            gdn_core_kernel_dispatches_per_decode: 12,
            gdn_core_explicit_buffer_barriers_per_decode: 12,
            gdn_core_recurrent_or_fused_threads_per_threadgroup: 256,
            gdn_core_threadgroups_per_decode: 126,
            gdn_core_launched_threads_per_decode: 30_864,
            gdn_core_source_declared_threadgroup_memory_bytes: 0,
            gdn_core_expected_pipeline_static_threadgroup_memory_bytes: 0,
            gdn_core_internal_threadgroup_barrier_sites_per_threadgroup: 0,
            commits_per_decode: 1,
            waits_per_decode: 1,
            intermediate_host_finite_checks_per_decode: 0,
            final_output_finite_checks_per_decode: 1,
        };
        let final_mlp = Qwen35MetalW8MlpBlockBufferLedger {
            layer_index: 23,
            ledger: apxinf_metal::MlpBlockBufferLedger {
                scope: "resident-mtlbuffer-only",
                allocated_buffers: 8,
                shared_buffers: 6,
                private_buffers: 2,
                packed_weight_bytes: 11_010_048,
                packed_scale_bytes: 688_128,
                activation_bytes: 51_200,
                total_persistent_bytes: 11_749_376,
                host_input_bytes_per_decode: 4_096,
                host_output_bytes_per_decode: 4_096,
                state_host_transfer_bytes_per_decode: 0,
                command_buffers_per_decode: 1,
                compute_encoders_per_decode: 3,
                commits_per_decode: 1,
                waits_per_decode: 1,
            },
        };
        let boundaries = QWEN35_MLP_STACK3_BOUNDARY_REGIONS_V1
            .map(|(boundary_mlp_layer_index, stack_layer_indices)| {
                Qwen35MetalW8MlpStack3BoundaryRegionBufferLedgerV1 {
                    boundary_mlp_layer_index,
                    stack_layer_indices,
                    ledger: boundary_ledger,
                }
            })
            .to_vec();

        let aggregate = Qwen35MetalW8MlpStack3BoundaryBodyV1AggregateLedger::new(
            initial_stack,
            boundaries,
            final_mlp,
        )
        .unwrap();

        assert_eq!(aggregate.total_persistent_mtlbuffer_bytes, 528_369_664);
        assert_eq!(aggregate.allocated_buffers, 489);
        assert_eq!(aggregate.shared_buffers, 439);
        assert_eq!(aggregate.private_buffers, 50);
        assert_eq!(aggregate.host_to_device_bytes_per_decode, 28_672);
        assert_eq!(aggregate.device_to_host_bytes_per_decode, 28_672);
        assert_eq!(aggregate.state_host_transfer_bytes_per_decode, 0);
        assert_eq!(aggregate.command_buffers_per_decode, 7);
        assert_eq!(aggregate.compute_encoders_per_decode, 26);
        assert_eq!(aggregate.kernel_dispatches_per_decode, 262);
        assert_eq!(aggregate.commits_per_decode, 7);
        assert_eq!(aggregate.waits_per_decode, 7);
        assert_eq!(aggregate.intermediate_host_finite_checks_per_decode, 0);
        assert_eq!(aggregate.final_output_finite_checks_per_decode, 6);
    }

    #[cfg(feature = "metal-experiments")]
    #[test]
    fn metal_boundary_tail_head_gdn_core_fused_v1_is_explicit_fixed_shape_and_qk_closed() {
        let (config, tensors) = fixture();
        let ordinary =
            GeneralQwen35::from_weights(config.clone(), tensors, Device::Cpu, 16).unwrap();
        assert!(ordinary
            .metal_w8_mlp_stack3_boundary_tail_head_v1_stats()
            .is_none());
        assert!(ordinary
            .metal_w8_mlp_stack3_boundary_tail_head_v1_aggregate_ledger()
            .is_none());
        let ordinary_receipt = ordinary.generation_path_receipt().unwrap();
        assert_eq!(
            ordinary_receipt["format"],
            "apxinf-qwen35-generation-path-v1"
        );
        assert_eq!(ordinary_receipt["metal_w8_mlp_block"], false);
        assert_eq!(ordinary_receipt["metal_w8_lm_head"], false);
        assert!(ordinary_receipt.get("gdn_core_profile").is_none());

        let shape_error =
            GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_fused_v1(
                config.clone(),
                HashMap::new(),
                Device::Cpu,
                16,
            )
            .err()
            .expect("small-shape fused production route must fail closed");
        assert!(shape_error
            .to_string()
            .contains("fixed to H=1024/KH=16/VH=16/KD=128/VD=128/conv=4/eps=1e-6"));

        let qk_error = GeneralQwen35::from_weights_with_metal_w8_mlp_stack3_boundary_tail_head_gdn_core_profile_v1(
            config,
            HashMap::new(),
            Device::Cpu,
            16,
            apxinf_metal::GdnCoreProfileV1::QkStagedFourDispatch,
        )
        .err()
        .expect("diagnostic-only qk-staged control must not enter the production route");
        assert!(qk_error.to_string().contains("diagnostic-only qk-staged"));

        assert_eq!(
            qwen35_boundary_tail_head_mechanism_for_gdn_core_profile_v1(
                apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch,
            ),
            "metal-w8-mlp-stack3-boundary-tail-head-v1"
        );
        assert_eq!(
            qwen35_boundary_tail_head_mechanism_for_gdn_core_profile_v1(
                apxinf_metal::GdnCoreProfileV1::Fused128,
            ),
            "metal-w8-mlp-stack3-boundary-tail-head-gdn-core-fused-v1"
        );
        assert_eq!(
            qwen35_stack3_mechanism_for_gdn_core_profile_v1(
                apxinf_metal::GdnCoreProfileV1::Fused128,
            ),
            "metal-w8-linear-layer-stack3-gdn-core-fused-v1"
        );
        assert_eq!(
            qwen35_boundary_mechanism_for_gdn_core_profile_v1(
                apxinf_metal::GdnCoreProfileV1::Fused128,
            ),
            "metal-w8-mlp-stack3-boundary-gdn-core-fused-v1"
        );
    }

    #[cfg(feature = "metal-experiments")]
    #[test]
    fn metal_boundary_tail_head_v1_qwen35_08b_profile_ledgers_are_exact() {
        for (
            profile,
            initial_dispatches,
            initial_barriers,
            boundary_dispatches,
            boundary_barriers,
            expected_dispatches,
            expected_barriers,
        ) in [
            (
                apxinf_metal::GdnCoreProfileV1::LegacyFourDispatch,
                39,
                36,
                44,
                40,
                267,
                243,
            ),
            (
                apxinf_metal::GdnCoreProfileV1::Fused128,
                30,
                27,
                35,
                31,
                213,
                189,
            ),
        ] {
            let initial_stack = Qwen35MetalW8LinearLayerStack3BufferLedger {
                layer_indices: [0, 1, 2],
                ledger: apxinf_metal::LinearLayerStack3BufferLedger {
                    gdn_core_profile: profile,
                    gdn_function_chain: profile.expected_function_chain(),
                    allocated_buffers: 76,
                    shared_buffers: 68,
                    private_buffers: 8,
                    packed_weight_bytes: 64_585_728,
                    packed_scale_bytes: 4_429_824,
                    f32_parameter_bytes: 321_408,
                    active_state_bytes: 3_440_640,
                    scratch_state_bytes: 3_440_640,
                    activation_bytes: 133_248,
                    total_persistent_bytes: 76_351_488,
                    host_input_bytes_per_decode: 4_096,
                    host_output_bytes_per_decode: 4_096,
                    state_host_transfer_bytes_per_decode: 0,
                    command_buffers_per_decode: 1,
                    compute_encoders_per_decode: 3,
                    kernel_dispatches_per_decode: initial_dispatches,
                    explicit_buffer_barriers_per_decode: initial_barriers,
                    gdn_core_seams_per_decode: 3,
                    gdn_core_kernel_dispatches_per_decode: profile.gdn_core_dispatches_for_seams(3)
                        as usize,
                    gdn_core_explicit_buffer_barriers_per_decode: profile
                        .gdn_core_dispatches_for_seams(3)
                        as usize,
                    gdn_core_recurrent_or_fused_threads_per_threadgroup: profile
                        .recurrent_threads_per_threadgroup()
                        as usize,
                    gdn_core_threadgroups_per_decode: profile.gdn_core_threadgroups_for_seams(3)
                        as usize,
                    gdn_core_launched_threads_per_decode: profile
                        .gdn_core_launched_threads_for_seams(3)
                        as usize,
                    gdn_core_source_declared_threadgroup_memory_bytes: profile
                        .source_declared_threadgroup_memory_bytes()
                        as usize,
                    gdn_core_expected_pipeline_static_threadgroup_memory_bytes: profile
                        .expected_pipeline_static_threadgroup_memory_bytes()
                        as usize,
                    gdn_core_internal_threadgroup_barrier_sites_per_threadgroup: profile
                        .internal_threadgroup_barrier_sites_per_threadgroup()
                        as usize,
                    commits_per_decode: 1,
                    waits_per_decode: 1,
                    intermediate_host_finite_checks_per_decode: 0,
                    final_output_finite_checks_per_decode: 1,
                },
            };
            let boundary_ledger = apxinf_metal::MlpStack3BoundaryBufferLedgerV1 {
                gdn_core_profile: profile,
                gdn_function_chain: profile.expected_function_chain(),
                scope: "resident-mtlbuffer-only",
                exclusions: "boundary exclusions",
                abi_version: 1,
                stack_depth: 3,
                allocated_buffers: 81,
                shared_buffers: 73,
                private_buffers: 8,
                packed_weight_bytes: 75_595_776,
                packed_scale_bytes: 5_117_952,
                f32_parameter_bytes: 325_504,
                active_state_bytes: 3_440_640,
                scratch_state_bytes: 3_440_640,
                activation_bytes: 133_248,
                total_persistent_bytes: 88_053_760,
                host_input_bytes_per_decode: 4_096,
                host_output_bytes_per_decode: 4_096,
                state_host_transfer_bytes_per_decode: 0,
                command_buffers_per_decode: 1,
                compute_encoders_per_decode: 4,
                kernel_dispatches_per_decode: boundary_dispatches,
                explicit_buffer_barriers_per_decode: boundary_barriers,
                gdn_core_seams_per_decode: 3,
                gdn_core_kernel_dispatches_per_decode: profile.gdn_core_dispatches_for_seams(3)
                    as usize,
                gdn_core_explicit_buffer_barriers_per_decode: profile
                    .gdn_core_dispatches_for_seams(3)
                    as usize,
                gdn_core_recurrent_or_fused_threads_per_threadgroup: profile
                    .recurrent_threads_per_threadgroup()
                    as usize,
                gdn_core_threadgroups_per_decode: profile.gdn_core_threadgroups_for_seams(3)
                    as usize,
                gdn_core_launched_threads_per_decode: profile.gdn_core_launched_threads_for_seams(3)
                    as usize,
                gdn_core_source_declared_threadgroup_memory_bytes: profile
                    .source_declared_threadgroup_memory_bytes()
                    as usize,
                gdn_core_expected_pipeline_static_threadgroup_memory_bytes: profile
                    .expected_pipeline_static_threadgroup_memory_bytes()
                    as usize,
                gdn_core_internal_threadgroup_barrier_sites_per_threadgroup: profile
                    .internal_threadgroup_barrier_sites_per_threadgroup()
                    as usize,
                commits_per_decode: 1,
                waits_per_decode: 1,
                intermediate_host_finite_checks_per_decode: 0,
                final_output_finite_checks_per_decode: 1,
            };
            let boundaries = QWEN35_MLP_STACK3_BOUNDARY_REGIONS_V1
                .map(|(boundary_mlp_layer_index, stack_layer_indices)| {
                    Qwen35MetalW8MlpStack3BoundaryRegionBufferLedgerV1 {
                        boundary_mlp_layer_index,
                        stack_layer_indices,
                        ledger: boundary_ledger,
                    }
                })
                .to_vec();
            let tail =
                apxinf_metal::TailMlpHeadBufferLedgerV1::from_dimensions(1_024, 3_584, 248_320)
                    .unwrap();
            let aggregate = Qwen35MetalW8MlpStack3BoundaryTailHeadV1AggregateLedger::new(
                initial_stack,
                boundaries,
                23,
                tail,
            )
            .unwrap();

            assert_eq!(aggregate.gdn_core_profile, profile);
            assert_eq!(
                aggregate.gdn_function_chain,
                profile.expected_function_chain()
            );
            assert_eq!(aggregate.total_persistent_mtlbuffer_bytes, 799_543_312);
            assert_eq!(aggregate.allocated_buffers, 494);
            assert_eq!(aggregate.shared_buffers, 443);
            assert_eq!(aggregate.private_buffers, 51);
            assert_eq!(aggregate.host_to_device_bytes_per_decode, 28_672);
            assert_eq!(aggregate.device_to_host_bytes_per_decode, 28_688);
            assert_eq!(aggregate.state_host_transfer_bytes_per_decode, 0);
            assert_eq!(aggregate.command_buffers_per_decode, 7);
            assert_eq!(aggregate.compute_encoders_per_decode, 24);
            assert_eq!(aggregate.kernel_dispatches_per_decode, expected_dispatches);
            assert_eq!(
                aggregate.explicit_buffer_barriers_per_decode,
                expected_barriers
            );
            assert_eq!(aggregate.commits_per_decode, 7);
            assert_eq!(aggregate.waits_per_decode, 7);
            assert!(aggregate.exclusions.contains("F32 tied embedding"));
            assert!(aggregate.exclusions.contains("four-candidate rerank"));
        }
    }

    #[cfg(feature = "metal-experiments")]
    #[test]
    fn metal_stack3_head_v2_static_ledger_adds_the_existing_tied_head_once() {
        let body = Qwen35MetalW8LinearLayerStacksV1AggregateLedger {
            scope: "resident-mtlbuffer-only",
            exclusions: "body exclusions",
            includes_lm_head: false,
            stacks: Vec::new(),
            full_attention_mlp_layers: Vec::new(),
            total_persistent_mtlbuffer_bytes: 528_605_184,
            allocated_buffers: 504,
            shared_buffers: 444,
            private_buffers: 60,
            host_to_device_bytes_per_decode: 49_152,
            device_to_host_bytes_per_decode: 49_152,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 12,
            compute_encoders_per_decode: 36,
            commits_per_decode: 12,
            waits_per_decode: 12,
            intermediate_host_finite_checks_per_decode: 0,
            final_output_finite_checks_per_decode: 6,
        };
        let head = apxinf_metal::LmHeadBufferLedger::from_dimensions(248_320, 1_024).unwrap();

        let aggregate = Qwen35MetalW8Stack3LmHeadV2AggregateLedger::new(body, head).unwrap();

        assert_eq!(aggregate.scope, "resident-mtlbuffer-only");
        assert!(aggregate.includes_lm_head);
        assert_eq!(aggregate.body.total_persistent_mtlbuffer_bytes, 528_605_184);
        assert_eq!(aggregate.lm_head.total_persistent_bytes, 271_169_552);
        assert_eq!(aggregate.total_persistent_mtlbuffer_bytes, 799_774_736);
        assert_eq!(aggregate.allocated_buffers, 509);
        assert_eq!(aggregate.shared_buffers, 448);
        assert_eq!(aggregate.private_buffers, 61);
        assert_eq!(aggregate.host_to_device_bytes_per_call, 53_248);
        assert_eq!(aggregate.device_to_host_bytes_per_call, 49_168);
        assert_eq!(aggregate.state_host_transfer_bytes_per_call, 0);
        assert_eq!(aggregate.command_buffers_per_call, 13);
        assert_eq!(aggregate.compute_encoders_per_call, 38);
        assert_eq!(aggregate.commits_per_call, 13);
        assert_eq!(aggregate.waits_per_call, 13);
        assert_eq!(aggregate.intermediate_host_finite_checks_per_call, 0);
        assert_eq!(aggregate.final_output_finite_checks_per_call, 6);
        assert!(aggregate.exclusions.contains("host F32 tied embedding"));
        assert!(aggregate.exclusions.contains("F32 rerank"));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_stack3_v1_fault_keeps_all_three_states_atomic_and_latches_the_body_lane() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_v1(
            config,
            tensors,
            Device::Cpu,
            16,
        )
        .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.inject_metal_w8_stack3_failure_once_for_test(0);

        let error = diagnostic.forward_hidden(&[3], 2).unwrap_err();

        assert!(error.to_string().contains("stack3-v1"));
        assert!(error.to_string().contains("injected"));
        assert!(error.to_string().contains("terminal"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 2);
        let failed = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert!(failed.terminal_error);
        assert_eq!(failed.stacks[0].execution.decode_calls, 1);
        assert_eq!(failed.stacks[0].execution.successful_decodes, 0);
        assert_eq!(failed.stacks[0].execution.failed_decodes, 1);
        assert_eq!(failed.stacks[0].execution.state_commits, 0);
        assert_eq!(failed.stacks[0].execution.last_state_commit_mask, 0);
        assert_eq!(failed.stacks[0].execution.committed_stack_version, 0);
        assert!(failed.stacks[1..]
            .iter()
            .all(|stack| stack.execution.decode_calls == 0));
        assert!(failed
            .full_attention_mlp_layers
            .iter()
            .all(|stats| stats.decode_calls == 0));

        let retry = diagnostic.forward_hidden(&[3], 2).unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(
            diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap(),
            failed,
            "terminal preflight must reject without another submission"
        );

        diagnostic.reset();
        let reset = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert!(!reset.terminal_error);
        assert!(reset.stacks.iter().all(|stack| {
            stack.prefill_seed_calls == [0, 0, 0]
                && stack.execution == Default::default()
                && !stack.terminal_error
        }));
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[3], 2).unwrap();
        assert!(diagnostic
            .metal_w8_linear_layer_stacks_v1_stats()
            .unwrap()
            .stacks
            .iter()
            .all(|stack| stack.execution.committed_stack_version == 1));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_stack3_v1_latches_the_whole_lane_when_full_layer_three_fails_after_stack_zero_commits()
    {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_v1(
            config,
            tensors,
            Device::Cpu,
            16,
        )
        .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.inject_failure_after_layer_once_for_test(3);

        let error = diagnostic.forward_hidden(&[3], 2).unwrap_err();

        assert!(error.to_string().contains("stack3-v1"));
        assert!(error.to_string().contains("injected after layer 3"));
        assert!(error.to_string().contains("terminal"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 2);
        let failed = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert!(failed.terminal_error);
        assert_eq!(failed.stacks[0].execution.successful_decodes, 1);
        assert_eq!(failed.stacks[0].execution.failed_decodes, 0);
        assert_eq!(failed.stacks[0].execution.state_commits, 3);
        assert_eq!(failed.stacks[0].execution.last_state_commit_mask, 0b111);
        assert_eq!(failed.stacks[0].execution.committed_stack_version, 1);
        assert!(failed.stacks[1..]
            .iter()
            .all(|stack| stack.execution.decode_calls == 0));
        assert_eq!(failed.full_attention_mlp_layers[0].layer_index, 3);
        assert_eq!(failed.full_attention_mlp_layers[0].decode_calls, 1);
        assert!(failed.full_attention_mlp_layers[1..]
            .iter()
            .all(|stats| stats.decode_calls == 0));

        let retry = diagnostic.forward_hidden(&[3], 2).unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 2);
        assert_eq!(
            diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap(),
            failed,
            "terminal preflight must reject retry before any stack or full MLP advances twice"
        );

        diagnostic.reset();
        assert_eq!(diagnostic.state.position(), 0);
        let reset = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert!(!reset.terminal_error);
        assert!(reset.stacks.iter().all(|stack| {
            stack.prefill_seed_calls == [0, 0, 0]
                && stack.execution == Default::default()
                && !stack.terminal_error
        }));
        assert!(reset
            .full_attention_mlp_layers
            .iter()
            .all(|stats| stats.decode_calls == 0));

        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[3], 2).unwrap();
        let recovered = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert!(!recovered.terminal_error);
        assert!(recovered.stacks.iter().all(|stack| {
            stack.execution.successful_decodes == 1
                && stack.execution.state_commits == 3
                && stack.execution.committed_stack_version == 1
        }));
        assert!(recovered
            .full_attention_mlp_layers
            .iter()
            .all(|stats| stats.decode_calls == 1));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_stack3_v1_partial_prefill_is_terminal_until_reset() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_all_linear_layer_stacks_v1(
            config,
            tensors,
            Device::Cpu,
            16,
        )
        .unwrap();
        diagnostic.inject_failure_after_layer_once_for_test(1);

        let error = diagnostic.forward_hidden(&[1, 2], 0).unwrap_err();

        assert!(error.to_string().contains("injected after layer 1"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 0);
        let failed = diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap();
        assert!(failed.terminal_error);
        assert_eq!(failed.stacks[0].prefill_seed_calls, [1, 1, 0]);
        assert_eq!(failed.stacks[0].execution.decode_calls, 0);
        assert!(failed.stacks[1..]
            .iter()
            .all(|stack| stack.prefill_seed_calls == [0, 0, 0]));
        let retry = diagnostic.forward_hidden(&[1, 2], 0).unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(
            diagnostic.metal_w8_linear_layer_stacks_v1_stats().unwrap(),
            failed
        );

        diagnostic.reset();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        assert!(diagnostic
            .metal_w8_linear_layer_stacks_v1_stats()
            .unwrap()
            .stacks
            .iter()
            .all(|stack| stack.prefill_seed_calls == [1, 1, 1]));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_precision_v2_all_linear_layers_latch_the_whole_lane_after_partial_commit() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_all_linear_layers_precision_v2(
                config,
                tensors,
                Device::Cpu,
                16,
                Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2,
            )
            .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.inject_failure_after_layer_once_for_test(3);

        let error = diagnostic.forward_hidden(&[3], 2).unwrap_err();

        assert!(error.to_string().contains("all-linear"));
        assert!(error.to_string().contains("injected after layer 3"));
        assert!(error.to_string().contains("terminal"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 2);
        let failed = diagnostic
            .metal_w8_all_linear_layers_precision_v2_stats()
            .unwrap();
        assert!(failed.terminal_error);
        assert!(failed.linear_layers[..3].iter().all(|stats| {
            stats.execution.successful_decodes == 1
                && stats.execution.failed_decodes == 0
                && stats.execution.committed_state_version == 1
        }));
        assert!(failed.linear_layers[3..]
            .iter()
            .all(|stats| stats.execution.decode_calls == 0));
        assert_eq!(failed.full_attention_mlp_layers[0].layer_index, 3);
        assert_eq!(failed.full_attention_mlp_layers[0].decode_calls, 1);

        let retry = diagnostic.forward_hidden(&[3], 2).unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(
            diagnostic
                .metal_w8_all_linear_layers_precision_v2_stats()
                .unwrap(),
            failed,
            "terminal preflight must reject retry before any lane advances twice"
        );

        diagnostic.reset();
        let reset = diagnostic
            .metal_w8_all_linear_layers_precision_v2_stats()
            .unwrap();
        assert!(!reset.terminal_error);
        assert!(reset.linear_layers.iter().all(|stats| {
            stats.execution.prefill_seed_calls == 0
                && stats.execution.decode_calls == 0
                && stats.execution.failed_decodes == 0
                && stats.execution.committed_state_version == 0
                && !stats.execution.terminal_error
        }));
        assert!(reset
            .full_attention_mlp_layers
            .iter()
            .all(|stats| stats.decode_calls == 0));
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[3], 2).unwrap();
        let recovered = diagnostic
            .metal_w8_all_linear_layers_precision_v2_stats()
            .unwrap();
        assert!(!recovered.terminal_error);
        assert!(recovered.linear_layers.iter().all(|stats| {
            stats.execution.successful_decodes == 1 && stats.execution.committed_state_version == 1
        }));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_precision_v2_all_linear_layers_fail_closed_after_partial_prefill() {
        let (config, tensors) = metal_all_linear_layers_fixture();
        let mut diagnostic =
            GeneralQwen35::from_weights_with_metal_w8_all_linear_layers_precision_v2(
                config,
                tensors,
                Device::Cpu,
                16,
                Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2,
            )
            .unwrap();
        diagnostic.inject_failure_after_layer_once_for_test(3);

        let error = diagnostic.forward_hidden(&[1, 2], 0).unwrap_err();

        assert!(error.to_string().contains("injected after layer 3"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 0);
        let failed = diagnostic
            .metal_w8_all_linear_layers_precision_v2_stats()
            .unwrap();
        assert!(failed.terminal_error);
        assert!(failed.linear_layers[..3].iter().all(|stats| {
            stats.execution.prefill_seed_calls == 1
                && stats.execution.decode_calls == 0
                && stats.execution.committed_state_version == 0
        }));
        assert!(failed.linear_layers[3..]
            .iter()
            .all(|stats| stats.execution.prefill_seed_calls == 0));

        let retry = diagnostic.forward_hidden(&[1, 2], 0).unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(
            diagnostic
                .metal_w8_all_linear_layers_precision_v2_stats()
                .unwrap(),
            failed,
            "a dirty partial prefill must not be replayed before reset"
        );

        diagnostic.reset();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        assert_eq!(diagnostic.state.position(), 2);
        let recovered = diagnostic
            .metal_w8_all_linear_layers_precision_v2_stats()
            .unwrap();
        assert!(!recovered.terminal_error);
        assert!(recovered
            .linear_layers
            .iter()
            .all(|stats| stats.execution.prefill_seed_calls == 1));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_precision_v2_fault_is_terminal_transactional_and_resettable() {
        let (config, tensors) = metal_linear_layer_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_linear_layer_precision_v2(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
            Qwen35MetalW8LinearLayerPrecisionProfile::GdnOutG32V2,
        )
        .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        let state_before = diagnostic
            .metal_w8_linear_layer
            .as_ref()
            .unwrap()
            .block
            .state_snapshot()
            .unwrap();
        diagnostic.inject_metal_w8_linear_layer_failure_once_for_test();

        let error = diagnostic.forward_hidden(&[3], 2).unwrap_err();

        assert!(error.to_string().contains("injected"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 2);
        assert_eq!(
            diagnostic
                .metal_w8_linear_layer
                .as_ref()
                .unwrap()
                .block
                .state_snapshot()
                .unwrap(),
            state_before
        );
        let failed = diagnostic
            .metal_w8_linear_layer_precision_v2_stats()
            .unwrap();
        assert_eq!(failed.execution.decode_calls, 1);
        assert_eq!(failed.execution.successful_decodes, 0);
        assert_eq!(failed.execution.failed_decodes, 1);
        assert_eq!(failed.execution.committed_state_version, 0);
        assert!(failed.execution.terminal_error);

        let retry = diagnostic.forward_hidden(&[3], 2).unwrap_err();
        assert!(retry.to_string().contains("reset required"));
        assert_eq!(
            diagnostic
                .metal_w8_linear_layer_precision_v2_stats()
                .unwrap(),
            failed
        );

        diagnostic.reset();
        let reset = diagnostic
            .metal_w8_linear_layer_precision_v2_stats()
            .unwrap();
        assert_eq!(reset.profile, failed.profile);
        assert_eq!(reset.execution.decode_calls, 0);
        assert_eq!(reset.execution.failed_decodes, 0);
        assert_eq!(reset.execution.committed_state_version, 0);
        assert!(!reset.execution.terminal_error);
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[3], 2).unwrap();
        assert_eq!(diagnostic.state.position(), 3);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_w8_linear_layer_fault_is_terminal_and_commits_no_state_or_position() {
        let (config, tensors) = metal_linear_layer_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_linear_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        let metal_state_before = diagnostic
            .metal_w8_linear_layer
            .as_ref()
            .unwrap()
            .block
            .state_snapshot()
            .unwrap();
        let cpu_state_before = gdn_decode_state_from_cpu(
            &diagnostic.config.text,
            diagnostic.state.linear_state(0).unwrap(),
        )
        .unwrap();
        diagnostic.inject_metal_w8_linear_layer_failure_once_for_test();

        let error = diagnostic.forward_hidden(&[3], 2).unwrap_err();

        assert!(error.to_string().contains("injected"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 2);
        assert_eq!(
            diagnostic
                .metal_w8_linear_layer
                .as_ref()
                .unwrap()
                .block
                .state_snapshot()
                .unwrap(),
            metal_state_before
        );
        assert_eq!(
            gdn_decode_state_from_cpu(
                &diagnostic.config.text,
                diagnostic.state.linear_state(0).unwrap(),
            )
            .unwrap(),
            cpu_state_before
        );
        let failed = diagnostic.metal_w8_linear_layer_stats().unwrap();
        assert_eq!(failed.prefill_seed_calls, 1);
        assert_eq!(failed.decode_calls, 1);
        assert_eq!(failed.successful_decodes, 0);
        assert_eq!(failed.failed_decodes, 1);
        assert_eq!(failed.command_buffers, 1);
        assert_eq!(failed.compute_encoders, 1);
        assert_eq!(failed.commits, 1);
        assert_eq!(failed.waits, 1);
        assert_eq!(failed.host_to_device_bytes, 64 * 4);
        assert_eq!(failed.device_to_host_bytes, 0);
        assert_eq!(failed.committed_state_version, 0);
        assert!(failed.terminal_error);
        assert!(failed.block_elapsed_ns > 0);

        let retry_error = diagnostic.forward_hidden(&[3], 2).unwrap_err();
        assert!(retry_error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 2);
        assert_eq!(diagnostic.metal_w8_linear_layer_stats().unwrap(), failed);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos", debug_assertions))]
    #[test]
    fn metal_w8_linear_layer_latches_terminal_when_a_later_layer_fails() {
        let (config, tensors) = metal_linear_layer_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_linear_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        let before = diagnostic.metal_w8_linear_layer_stats().unwrap();
        diagnostic.inject_failure_after_layer_once_for_test(1);

        let error = diagnostic.forward_hidden(&[3], 2).unwrap_err();

        assert!(error.to_string().contains("injected after layer 1"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 2);
        let failed = diagnostic.metal_w8_linear_layer_stats().unwrap();
        assert_eq!(failed.decode_calls, before.decode_calls + 1);
        assert_eq!(failed.successful_decodes, before.successful_decodes + 1);
        assert_eq!(
            failed.committed_state_version,
            before.committed_state_version + 1
        );
        assert!(failed.terminal_error);

        let retry_error = diagnostic.forward_hidden(&[3], 2).unwrap_err();
        assert!(retry_error.to_string().contains("reset required"));
        assert_eq!(diagnostic.state.position(), 2);
        assert_eq!(
            diagnostic.metal_w8_linear_layer_stats().unwrap(),
            failed,
            "terminal preflight must reject the retry before the lane advances again"
        );

        diagnostic.reset();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[3], 2).unwrap();
        assert_eq!(diagnostic.state.position(), 3);
        assert!(
            !diagnostic
                .metal_w8_linear_layer_stats()
                .unwrap()
                .terminal_error
        );
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_w8_linear_layer_reset_clears_ownership_receipts_and_fault_latch() {
        let (config, tensors) = metal_linear_layer_fixture();
        let mut diagnostic = GeneralQwen35::from_weights_with_metal_w8_linear_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();
        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[3], 2).unwrap();
        assert_eq!(diagnostic.state.position(), 3);
        assert_eq!(
            diagnostic
                .metal_w8_linear_layer_stats()
                .unwrap()
                .committed_state_version,
            1
        );

        diagnostic.reset();

        assert_eq!(diagnostic.state.position(), 0);
        let reset = diagnostic.metal_w8_linear_layer_stats().unwrap();
        assert_eq!(
            reset,
            Qwen35MetalW8LinearLayerStats {
                layer_index: 0,
                prefill_seed_calls: 0,
                decode_calls: 0,
                successful_decodes: 0,
                failed_decodes: 0,
                command_buffers: 0,
                compute_encoders: 0,
                commits: 0,
                waits: 0,
                host_to_device_bytes: 0,
                device_to_host_bytes: 0,
                committed_state_version: 0,
                terminal_error: false,
                block_elapsed_ns: 0,
            }
        );
        assert!(diagnostic
            .metal_w8_linear_layer
            .as_ref()
            .unwrap()
            .block
            .state_snapshot()
            .is_err());

        diagnostic.forward_hidden(&[1, 2], 0).unwrap();
        diagnostic.forward_hidden(&[3], 2).unwrap();
        let fresh = diagnostic.metal_w8_linear_layer_stats().unwrap();
        assert_eq!(fresh.prefill_seed_calls, 1);
        assert_eq!(fresh.decode_calls, 1);
        assert_eq!(fresh.committed_state_version, 1);
        assert!(!fresh.terminal_error);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_w8_linear_layer_constructor_rejects_full_attention_selection() {
        let (config, tensors) = metal_linear_layer_fixture();
        let error = GeneralQwen35::from_weights_with_metal_w8_linear_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            3,
        )
        .err()
        .expect("a full-attention layer cannot silently become a complete linear layer lane");
        assert!(error.to_string().contains("not linear attention"));
    }

    #[cfg(feature = "metal-experiments")]
    #[test]
    fn packed_w8_linear_layer_reference_is_explicit_seeds_and_owns_decode() {
        let (config, tensors) = metal_linear_layer_fixture();
        let mut cpu =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        assert!(cpu.packed_w8_linear_layer_reference_stats().is_none());
        let mut reference = GeneralQwen35::from_weights_with_packed_w8_linear_layer_reference(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();

        let cpu_prefill = cpu.forward_hidden(&[1, 2], 0).unwrap();
        let reference_prefill = reference.forward_hidden(&[1, 2], 0).unwrap();
        assert_close(&reference_prefill, &cpu_prefill, 1.0e-7);
        let cpu_state_after_prefill = gdn_decode_state_from_cpu(
            &reference.config.text,
            reference.state.linear_state(0).unwrap(),
        )
        .unwrap();
        assert_eq!(
            reference
                .packed_w8_linear_layer_reference
                .as_ref()
                .unwrap()
                .state
                .as_ref()
                .unwrap(),
            &cpu_state_after_prefill
        );
        let prefill = reference.packed_w8_linear_layer_reference_stats().unwrap();
        assert_eq!(prefill.prefill_seed_calls, 1);
        assert_eq!(prefill.decode_calls, 0);

        for (token, position) in [(3, 2), (4, 3)] {
            let cpu_hidden = cpu.forward_hidden(&[token], position).unwrap();
            let reference_hidden = reference.forward_hidden(&[token], position).unwrap();
            assert_close(&reference_hidden, &cpu_hidden, 1.0e-2);
        }

        assert_eq!(
            gdn_decode_state_from_cpu(
                &reference.config.text,
                reference.state.linear_state(0).unwrap(),
            )
            .unwrap(),
            cpu_state_after_prefill,
            "selected CPU layer state must remain frozen after the packed reference takes ownership"
        );
        let stats = reference.packed_w8_linear_layer_reference_stats().unwrap();
        assert_eq!(stats.layer_index, 0);
        assert_eq!(stats.prefill_seed_calls, 1);
        assert_eq!(stats.decode_calls, 2);
        assert_eq!(stats.successful_decodes, 2);
        assert_eq!(stats.failed_decodes, 0);
        assert_eq!(stats.committed_state_version, 2);
        assert!(!stats.terminal_error);
        assert!(stats.block_elapsed_ns > 0);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn packed_reference_precision_profiles_only_change_the_selected_projection_groups() {
        let (config, tensors) = metal_linear_layer_fixture();
        let legacy = GeneralQwen35::from_weights_with_packed_w8_linear_layer_reference(
            config.clone(),
            tensors.clone(),
            Device::Cpu,
            16,
            0,
        )
        .unwrap()
        .packed_w8_linear_layer_reference_stats()
        .unwrap();
        assert_eq!(
            legacy.profile,
            Qwen35PackedW8LinearLayerReferenceProfile::G64
        );

        for (profile, gdn_output_group, mlp_down_group) in [
            (
                Qwen35PackedW8LinearLayerReferenceProfile::G64,
                apxinf_metal::W8GroupSize::G64,
                apxinf_metal::W8GroupSize::G64,
            ),
            (
                Qwen35PackedW8LinearLayerReferenceProfile::GdnOutG32,
                apxinf_metal::W8GroupSize::G32,
                apxinf_metal::W8GroupSize::G64,
            ),
            (
                Qwen35PackedW8LinearLayerReferenceProfile::MlpDownG32,
                apxinf_metal::W8GroupSize::G64,
                apxinf_metal::W8GroupSize::G32,
            ),
            (
                Qwen35PackedW8LinearLayerReferenceProfile::GdnOutAndMlpDownG32,
                apxinf_metal::W8GroupSize::G32,
                apxinf_metal::W8GroupSize::G32,
            ),
        ] {
            let stats = GeneralQwen35::from_weights_with_packed_w8_linear_layer_reference_profile(
                config.clone(),
                tensors.clone(),
                Device::Cpu,
                16,
                0,
                profile,
            )
            .unwrap()
            .packed_w8_linear_layer_reference_stats()
            .unwrap();
            let ledger = stats.quantization;
            assert_eq!(stats.profile, profile);
            assert_eq!(ledger.gdn_input_group_size, apxinf_metal::W8GroupSize::G64);
            assert_eq!(ledger.gdn_output_group_size, gdn_output_group);
            assert_eq!(ledger.mlp_gate_group_size, apxinf_metal::W8GroupSize::G64);
            assert_eq!(ledger.mlp_up_group_size, apxinf_metal::W8GroupSize::G64);
            assert_eq!(ledger.mlp_down_group_size, mlp_down_group);
            assert_eq!(
                ledger.gdn_output_scale_bytes,
                legacy.quantization.gdn_output_scale_bytes
                    * if gdn_output_group == apxinf_metal::W8GroupSize::G32 {
                        2
                    } else {
                        1
                    }
            );
            assert_eq!(
                ledger.mlp_down_scale_bytes,
                legacy.quantization.mlp_down_scale_bytes
                    * if mlp_down_group == apxinf_metal::W8GroupSize::G32 {
                        2
                    } else {
                        1
                    }
            );
            assert_eq!(
                ledger.gdn_input_scale_bytes,
                legacy.quantization.gdn_input_scale_bytes
            );
            assert_eq!(
                ledger.mlp_gate_scale_bytes,
                legacy.quantization.mlp_gate_scale_bytes
            );
            assert_eq!(
                ledger.mlp_up_scale_bytes,
                legacy.quantization.mlp_up_scale_bytes
            );
        }
    }

    #[cfg(all(feature = "metal-experiments", debug_assertions))]
    #[test]
    fn packed_w8_linear_layer_reference_fault_is_terminal_and_transactional() {
        let (config, tensors) = metal_linear_layer_fixture();
        let mut reference = GeneralQwen35::from_weights_with_packed_w8_linear_layer_reference(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();
        reference.forward_hidden(&[1, 2], 0).unwrap();
        let packed_state_before = reference
            .packed_w8_linear_layer_reference
            .as_ref()
            .unwrap()
            .state
            .clone()
            .unwrap();
        let cpu_state_before = gdn_decode_state_from_cpu(
            &reference.config.text,
            reference.state.linear_state(0).unwrap(),
        )
        .unwrap();
        reference.inject_packed_w8_linear_layer_reference_failure_once_for_test();

        let error = reference.forward_hidden(&[3], 2).unwrap_err();

        assert!(error.to_string().contains("injected"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(reference.state.position(), 2);
        assert_eq!(
            reference
                .packed_w8_linear_layer_reference
                .as_ref()
                .unwrap()
                .state
                .as_ref()
                .unwrap(),
            &packed_state_before
        );
        assert_eq!(
            gdn_decode_state_from_cpu(
                &reference.config.text,
                reference.state.linear_state(0).unwrap(),
            )
            .unwrap(),
            cpu_state_before
        );
        let failed = reference.packed_w8_linear_layer_reference_stats().unwrap();
        assert_eq!(failed.decode_calls, 1);
        assert_eq!(failed.successful_decodes, 0);
        assert_eq!(failed.failed_decodes, 1);
        assert_eq!(failed.committed_state_version, 0);
        assert!(failed.terminal_error);

        let retry_error = reference.forward_hidden(&[3], 2).unwrap_err();
        assert!(retry_error.to_string().contains("reset required"));
        assert_eq!(reference.state.position(), 2);
        assert_eq!(
            reference.packed_w8_linear_layer_reference_stats().unwrap(),
            failed
        );
    }

    #[cfg(all(feature = "metal-experiments", debug_assertions))]
    #[test]
    fn packed_w8_linear_layer_reference_latches_terminal_when_a_later_layer_fails() {
        let (config, tensors) = metal_linear_layer_fixture();
        let mut reference = GeneralQwen35::from_weights_with_packed_w8_linear_layer_reference(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();
        reference.forward_hidden(&[1, 2], 0).unwrap();
        let before = reference.packed_w8_linear_layer_reference_stats().unwrap();
        reference.inject_failure_after_layer_once_for_test(1);

        let error = reference.forward_hidden(&[3], 2).unwrap_err();

        assert!(error.to_string().contains("injected after layer 1"));
        assert!(error.to_string().contains("reset required"));
        assert_eq!(reference.state.position(), 2);
        let failed = reference.packed_w8_linear_layer_reference_stats().unwrap();
        assert_eq!(failed.decode_calls, before.decode_calls + 1);
        assert_eq!(failed.successful_decodes, before.successful_decodes + 1);
        assert_eq!(
            failed.committed_state_version,
            before.committed_state_version + 1
        );
        assert!(failed.terminal_error);

        let retry_error = reference.forward_hidden(&[3], 2).unwrap_err();
        assert!(retry_error.to_string().contains("reset required"));
        assert_eq!(reference.state.position(), 2);
        assert_eq!(
            reference.packed_w8_linear_layer_reference_stats().unwrap(),
            failed,
            "terminal preflight must reject the retry before the lane advances again"
        );

        reference.reset();
        reference.forward_hidden(&[1, 2], 0).unwrap();
        reference.forward_hidden(&[3], 2).unwrap();
        assert_eq!(reference.state.position(), 3);
        assert!(
            !reference
                .packed_w8_linear_layer_reference_stats()
                .unwrap()
                .terminal_error
        );
    }

    #[cfg(feature = "metal-experiments")]
    #[test]
    fn packed_w8_linear_layer_reference_reset_clears_state_and_receipts() {
        let (config, tensors) = metal_linear_layer_fixture();
        let mut reference = GeneralQwen35::from_weights_with_packed_w8_linear_layer_reference(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();
        reference.forward_hidden(&[1, 2], 0).unwrap();
        reference.forward_hidden(&[3], 2).unwrap();

        reference.reset();

        assert_eq!(reference.state.position(), 0);
        let reset = reference.packed_w8_linear_layer_reference_stats().unwrap();
        assert_eq!(reset.prefill_seed_calls, 0);
        assert_eq!(reset.decode_calls, 0);
        assert_eq!(reset.successful_decodes, 0);
        assert_eq!(reset.failed_decodes, 0);
        assert_eq!(reset.committed_state_version, 0);
        assert!(!reset.terminal_error);
        assert_eq!(reset.block_elapsed_ns, 0);
        assert!(reference
            .packed_w8_linear_layer_reference
            .as_ref()
            .unwrap()
            .state
            .is_none());

        reference.forward_hidden(&[1, 2], 0).unwrap();
        reference.forward_hidden(&[3], 2).unwrap();
        let fresh = reference.packed_w8_linear_layer_reference_stats().unwrap();
        assert_eq!(fresh.prefill_seed_calls, 1);
        assert_eq!(fresh.decode_calls, 1);
        assert_eq!(fresh.committed_state_version, 1);
    }

    #[cfg(feature = "metal-experiments")]
    #[test]
    fn packed_w8_linear_layer_reference_rejects_full_attention_selection() {
        let (config, tensors) = metal_linear_layer_fixture();
        let error = GeneralQwen35::from_weights_with_packed_w8_linear_layer_reference(
            config,
            tensors,
            Device::Cpu,
            16,
            3,
        )
        .err()
        .expect("a full-attention layer cannot silently become a packed reference lane");
        assert!(error.to_string().contains("not linear attention"));
    }

    #[test]
    fn q_projection_is_deinterleaved_per_head() {
        let raw = Tensor::from_f32(vec![8, 1], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
        let (query, gate) = transpose_interleaved_q_gate(&raw, 2, 2, 1).unwrap();
        assert_eq!(query.as_f32().unwrap(), &[0.0, 1.0, 4.0, 5.0]);
        assert_eq!(gate.as_f32().unwrap(), &[2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn tied_embeddings_do_not_materialize_a_second_lm_head() {
        let (config, tensors) = fixture();
        let model = GeneralQwen35::from_weights(config, tensors, Device::Cpu, 16).unwrap();

        assert!(model.config.text.tie_word_embeddings);
        assert!(model.weights.lm_head.is_none());
        assert_eq!(
            model.weights.token_embedding.shape().dims(),
            &[model.config.text.vocab_size, model.config.text.hidden_size]
        );
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_w8_body_is_explicit_and_only_runs_for_selected_decode_layer() {
        let (config, tensors) = metal_width_fixture();
        let mut cpu =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        let mut metal = GeneralQwen35::from_weights_with_metal_w8_body_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();

        let cpu_prefill = cpu.forward(&[1, 2], 0).unwrap();
        let metal_prefill = metal.forward(&[1, 2], 0).unwrap();
        assert_close(&metal_prefill, &cpu_prefill, 1.0e-7);
        assert_eq!(metal.metal_w8_body_stats().unwrap().decode_calls, 0);

        let cpu_decode = cpu.forward(&[3], 2).unwrap();
        let metal_decode = metal.forward(&[3], 2).unwrap();
        assert_close(&metal_decode, &cpu_decode, 5.0e-4);
        let stats = metal.metal_w8_body_stats().unwrap();
        assert_eq!(stats.layer_index, 0);
        assert_eq!(stats.decode_calls, 1);
        assert!(stats.projection_elapsed_ns > 0);
    }

    #[cfg(all(feature = "metal-w8-head-mlp", target_os = "macos"))]
    #[test]
    fn metal_w8_mlp_block_is_explicit_and_only_runs_for_selected_decode_layer() {
        let (config, tensors) = metal_mlp_block_fixture();
        let mut cpu =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        assert!(cpu.metal_w8_mlp_block_layer_stats().is_empty());
        let mut metal = GeneralQwen35::from_weights_with_metal_w8_mlp_block_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();

        let cpu_prefill = cpu.forward(&[1, 2], 0).unwrap();
        let metal_prefill = metal.forward(&[1, 2], 0).unwrap();
        assert_close(&metal_prefill, &cpu_prefill, 1.0e-7);
        assert_eq!(metal.metal_w8_mlp_block_stats().unwrap().decode_calls, 0);

        let cpu_decode = cpu.forward(&[3], 2).unwrap();
        let metal_decode = metal.forward(&[3], 2).unwrap();
        assert_close(&metal_decode, &cpu_decode, 2.0e-3);
        let stats = metal.metal_w8_mlp_block_stats().unwrap();
        assert_eq!(stats.layer_index, 0);
        assert_eq!(stats.decode_calls, 1);
        assert!(stats.block_elapsed_ns > 0);
    }

    #[cfg(all(feature = "metal-w8-head-mlp", target_os = "macos"))]
    #[test]
    fn metal_w8_mlp_block_layer_set_hits_every_selected_layer_once_per_decode() {
        let (config, tensors) = metal_mlp_block_fixture();
        let mut model = GeneralQwen35::from_weights_with_metal_w8_mlp_block_layers(
            config,
            tensors,
            Device::Cpu,
            16,
            &[0, 2],
        )
        .unwrap();
        model.forward(&[1, 2], 0).unwrap();
        assert!(model
            .metal_w8_mlp_block_layer_stats()
            .iter()
            .all(|stats| stats.decode_calls == 0));

        model.forward(&[3], 2).unwrap();
        let stats = model.metal_w8_mlp_block_layer_stats();
        assert_eq!(
            stats
                .iter()
                .map(|stats| stats.layer_index)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert!(stats.iter().all(|stats| stats.decode_calls == 1));
        assert!(stats.iter().all(|stats| stats.block_elapsed_ns > 0));
    }

    #[cfg(all(feature = "metal-w8-head-mlp", target_os = "macos"))]
    #[test]
    fn metal_w8_combined_prefill_and_decode_hit_each_lane_exactly() {
        let (config, tensors) = metal_mlp_block_fixture();
        let mut cpu =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        let mut combined = GeneralQwen35::from_weights_with_metal_w8_mlp_blocks_and_lm_head(
            config,
            tensors,
            Device::Cpu,
            16,
        )
        .unwrap();

        let cpu_prefill = cpu.prefill_for_generation(LlmInput::text(&[1, 2])).unwrap();
        let cpu_prefill_token = argmax_f32_row(&cpu_prefill, cpu.vocab_size()).unwrap();
        let combined_prefill_token = combined
            .prefill_token_for_generation(LlmInput::text(&[1, 2]))
            .expect("combined lane must own the generation prefill hook")
            .unwrap();
        assert_eq!(combined_prefill_token, cpu_prefill_token);
        assert!(combined
            .metal_w8_mlp_block_layer_stats()
            .iter()
            .all(|stats| stats.decode_calls == 0));
        let head_stats = combined.metal_w8_lm_head_stats().unwrap();
        assert_eq!(head_stats.prefill_calls, 1);
        assert_eq!(head_stats.decode_calls, 0);
        assert_eq!(head_stats.teacher_calls, 0);

        let cpu_decode = cpu.forward(&[3], 2).unwrap();
        let cpu_decode_token = argmax_f32_row(&cpu_decode, cpu.vocab_size()).unwrap();
        let combined_decode_token = combined
            .decode_token(3, 2)
            .expect("combined lane must own the decode hook")
            .unwrap();
        assert_eq!(combined_decode_token, cpu_decode_token);
        let block_stats = combined.metal_w8_mlp_block_layer_stats();
        assert_eq!(block_stats.len(), combined.config.text.n_layers);
        assert!(block_stats.iter().all(|stats| stats.decode_calls == 1));
        let head_stats = combined.metal_w8_lm_head_stats().unwrap();
        assert_eq!(head_stats.prefill_calls, 1);
        assert_eq!(head_stats.decode_calls, 1);
        assert_eq!(head_stats.teacher_calls, 0);
        assert!(head_stats.topk_elapsed_ns > 0);
        assert!(head_stats.rerank_elapsed_ns > 0);
        let receipt = combined.generation_path_receipt().unwrap();
        assert_eq!(receipt["metal_w8_mlp_block"], true);
        assert_eq!(receipt["metal_w8_lm_head"], true);
        assert_eq!(receipt["mlp_block_layers"].as_array().unwrap().len(), 4);
        assert!(receipt["mlp_block_layers"]
            .as_array()
            .unwrap()
            .iter()
            .all(|layer| layer["decode_calls"] == 1));
        assert_eq!(receipt["lm_head"]["prefill_calls"], 1);
        assert_eq!(receipt["lm_head"]["decode_calls"], 1);
    }

    #[cfg(all(feature = "metal-w8-head-mlp", target_os = "macos"))]
    #[test]
    fn registry_metal_options_select_mlp_only_or_combined_without_double_build() {
        let (config, tensors) = metal_mlp_block_fixture();
        let mlp_only = GeneralQwen35::from_weights_with_backend_metal_options(
            config.clone(),
            tensors.clone(),
            create_backend(Device::Cpu).unwrap(),
            16,
            false,
            true,
        )
        .unwrap();
        let mlp_receipt = mlp_only.generation_path_receipt().unwrap();
        assert_eq!(mlp_receipt["metal_w8_mlp_block"], true);
        assert_eq!(mlp_receipt["metal_w8_lm_head"], false);
        assert_eq!(mlp_receipt["mlp_block_layers"].as_array().unwrap().len(), 4);
        assert!(mlp_receipt["lm_head"].is_null());

        let combined = GeneralQwen35::from_weights_with_backend_metal_options(
            config,
            tensors,
            create_backend(Device::Cpu).unwrap(),
            16,
            true,
            true,
        )
        .unwrap();
        let combined_receipt = combined.generation_path_receipt().unwrap();
        assert_eq!(combined_receipt["metal_w8_mlp_block"], true);
        assert_eq!(combined_receipt["metal_w8_lm_head"], true);
        assert_eq!(
            combined_receipt["mlp_block_layers"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        assert!(combined_receipt["lm_head"].is_object());
    }

    #[cfg(all(feature = "metal-w8-head-mlp", target_os = "macos"))]
    #[test]
    fn metal_w8_mlp_block_layer_set_rejects_invalid_selections() {
        let (config, tensors) = metal_mlp_block_fixture();
        let empty = GeneralQwen35::from_weights_with_metal_w8_mlp_block_layers(
            config.clone(),
            tensors.clone(),
            Device::Cpu,
            16,
            &[],
        )
        .err()
        .expect("an empty MLP block layer set must fail closed");
        assert!(empty.to_string().contains("at least one selected layer"));

        let duplicate = GeneralQwen35::from_weights_with_metal_w8_mlp_block_layers(
            config.clone(),
            tensors.clone(),
            Device::Cpu,
            16,
            &[1, 1],
        )
        .err()
        .expect("a duplicate MLP block layer set must fail closed");
        assert!(duplicate.to_string().contains("selected more than once"));

        let invalid_layer = config.text.n_layers;
        let outside = GeneralQwen35::from_weights_with_metal_w8_mlp_block_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            invalid_layer,
        )
        .err()
        .expect("an out-of-range MLP block layer must fail closed");
        assert!(outside.to_string().contains("outside 0..4"));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_w8_body_rejects_an_out_of_range_layer_without_fallback() {
        let (config, tensors) = metal_width_fixture();
        let invalid_layer = config.text.n_layers;
        let error = GeneralQwen35::from_weights_with_metal_w8_body_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            invalid_layer,
        )
        .err()
        .expect("an invalid Metal body layer must fail closed");
        assert!(error.to_string().contains("outside 0..4"));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_w8_body_does_not_treat_a_one_token_prefill_as_decode() {
        let (config, tensors) = metal_width_fixture();
        let mut model = GeneralQwen35::from_weights_with_metal_w8_body_layer(
            config,
            tensors,
            Device::Cpu,
            16,
            0,
        )
        .unwrap();
        model.forward(&[1], 0).unwrap();
        assert_eq!(model.metal_w8_body_stats().unwrap().decode_calls, 0);
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_w8_body_layer_set_hits_every_selected_layer_once_per_decode() {
        let (config, tensors) = metal_width_fixture();
        let mut model = GeneralQwen35::from_weights_with_metal_w8_body_layers(
            config,
            tensors,
            Device::Cpu,
            16,
            &[0, 2],
        )
        .unwrap();
        model.forward(&[1, 2], 0).unwrap();
        assert_eq!(
            model.metal_w8_body_layer_stats(),
            vec![
                Qwen35MetalW8BodyStats {
                    layer_index: 0,
                    decode_calls: 0,
                    projection_elapsed_ns: 0,
                },
                Qwen35MetalW8BodyStats {
                    layer_index: 2,
                    decode_calls: 0,
                    projection_elapsed_ns: 0,
                },
            ]
        );
        model.forward(&[3], 2).unwrap();
        let stats = model.metal_w8_body_layer_stats();
        assert_eq!(
            stats
                .iter()
                .map(|stats| stats.layer_index)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert!(stats.iter().all(|stats| stats.decode_calls == 1));
        assert!(stats.iter().all(|stats| stats.projection_elapsed_ns > 0));
    }

    #[cfg(all(feature = "metal-experiments", target_os = "macos"))]
    #[test]
    fn metal_w8_body_layer_set_rejects_empty_and_duplicate_selections() {
        let (config, tensors) = metal_width_fixture();
        let empty = GeneralQwen35::from_weights_with_metal_w8_body_layers(
            config.clone(),
            tensors.clone(),
            Device::Cpu,
            16,
            &[],
        )
        .err()
        .expect("an empty body layer set must fail closed");
        assert!(empty.to_string().contains("at least one selected layer"));

        let duplicate = GeneralQwen35::from_weights_with_metal_w8_body_layers(
            config,
            tensors,
            Device::Cpu,
            16,
            &[1, 1],
        )
        .err()
        .expect("a duplicate body layer set must fail closed");
        assert!(duplicate.to_string().contains("selected more than once"));
    }

    #[cfg(feature = "metal-w8-head-mlp")]
    #[test]
    fn f32_candidate_rerank_is_order_independent_and_uses_lowest_tie() {
        let hidden = [1.0f32, 2.0];
        let embedding = [
            0.0, 0.0, // token 0: 0
            1.0, 1.0, // token 1: 3 (tie, lower token)
            2.0, 0.5, // token 2: 3
            0.5, 0.5, // token 3: 1.5
        ];
        assert_eq!(
            rerank_tied_f32_candidates(&embedding, &hidden, 4, 2, [3, 2, 0, 1]).unwrap(),
            1
        );
    }

    #[test]
    fn cpu_generation_receipt_keeps_the_plain_path_when_optional_routes_are_compiled() {
        let (config, tensors) = fixture();
        let mut model = GeneralQwen35::from_weights(config, tensors, Device::Cpu, 16).unwrap();
        let (tokens, _) = model
            .generate_streaming(LlmInput::text(&[1, 2]), 2, |_| {}, None)
            .unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(
            model.generation_path_receipt().unwrap(),
            serde_json::json!({
                "format": "apxinf-qwen35-generation-path-v1",
                "metal_w8_mlp_block": false,
                "metal_w8_lm_head": false,
                "mlp_block_layers": [],
                "lm_head": null,
            })
        );
    }

    #[test]
    fn untied_lm_head_keeps_the_packed_projection_path() {
        let raw = MINI_CONFIG.replacen(
            "\"tie_word_embeddings\": true",
            "\"tie_word_embeddings\": false",
            1,
        );
        let config = Qwen35Config::from_json_str(&raw).unwrap();
        let schema = Qwen35WeightSchema::new(&config).unwrap();
        let tensors = schema
            .specs()
            .iter()
            .map(|spec| {
                let count = spec.shape.iter().product();
                (
                    spec.name.clone(),
                    Tensor::from_f32(spec.shape.clone(), &tensor_values(&spec.name, count))
                        .unwrap(),
                )
            })
            .collect();
        let mut model = GeneralQwen35::from_weights(config, tensors, Device::Cpu, 16).unwrap();

        let lm_head = model.weights.lm_head.as_ref().unwrap();
        assert_eq!(
            lm_head.shape().dims(),
            &[model.config.text.hidden_size, model.config.text.vocab_size]
        );
        let logits = model.forward(&[1, 2], 0).unwrap();
        assert_eq!(logits.shape().dims(), &[2, model.config.text.vocab_size]);
    }

    #[test]
    fn prefill_then_decode_matches_one_shot_hybrid_state() {
        let (config, tensors) = fixture();
        let mut incremental =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        let mut one_shot = GeneralQwen35::from_weights(config, tensors, Device::Cpu, 16).unwrap();

        incremental.forward(&[1, 2, 3], 0).unwrap();
        let decode_logits = incremental.forward(&[4], 3).unwrap();
        let full_logits = one_shot.forward(&[1, 2, 3, 4], 0).unwrap();
        let vocab = one_shot.vocab_size();
        let full_last = Tensor::from_f32(
            vec![1, vocab],
            &full_logits.as_f32().unwrap()[3 * vocab..4 * vocab],
        )
        .unwrap();
        assert_close(&decode_logits, &full_last, 2.0e-5);

        assert_eq!(incremental.state.position(), 4);
        assert_eq!(one_shot.state.position(), 4);
        for layer in 0..incremental.config.text.n_layers {
            let Some(left) = incremental.state.linear_state(layer) else {
                continue;
            };
            let right = one_shot.state.linear_state(layer).unwrap();
            assert_eq!(
                left.recurrent().unwrap().shape().dims(),
                &[
                    incremental.config.text.linear_num_value_heads,
                    incremental.config.text.linear_key_head_dim,
                    incremental.config.text.linear_value_head_dim,
                ]
            );
            assert_close(
                left.recurrent().unwrap(),
                right.recurrent().unwrap(),
                2.0e-5,
            );
            let suffix_widths = [
                incremental.config.text.linear_key_width(),
                incremental.config.text.linear_key_width(),
                incremental.config.text.linear_value_width(),
            ];
            for ((left, right), width) in left
                .convolution_suffixes()
                .into_iter()
                .zip(right.convolution_suffixes())
                .zip(suffix_widths)
            {
                assert_eq!(
                    left.unwrap().shape().dims(),
                    &[incremental.config.text.linear_conv_kernel_dim, width]
                );
                assert_close(left.unwrap(), right.unwrap(), 1.0e-7);
            }
        }
    }

    #[test]
    fn generation_prefill_projects_only_the_last_row() {
        let (config, tensors) = fixture();
        let mut full =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        let mut generation =
            GeneralQwen35::from_weights(config.clone(), tensors.clone(), Device::Cpu, 16).unwrap();
        let mut streaming = GeneralQwen35::from_weights(config, tensors, Device::Cpu, 16).unwrap();
        let tokens = [1, 2, 3];

        let full_logits = full.forward(&tokens, 0).unwrap();
        let last_logits = generation
            .prefill_for_generation(LlmInput::text(&tokens))
            .unwrap();
        let vocab = full.vocab_size();

        assert_eq!(full_logits.shape().dims(), &[tokens.len(), vocab]);
        assert_eq!(last_logits.shape().dims(), &[1, vocab]);
        assert_eq!(full.state.position(), tokens.len());
        assert_eq!(generation.state.position(), tokens.len());

        let expected_last = Tensor::from_f32(
            vec![1, vocab],
            &full_logits.as_f32().unwrap()[(tokens.len() - 1) * vocab..tokens.len() * vocab],
        )
        .unwrap();
        assert_close(&last_logits, &expected_last, 1.0e-7);

        // The shared loop must infer the returned row count instead of
        // assuming generation prefill returned one row per prompt token.
        let mut observed = Vec::new();
        let (generated, _) = streaming
            .generate_streaming(
                LlmInput::text(&tokens),
                1,
                |token| observed.push(token),
                None,
            )
            .unwrap();
        assert_eq!(generated.len(), 1);
        assert_eq!(generated, observed);
        assert_eq!(streaming.state.position(), tokens.len());
    }

    #[test]
    fn zero_generation_budget_skips_prefill_and_state_mutation() {
        let (config, tensors) = fixture();
        let mut model = GeneralQwen35::from_weights(config, tensors, Device::Cpu, 16).unwrap();
        let mut observed = Vec::new();

        let (generated, profile) = model
            .generate_streaming(
                LlmInput::text(&[1, 2, 3]),
                0,
                |token| observed.push(token),
                None,
            )
            .unwrap();

        assert!(generated.is_empty());
        assert!(observed.is_empty());
        assert_eq!(profile.input_tokens(), 3);
        assert_eq!(profile.output_tokens(), 0);
        assert_eq!(model.state.position(), 0);
        for layer in 0..model.config.text.n_layers {
            let Some(state) = model.state.linear_state(layer) else {
                continue;
            };
            assert!(state.recurrent().is_none());
            assert!(state
                .convolution_suffixes()
                .into_iter()
                .all(|suffix| suffix.is_none()));
        }
    }

    #[test]
    fn over_budget_generation_fails_before_state_mutation() {
        let (config, tensors) = fixture();
        let mut model = GeneralQwen35::from_weights(config, tensors, Device::Cpu, 4).unwrap();
        let mut observed = Vec::new();

        let error = model
            .generate_streaming(
                LlmInput::text(&[1, 2, 3]),
                2,
                |token| observed.push(token),
                None,
            )
            .err()
            .expect("over-budget request should fail");

        assert!(error.to_string().contains("exceeds configured maximum 4"));
        assert!(observed.is_empty());
        assert_eq!(model.state.position(), 0);
        for layer in 0..model.config.text.n_layers {
            let Some(state) = model.state.linear_state(layer) else {
                continue;
            };
            assert!(state.recurrent().is_none());
            assert!(state
                .convolution_suffixes()
                .into_iter()
                .all(|suffix| suffix.is_none()));
        }
    }

    #[test]
    fn reset_restores_fresh_forward_and_bounds_are_checked() {
        let (config, tensors) = fixture();
        let mut model = GeneralQwen35::from_weights(config, tensors, Device::Cpu, 4).unwrap();
        let first = model.forward(&[2, 5], 0).unwrap();
        assert!(model.forward(&[7], 0).is_err());
        assert!(model.forward(&[7, 8, 9], 2).is_err());
        model.reset();
        let after_reset = model.forward(&[2, 5], 0).unwrap();
        assert_close(&first, &after_reset, 1.0e-7);
    }
}
