//! Decode-only Metal W8 projection with fused GPU argmax.
//!
//! The stable packed format is Hugging Face row-major `[rows, columns]`,
//! symmetric signed int8, 64 consecutive columns per group, and one little-
//! endian F32 scale per row/group. Values use `round()` followed by clamping
//! to `[-127, 127]`; an all-zero group records scale `1.0`.

#![cfg(feature = "w8-head-mlp")]

use std::error::Error;
use std::fmt::{Display, Formatter};

#[cfg(feature = "experiments")]
mod full_attention_decode_v1;
#[cfg(feature = "experiments")]
mod gdn;
#[cfg(feature = "experiments")]
mod gdn_core_fused_profile_v1;
#[cfg(feature = "experiments")]
mod gdn_recurrent_profile_v1;
#[cfg(feature = "experiments")]
mod linear_layer;
#[cfg(feature = "experiments")]
mod tail_mlp_head_v1;

#[cfg(feature = "experiments")]
pub use full_attention_decode_v1::*;
#[cfg(feature = "experiments")]
pub use gdn::{
    GdnDecodeResult, GdnDecodeState, GdnDimensions, GdnF32Weights, GdnMetalStats, MetalW8GdnBlock,
    PackedW8GdnBlock,
};
#[cfg(feature = "experiments")]
pub use gdn_core_fused_profile_v1::*;
#[cfg(feature = "experiments")]
pub use gdn_recurrent_profile_v1::{
    GdnRecurrentCount18RuntimeReceiptV1, GdnRecurrentCount18SnapshotV1, GdnRecurrentProfileV1,
    MetalGdnRecurrentCount18PrimitiveV1, QWEN35_GDN_CORE_ELEMENTS_PER_SEAM_V1,
    QWEN35_GDN_KEY_DIM_V1, QWEN35_GDN_KEY_HEADS_V1, QWEN35_GDN_PROCESSED_ELEMENTS_PER_SEAM_V1,
    QWEN35_GDN_PROJECTED_ELEMENTS_PER_SEAM_V1, QWEN35_GDN_RECURRENT_ELEMENTS_PER_SEAM_V1,
    QWEN35_GDN_RECURRENT_SEAMS_PER_DECODE_V1, QWEN35_GDN_VALUE_DIM_V1, QWEN35_GDN_VALUE_HEADS_V1,
};
#[cfg(feature = "experiments")]
pub use linear_layer::{
    LinearLayerBufferLedger, LinearLayerDecodeResult, LinearLayerMetalStats,
    LinearLayerQuantizationLedger, LinearLayerStack3BufferLedger, LinearLayerStack3MetalStats,
    MetalW8LinearLayerBlock, MetalW8LinearLayerStack3, MetalW8MlpStack3BoundaryV1,
    MlpStack3BoundaryBufferLedgerV1, MlpStack3BoundaryDecodeResultV1,
    MlpStack3BoundaryMetalStatsV1, PackedW8LinearLayerBlock, PackedW8MlpStack3BoundaryV1,
};
#[cfg(feature = "experiments")]
pub use tail_mlp_head_v1::{
    MetalW8TailMlpHeadV1, PackedW8TailMlpHeadV1, TailMlpHeadBufferLedgerV1,
    TailMlpHeadDecodeResultV1, TailMlpHeadDecodeViewV1, TailMlpHeadMetalStatsV1,
};

pub const W8_GROUP_SIZE: usize = 64;
pub const W8_TOP_K: usize = 4;

/// Quantization group sizes supported by the CPU packed-weight oracle.
///
/// Legacy Metal entry points remain ABI-locked to [`Self::G64`]. [`Self::G32`]
/// is also accepted by explicitly versioned precision APIs, currently the GDN
/// output projection in complete-layer v2 and stack3 v1; other uses remain
/// CPU precision screens and are rejected by legacy handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W8GroupSize {
    G32,
    G64,
}

impl W8GroupSize {
    pub const fn columns(self) -> usize {
        match self {
            Self::G32 => 32,
            Self::G64 => W8_GROUP_SIZE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalW8Error(String);

impl MetalW8Error {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for MetalW8Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for MetalW8Error {}

/// Row-wise/group-wise W8 representation used by the packed CPU oracle.
/// G64 is the canonical legacy Metal ABI. G32 is diagnostic data that may be
/// consumed only by an explicitly versioned precision-specific Metal API.
#[derive(Clone, Debug)]
pub struct PackedW8Rows {
    rows: usize,
    columns: usize,
    group_size: W8GroupSize,
    values: Vec<i8>,
    scales: Vec<f32>,
}

impl PackedW8Rows {
    pub fn pack_f32(source: &[f32], rows: usize, columns: usize) -> Result<Self, MetalW8Error> {
        Self::pack_f32_with_group_size(source, rows, columns, W8GroupSize::G64)
    }

    /// Explicit G32 precision packing. Legacy Metal handles reject this
    /// representation; only versioned precision-specific APIs may accept it.
    pub fn pack_f32_g32(source: &[f32], rows: usize, columns: usize) -> Result<Self, MetalW8Error> {
        Self::pack_f32_with_group_size(source, rows, columns, W8GroupSize::G32)
    }

    pub(crate) fn pack_f32_with_group_size(
        source: &[f32],
        rows: usize,
        columns: usize,
        group_size: W8GroupSize,
    ) -> Result<Self, MetalW8Error> {
        if rows == 0 || columns == 0 {
            return Err(MetalW8Error::new(
                "Metal W8 dimensions must both be greater than zero",
            ));
        }
        let group_columns = group_size.columns();
        if columns % group_columns != 0 || columns % 4 != 0 {
            return Err(MetalW8Error::new(format!(
                "Metal W8 columns must be divisible by {group_columns}, got {columns}"
            )));
        }
        let elements = rows
            .checked_mul(columns)
            .ok_or_else(|| MetalW8Error::new("Metal W8 dimensions overflow"))?;
        if source.len() != elements {
            return Err(MetalW8Error::new(format!(
                "Metal W8 source has {} elements, expected {elements}",
                source.len()
            )));
        }
        if let Some(index) = source.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 source contains a non-finite value at element {index}"
            )));
        }

        let groups_per_row = columns / group_columns;
        let scale_count = rows
            .checked_mul(groups_per_row)
            .ok_or_else(|| MetalW8Error::new("Metal W8 scale dimensions overflow"))?;
        let mut values = vec![0i8; elements];
        let mut scales = vec![1.0f32; scale_count];

        for row in 0..rows {
            let row_offset = row * columns;
            for group in 0..groups_per_row {
                let column_offset = group * group_columns;
                let start = row_offset + column_offset;
                let end = start + group_columns;
                let max_abs = source[start..end]
                    .iter()
                    .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                scales[row * groups_per_row + group] = scale;
                for index in start..end {
                    values[index] = (source[index] / scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }

        Ok(Self {
            rows,
            columns,
            group_size,
            values,
            scales,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn group_size(&self) -> W8GroupSize {
        self.group_size
    }

    pub fn values(&self) -> &[i8] {
        &self.values
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub(crate) fn require_metal_g64(&self, projection: &str) -> Result<(), MetalW8Error> {
        if self.group_size != W8GroupSize::G64 {
            return Err(MetalW8Error::new(format!(
                "Metal W8 {projection} requires group size 64, got {} (use a versioned precision-specific API for supported G32 projections)",
                self.group_size.columns()
            )));
        }
        Ok(())
    }

    /// Reference implementation for the quantized projection itself.
    pub fn scores(&self, input: &[f32]) -> Result<Vec<f32>, MetalW8Error> {
        if input.len() != self.columns {
            return Err(MetalW8Error::new(format!(
                "Metal W8 input has {} elements, expected {}",
                input.len(),
                self.columns
            )));
        }
        if let Some(index) = input.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 input contains a non-finite value at element {index}"
            )));
        }
        let group_columns = self.group_size.columns();
        let groups_per_row = self.columns / group_columns;
        let mut output = vec![0.0f32; self.rows];
        for (row, score) in output.iter_mut().enumerate() {
            let row_offset = row * self.columns;
            let mut sum = 0.0f32;
            for column in 0..self.columns {
                let group = column / group_columns;
                sum += self.values[row_offset + column] as f32
                    * self.scales[row * groups_per_row + group]
                    * input[column];
            }
            *score = sum;
        }
        Ok(output)
    }

    pub fn argmax(&self, input: &[f32]) -> Result<u32, MetalW8Error> {
        let scores = self.scores(input)?;
        argmax_scores(&scores)
    }

    /// Deterministic CPU oracle for the four highest quantized scores.
    pub fn topk4(&self, input: &[f32]) -> Result<[u32; W8_TOP_K], MetalW8Error> {
        let scores = self.scores(input)?;
        topk4_scores(&scores)
    }
}

/// Canonical packed weights for a decode `M=1` gated MLP.
///
/// Gate and up rows share one `[2 * intermediate, hidden]` projection. The
/// down projection is `[hidden, intermediate]`. [`Self::forward`] is the CPU
/// oracle for the exact packed weights consumed by the Metal block.
#[derive(Clone, Debug)]
pub struct PackedW8MlpBlock {
    pub(crate) gate_up: PackedW8Rows,
    pub(crate) down: PackedW8Rows,
}

/// Exact persistent-buffer and per-decode transaction contract for one
/// complete Metal W8 MLP block. Byte totals cover MTLBuffer allocations only;
/// CPU weights, host allocations, pipelines, libraries, queues, and driver
/// allocations are deliberately outside this ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlpBlockBufferLedger {
    pub scope: &'static str,
    pub allocated_buffers: usize,
    pub shared_buffers: usize,
    pub private_buffers: usize,
    pub packed_weight_bytes: usize,
    pub packed_scale_bytes: usize,
    pub activation_bytes: usize,
    pub total_persistent_bytes: usize,
    pub host_input_bytes_per_decode: usize,
    pub host_output_bytes_per_decode: usize,
    pub state_host_transfer_bytes_per_decode: usize,
    pub command_buffers_per_decode: usize,
    pub compute_encoders_per_decode: usize,
    pub commits_per_decode: usize,
    pub waits_per_decode: usize,
}

impl PackedW8MlpBlock {
    pub fn pack_f32(
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, MetalW8Error> {
        Self::pack_f32_with_down_group_size(
            gate,
            up,
            down,
            hidden_size,
            intermediate_size,
            W8GroupSize::G64,
        )
    }

    /// Precision-screen packer. Gate/up remain canonical G64; only the down
    /// projection may opt into CPU-only G32.
    pub fn pack_f32_with_down_group_size(
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
        down_group_size: W8GroupSize,
    ) -> Result<Self, MetalW8Error> {
        let projection_elements = hidden_size
            .checked_mul(intermediate_size)
            .ok_or_else(|| MetalW8Error::new("Metal W8 MLP dimensions overflow"))?;
        for (label, values) in [("gate", gate), ("up", up), ("down", down)] {
            if values.len() != projection_elements {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 MLP {label} has {} elements, expected {projection_elements}",
                    values.len()
                )));
            }
        }
        let gate_up_rows = intermediate_size
            .checked_mul(2)
            .ok_or_else(|| MetalW8Error::new("Metal W8 MLP dimensions overflow"))?;
        let gate_up_elements = gate_up_rows
            .checked_mul(hidden_size)
            .ok_or_else(|| MetalW8Error::new("Metal W8 MLP dimensions overflow"))?;
        let mut gate_up = Vec::with_capacity(gate_up_elements);
        gate_up.extend_from_slice(gate);
        gate_up.extend_from_slice(up);
        Ok(Self {
            gate_up: PackedW8Rows::pack_f32(&gate_up, gate_up_rows, hidden_size)?,
            down: PackedW8Rows::pack_f32_with_group_size(
                down,
                hidden_size,
                intermediate_size,
                down_group_size,
            )?,
        })
    }

    pub fn gate_up_group_size(&self) -> W8GroupSize {
        self.gate_up.group_size()
    }

    pub fn down_group_size(&self) -> W8GroupSize {
        self.down.group_size()
    }

    pub fn gate_up_scale_bytes(&self) -> usize {
        self.gate_up.scales().len() * std::mem::size_of::<f32>()
    }

    pub fn down_scale_bytes(&self) -> usize {
        self.down.scales().len() * std::mem::size_of::<f32>()
    }

    pub fn buffer_ledger(&self) -> Result<MlpBlockBufferLedger, MetalW8Error> {
        let hidden_size = self.down.rows;
        let intermediate_size = self.down.columns;
        if self.gate_up.rows != intermediate_size.saturating_mul(2)
            || self.gate_up.columns != hidden_size
        {
            return Err(MetalW8Error::new(
                "Metal W8 MLP packed projections have incompatible shapes",
            ));
        }
        let packed_weight_bytes = self
            .gate_up
            .values
            .len()
            .checked_add(self.down.values.len())
            .ok_or_else(|| MetalW8Error::new("Metal W8 MLP buffer ledger overflow"))?;
        let packed_scale_bytes = self
            .gate_up_scale_bytes()
            .checked_add(self.down_scale_bytes())
            .ok_or_else(|| MetalW8Error::new("Metal W8 MLP buffer ledger overflow"))?;
        let activation_elements = hidden_size
            .checked_mul(2)
            .and_then(|count| {
                intermediate_size
                    .checked_mul(3)
                    .and_then(|intermediate| count.checked_add(intermediate))
            })
            .ok_or_else(|| MetalW8Error::new("Metal W8 MLP buffer ledger overflow"))?;
        let activation_bytes = activation_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalW8Error::new("Metal W8 MLP buffer ledger overflow"))?;
        let total_persistent_bytes = packed_weight_bytes
            .checked_add(packed_scale_bytes)
            .and_then(|bytes| bytes.checked_add(activation_bytes))
            .ok_or_else(|| MetalW8Error::new("Metal W8 MLP buffer ledger overflow"))?;
        let host_row_bytes = hidden_size
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalW8Error::new("Metal W8 MLP buffer ledger overflow"))?;
        Ok(MlpBlockBufferLedger {
            scope: "resident-mtlbuffer-only",
            allocated_buffers: 8,
            shared_buffers: 6,
            private_buffers: 2,
            packed_weight_bytes,
            packed_scale_bytes,
            activation_bytes,
            total_persistent_bytes,
            host_input_bytes_per_decode: host_row_bytes,
            host_output_bytes_per_decode: host_row_bytes,
            state_host_transfer_bytes_per_decode: 0,
            command_buffers_per_decode: 1,
            compute_encoders_per_decode: 3,
            commits_per_decode: 1,
            waits_per_decode: 1,
        })
    }

    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, MetalW8Error> {
        let projected = self.gate_up.scores(input)?;
        let intermediate_size = self.down.columns;
        let mut activated = Vec::with_capacity(intermediate_size);
        for index in 0..intermediate_size {
            let gate = projected[index];
            let up = projected[index + intermediate_size];
            activated.push(gate / (1.0 + (-gate).exp()) * up);
        }
        self.down.scores(&activated)
    }
}

fn argmax_scores(scores: &[f32]) -> Result<u32, MetalW8Error> {
    if scores.is_empty() || scores.len() > u32::MAX as usize {
        return Err(MetalW8Error::new(
            "Metal W8 argmax requires 1..=u32::MAX scores",
        ));
    }
    let mut best_score = f32::NEG_INFINITY;
    let mut best_token = 0u32;
    for (token, &score) in scores.iter().enumerate() {
        if score > best_score {
            best_score = score;
            best_token = token as u32;
        }
    }
    Ok(best_token)
}

fn topk4_scores(scores: &[f32]) -> Result<[u32; W8_TOP_K], MetalW8Error> {
    if scores.len() < W8_TOP_K || scores.len() > u32::MAX as usize {
        return Err(MetalW8Error::new(format!(
            "Metal W8 top-4 requires {W8_TOP_K}..=u32::MAX scores"
        )));
    }
    let mut best_scores = [f32::NEG_INFINITY; W8_TOP_K];
    let mut best_tokens = [u32::MAX; W8_TOP_K];
    for (token, &score) in scores.iter().enumerate() {
        if score.is_nan()
            || !candidate_better(
                score,
                token as u32,
                best_scores[W8_TOP_K - 1],
                best_tokens[W8_TOP_K - 1],
            )
        {
            continue;
        }
        let mut position = W8_TOP_K - 1;
        while position > 0
            && candidate_better(
                score,
                token as u32,
                best_scores[position - 1],
                best_tokens[position - 1],
            )
        {
            best_scores[position] = best_scores[position - 1];
            best_tokens[position] = best_tokens[position - 1];
            position -= 1;
        }
        best_scores[position] = score;
        best_tokens[position] = token as u32;
    }
    Ok(best_tokens)
}

fn candidate_better(score: f32, token: u32, current_score: f32, current_token: u32) -> bool {
    score > current_score || (score == current_score && token < current_token)
}

/// Exact resident-buffer and per-call transaction contract for one Metal W8
/// tied language-model head. The ledger excludes the host F32 embedding used
/// by the exact four-candidate rerank, plus pipelines, libraries, queues,
/// command objects, and driver allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LmHeadBufferLedger {
    pub scope: &'static str,
    pub exclusions: &'static str,
    pub allocated_buffers: usize,
    pub shared_buffers: usize,
    pub private_buffers: usize,
    pub packed_weight_bytes: usize,
    pub packed_scale_bytes: usize,
    pub hidden_bytes: usize,
    pub partial_topk_bytes: usize,
    pub output_token_bytes: usize,
    pub total_persistent_bytes: usize,
    pub host_input_bytes_per_call: usize,
    pub host_output_bytes_per_call: usize,
    pub state_host_transfer_bytes_per_call: usize,
    pub command_buffers_per_call: usize,
    pub compute_encoders_per_call: usize,
    pub commits_per_call: usize,
    pub waits_per_call: usize,
}

impl LmHeadBufferLedger {
    /// Compute the ABI ledger without allocating the potentially very large
    /// packed vocabulary matrix or creating a Metal device handle.
    pub fn from_dimensions(rows: usize, columns: usize) -> Result<Self, MetalW8Error> {
        if rows < W8_TOP_K || columns == 0 || columns % W8_GROUP_SIZE != 0 {
            return Err(MetalW8Error::new(format!(
                "Metal W8 head ledger requires at least {W8_TOP_K} rows and non-zero columns divisible by {W8_GROUP_SIZE}"
            )));
        }
        if rows > u32::MAX as usize || columns > u32::MAX as usize {
            return Err(MetalW8Error::new(
                "Metal W8 head ledger dimensions exceed the u32 kernel contract",
            ));
        }
        let packed_weight_bytes = rows
            .checked_mul(columns)
            .ok_or_else(|| MetalW8Error::new("Metal W8 head ledger weight bytes overflow"))?;
        let packed_scale_bytes = rows
            .checked_mul(columns / W8_GROUP_SIZE)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| MetalW8Error::new("Metal W8 head ledger scale bytes overflow"))?;
        let hidden_bytes = columns
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MetalW8Error::new("Metal W8 head ledger hidden bytes overflow"))?;
        let partial_count = rows
            .checked_add(7)
            .ok_or_else(|| MetalW8Error::new("Metal W8 head ledger row count overflow"))?
            / 8;
        let partial_topk_bytes = partial_count
            .checked_mul(W8_TOP_K)
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| MetalW8Error::new("Metal W8 head ledger partial bytes overflow"))?;
        let output_token_bytes = W8_TOP_K * std::mem::size_of::<u32>();
        let total_persistent_bytes = [
            packed_weight_bytes,
            packed_scale_bytes,
            hidden_bytes,
            partial_topk_bytes,
            output_token_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| MetalW8Error::new("Metal W8 head ledger total bytes overflow"))?;
        Ok(Self {
            scope: "resident-mtlbuffer-only",
            exclusions: "host F32 tied embedding and four-candidate rerank, host allocations, Metal pipelines/libraries/queues, command objects, driver allocations, model body, and KV cache",
            allocated_buffers: 5,
            shared_buffers: 4,
            private_buffers: 1,
            packed_weight_bytes,
            packed_scale_bytes,
            hidden_bytes,
            partial_topk_bytes,
            output_token_bytes,
            total_persistent_bytes,
            host_input_bytes_per_call: hidden_bytes,
            host_output_bytes_per_call: output_token_bytes,
            state_host_transfer_bytes_per_call: 0,
            command_buffers_per_call: 1,
            compute_encoders_per_call: 2,
            commits_per_call: 1,
            waits_per_call: 1,
        })
    }
}

/// Persistent Metal resources for one tied W8 language-model head.
pub struct MetalW8LmHead {
    inner: platform::Handle,
    rows: usize,
    columns: usize,
}

impl MetalW8LmHead {
    pub fn from_packed(weights: &PackedW8Rows) -> Result<Self, MetalW8Error> {
        weights.require_metal_g64("LM head")?;
        if weights.rows < W8_TOP_K {
            return Err(MetalW8Error::new(format!(
                "Metal W8 head requires at least {W8_TOP_K} rows"
            )));
        }
        if weights.rows > u32::MAX as usize || weights.columns > u32::MAX as usize {
            return Err(MetalW8Error::new(
                "Metal W8 dimensions exceed the u32 kernel contract",
            ));
        }
        let inner = platform::Handle::new(weights)?;
        Ok(Self {
            inner,
            rows: weights.rows,
            columns: weights.columns,
        })
    }

    pub fn from_f32_rows(
        source: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Self, MetalW8Error> {
        let packed = PackedW8Rows::pack_f32(source, rows, columns)?;
        Self::from_packed(&packed)
    }

    pub fn buffer_ledger(&self) -> LmHeadBufferLedger {
        LmHeadBufferLedger::from_dimensions(self.rows, self.columns)
            .expect("constructed Metal W8 lm_head dimensions have a valid ledger")
    }

    /// Submit both reduction stages in one command buffer, wait exactly once,
    /// and copy only the four candidate token IDs back to the host.
    pub fn topk4(&mut self, hidden: &[f32]) -> Result<[u32; W8_TOP_K], MetalW8Error> {
        if hidden.len() != self.columns {
            return Err(MetalW8Error::new(format!(
                "Metal W8 hidden row has {} elements, expected {}",
                hidden.len(),
                self.columns
            )));
        }
        if let Some(index) = hidden.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 hidden row contains a non-finite value at element {index}"
            )));
        }
        let tokens = self.inner.topk4(hidden)?;
        for (index, &token) in tokens.iter().enumerate() {
            if token as usize >= self.rows {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 kernel returned candidate {index} token {token} outside vocabulary {}",
                    self.rows
                )));
            }
            if tokens[..index].contains(&token) {
                return Err(MetalW8Error::new(format!(
                    "Metal W8 kernel returned duplicate candidate token {token}"
                )));
            }
        }
        Ok(tokens)
    }

    /// Compatibility convenience for callers that need the quantized top-1.
    pub fn argmax(&mut self, hidden: &[f32]) -> Result<u32, MetalW8Error> {
        Ok(self.topk4(hidden)?[0])
    }
}

/// Persistent Metal resources for a generic decode `M=1` W8 projection.
///
/// The canonical packed weights are `[rows, columns]`. Each call transfers one
/// F32 input row and returns one F32 output row; weights and scales are uploaded
/// only when the handle is constructed.
#[cfg(feature = "experiments")]
pub struct MetalW8MatVec {
    inner: platform::MatVecHandle,
    rows: usize,
    columns: usize,
}

#[cfg(feature = "experiments")]
impl MetalW8MatVec {
    pub fn from_packed(weights: &PackedW8Rows) -> Result<Self, MetalW8Error> {
        weights.require_metal_g64("matvec")?;
        if weights.rows > u32::MAX as usize || weights.columns > u32::MAX as usize {
            return Err(MetalW8Error::new(
                "Metal W8 matvec dimensions exceed the u32 kernel contract",
            ));
        }
        Ok(Self {
            inner: platform::MatVecHandle::new(weights)?,
            rows: weights.rows,
            columns: weights.columns,
        })
    }

    pub fn from_f32_rows(
        source: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Self, MetalW8Error> {
        let packed = PackedW8Rows::pack_f32(source, rows, columns)?;
        Self::from_packed(&packed)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Submit one projection, wait once, and copy the complete F32 row back.
    pub fn multiply(&mut self, input: &[f32]) -> Result<Vec<f32>, MetalW8Error> {
        if input.len() != self.columns {
            return Err(MetalW8Error::new(format!(
                "Metal W8 matvec input has {} elements, expected {}",
                input.len(),
                self.columns
            )));
        }
        if let Some(index) = input.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 matvec input contains a non-finite value at element {index}"
            )));
        }
        self.inner.multiply(input, self.rows)
    }
}

/// Persistent Metal resources for one complete decode `M=1` gated MLP.
///
/// A call copies one hidden-width F32 row to Metal, submits gate+up W8,
/// SiLU-times-up, and down W8 in one command buffer, waits once, and copies one
/// hidden-width F32 row back. Packed weights, scales, and intermediate buffers
/// remain resident, and the host output allocation is reused across calls.
pub struct MetalW8MlpBlock {
    inner: platform::MlpBlockHandle,
    hidden_size: usize,
    intermediate_size: usize,
    output: Vec<f32>,
    buffer_ledger: MlpBlockBufferLedger,
}

impl MetalW8MlpBlock {
    pub fn from_packed(weights: &PackedW8MlpBlock) -> Result<Self, MetalW8Error> {
        weights
            .gate_up
            .require_metal_g64("MLP gate/up projection")?;
        weights.down.require_metal_g64("MLP down projection")?;
        let hidden_size = weights.down.rows;
        let intermediate_size = weights.down.columns;
        if weights.gate_up.rows != intermediate_size.saturating_mul(2)
            || weights.gate_up.columns != hidden_size
        {
            return Err(MetalW8Error::new(
                "Metal W8 MLP packed projections have incompatible shapes",
            ));
        }
        if hidden_size > u32::MAX as usize || intermediate_size > (u32::MAX as usize) / 2 {
            return Err(MetalW8Error::new(
                "Metal W8 MLP dimensions exceed the u32 kernel contract",
            ));
        }
        let buffer_ledger = weights.buffer_ledger()?;
        Ok(Self {
            inner: platform::MlpBlockHandle::new(weights)?,
            hidden_size,
            intermediate_size,
            output: vec![0.0; hidden_size],
            buffer_ledger,
        })
    }

    pub fn from_f32_weights(
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, MetalW8Error> {
        let packed = PackedW8MlpBlock::pack_f32(gate, up, down, hidden_size, intermediate_size)?;
        Self::from_packed(&packed)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    pub fn buffer_ledger(&self) -> MlpBlockBufferLedger {
        self.buffer_ledger
    }

    pub fn forward(&mut self, input: &[f32]) -> Result<&[f32], MetalW8Error> {
        if input.len() != self.hidden_size {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP input has {} elements, expected {}",
                input.len(),
                self.hidden_size
            )));
        }
        if let Some(index) = input.iter().position(|value| !value.is_finite()) {
            return Err(MetalW8Error::new(format!(
                "Metal W8 MLP input contains a non-finite value at element {index}"
            )));
        }
        self.inner.forward(input, &mut self.output)?;
        Ok(&self.output)
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{MetalW8Error, PackedW8MlpBlock, PackedW8Rows, W8_GROUP_SIZE, W8_TOP_K};
    use std::ffi::{c_char, c_int, c_void, CStr};
    use std::ptr::NonNull;

    const ERROR_CAPACITY: usize = 1024;

    extern "C" {
        fn apxinf_metal_w8_create(
            weights: *const i8,
            scales: *const f32,
            rows: u32,
            columns: u32,
            group_size: u32,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_topk4(
            handle: *mut c_void,
            hidden: *const f32,
            hidden_count: u32,
            output_tokens: *mut u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_destroy(handle: *mut c_void);
        #[cfg(feature = "experiments")]
        fn apxinf_metal_w8_matvec_create(
            weights: *const i8,
            scales: *const f32,
            rows: u32,
            columns: u32,
            group_size: u32,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        #[cfg(feature = "experiments")]
        fn apxinf_metal_w8_matvec_multiply(
            handle: *mut c_void,
            input: *const f32,
            input_count: u32,
            output: *mut f32,
            output_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        #[cfg(feature = "experiments")]
        fn apxinf_metal_w8_matvec_destroy(handle: *mut c_void);
        fn apxinf_metal_w8_mlp_block_create(
            gate_up_weights: *const i8,
            gate_up_scales: *const f32,
            down_weights: *const i8,
            down_scales: *const f32,
            hidden_size: u32,
            intermediate_size: u32,
            group_size: u32,
            output: *mut *mut c_void,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_mlp_block_forward(
            handle: *mut c_void,
            input: *const f32,
            input_count: u32,
            output: *mut f32,
            output_count: u32,
            error: *mut c_char,
            error_capacity: usize,
        ) -> c_int;
        fn apxinf_metal_w8_mlp_block_destroy(handle: *mut c_void);
    }

    pub(super) struct Handle(NonNull<c_void>);

    #[cfg(feature = "experiments")]
    pub(super) struct MatVecHandle(NonNull<c_void>);

    pub(super) struct MlpBlockHandle(NonNull<c_void>);

    impl Handle {
        pub(super) fn new(weights: &PackedW8Rows) -> Result<Self, MetalW8Error> {
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_create(
                    weights.values.as_ptr(),
                    weights.scales.as_ptr(),
                    weights.rows as u32,
                    weights.columns as u32,
                    W8_GROUP_SIZE as u32,
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("create Metal W8 head", &error));
            }
            let output = NonNull::new(output)
                .ok_or_else(|| MetalW8Error::new("create Metal W8 head returned a null handle"))?;
            Ok(Self(output))
        }

        pub(super) fn topk4(&mut self, hidden: &[f32]) -> Result<[u32; W8_TOP_K], MetalW8Error> {
            let mut tokens = [0u32; W8_TOP_K];
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_topk4(
                    self.0.as_ptr(),
                    hidden.as_ptr(),
                    hidden.len() as u32,
                    tokens.as_mut_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("run Metal W8 top-4 head", &error));
            }
            Ok(tokens)
        }
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            unsafe { apxinf_metal_w8_destroy(self.0.as_ptr()) };
        }
    }

    #[cfg(feature = "experiments")]
    impl MatVecHandle {
        pub(super) fn new(weights: &PackedW8Rows) -> Result<Self, MetalW8Error> {
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_matvec_create(
                    weights.values.as_ptr(),
                    weights.scales.as_ptr(),
                    weights.rows as u32,
                    weights.columns as u32,
                    W8_GROUP_SIZE as u32,
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("create Metal W8 matvec", &error));
            }
            NonNull::new(output)
                .map(Self)
                .ok_or_else(|| MetalW8Error::new("create Metal W8 matvec returned a null handle"))
        }

        pub(super) fn multiply(
            &mut self,
            input: &[f32],
            output_count: usize,
        ) -> Result<Vec<f32>, MetalW8Error> {
            let mut output = vec![0.0f32; output_count];
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_matvec_multiply(
                    self.0.as_ptr(),
                    input.as_ptr(),
                    input.len() as u32,
                    output.as_mut_ptr(),
                    output.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("run Metal W8 matvec", &error));
            }
            Ok(output)
        }
    }

    #[cfg(feature = "experiments")]
    impl Drop for MatVecHandle {
        fn drop(&mut self) {
            unsafe { apxinf_metal_w8_matvec_destroy(self.0.as_ptr()) };
        }
    }

    impl MlpBlockHandle {
        pub(super) fn new(weights: &PackedW8MlpBlock) -> Result<Self, MetalW8Error> {
            let mut output = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_mlp_block_create(
                    weights.gate_up.values.as_ptr(),
                    weights.gate_up.scales.as_ptr(),
                    weights.down.values.as_ptr(),
                    weights.down.scales.as_ptr(),
                    weights.down.rows as u32,
                    weights.down.columns as u32,
                    W8_GROUP_SIZE as u32,
                    &mut output,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("create Metal W8 MLP block", &error));
            }
            NonNull::new(output).map(Self).ok_or_else(|| {
                MetalW8Error::new("create Metal W8 MLP block returned a null handle")
            })
        }

        pub(super) fn forward(
            &mut self,
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), MetalW8Error> {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            let status = unsafe {
                apxinf_metal_w8_mlp_block_forward(
                    self.0.as_ptr(),
                    input.as_ptr(),
                    input.len() as u32,
                    output.as_mut_ptr(),
                    output.len() as u32,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(bridge_error("run Metal W8 MLP block", &error));
            }
            Ok(())
        }
    }

    impl Drop for MlpBlockHandle {
        fn drop(&mut self) {
            unsafe { apxinf_metal_w8_mlp_block_destroy(self.0.as_ptr()) };
        }
    }

    fn bridge_error(context: &str, buffer: &[c_char]) -> MetalW8Error {
        let detail = unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        if detail.is_empty() {
            MetalW8Error::new(context)
        } else {
            MetalW8Error::new(format!("{context}: {detail}"))
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::{MetalW8Error, PackedW8MlpBlock, PackedW8Rows, W8_TOP_K};

    pub(super) struct Handle;
    #[cfg(feature = "experiments")]
    pub(super) struct MatVecHandle;

    pub(super) struct MlpBlockHandle;

    impl Handle {
        pub(super) fn new(_weights: &PackedW8Rows) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 language-model head requires macOS",
            ))
        }

        pub(super) fn topk4(&mut self, _hidden: &[f32]) -> Result<[u32; W8_TOP_K], MetalW8Error> {
            Err(MetalW8Error::new(
                "Metal W8 language-model head requires macOS",
            ))
        }
    }

    #[cfg(feature = "experiments")]
    impl MatVecHandle {
        pub(super) fn new(_weights: &PackedW8Rows) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 matvec requires macOS"))
        }

        pub(super) fn multiply(
            &mut self,
            _input: &[f32],
            _output_count: usize,
        ) -> Result<Vec<f32>, MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 matvec requires macOS"))
        }
    }

    impl MlpBlockHandle {
        pub(super) fn new(_weights: &PackedW8MlpBlock) -> Result<Self, MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 MLP block requires macOS"))
        }

        pub(super) fn forward(
            &mut self,
            _input: &[f32],
            _output: &mut [f32],
        ) -> Result<(), MetalW8Error> {
            Err(MetalW8Error::new("Metal W8 MLP block requires macOS"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(rows: usize, columns: usize) -> (Vec<f32>, Vec<f32>) {
        let weights = (0..rows * columns)
            .map(|index| {
                let value = ((index * 37 + index / 11 + 17) % 251) as f32 - 125.0;
                value * 0.0017
            })
            .collect();
        let hidden = (0..columns)
            .map(|index| (((index * 29 + 5) % 101) as f32 - 50.0) * 0.006)
            .collect();
        (weights, hidden)
    }

    #[test]
    fn grouped_w8_scores_track_f32_and_preserve_fixture_argmax() {
        let rows = 37;
        let columns = 128;
        let (mut weights, hidden) = fixture(rows, columns);
        // Give one row a clear margin so this test measures quantization math,
        // not the instability of an intentionally near-tied argmax.
        for (column, value) in hidden.iter().enumerate() {
            weights[19 * columns + column] += value * 2.0;
        }
        let packed = PackedW8Rows::pack_f32(&weights, rows, columns).unwrap();
        let quantized = packed.scores(&hidden).unwrap();
        let reference: Vec<f32> = weights
            .chunks_exact(columns)
            .map(|row| {
                row.iter()
                    .zip(&hidden)
                    .map(|(left, right)| left * right)
                    .sum()
            })
            .collect();
        let max_error = reference
            .iter()
            .zip(&quantized)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 0.01, "maximum W8 score error was {max_error}");
        assert_eq!(argmax_scores(&reference).unwrap(), 19);
        assert_eq!(packed.argmax(&hidden).unwrap(), 19);
    }

    #[test]
    fn packing_rejects_shape_and_non_finite_values() {
        assert!(PackedW8Rows::pack_f32(&[0.0; 65], 1, 65).is_err());
        let mut source = vec![0.0; W8_GROUP_SIZE];
        source[7] = f32::NAN;
        assert!(PackedW8Rows::pack_f32(&source, 1, W8_GROUP_SIZE).is_err());
    }

    #[test]
    fn explicit_g32_rows_are_self_describing_and_score_with_their_own_groups() {
        let rows = 3;
        let columns = 64;
        let (weights, hidden) = fixture(rows, columns);
        let legacy = PackedW8Rows::pack_f32(&weights, rows, columns).unwrap();
        let g32 = PackedW8Rows::pack_f32_g32(&weights, rows, columns).unwrap();

        assert_eq!(legacy.group_size(), W8GroupSize::G64);
        assert_eq!(g32.group_size(), W8GroupSize::G32);
        assert_eq!(g32.scales().len(), legacy.scales().len() * 2);

        let groups_per_row = columns / 32;
        let expected = (0..rows)
            .map(|row| {
                (0..columns)
                    .map(|column| {
                        g32.values()[row * columns + column] as f32
                            * g32.scales()[row * groups_per_row + column / 32]
                            * hidden[column]
                    })
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(g32.scores(&hidden).unwrap(), expected);
        for error in [
            #[cfg(feature = "experiments")]
            MetalW8MatVec::from_packed(&g32)
                .err()
                .expect("legacy Metal matvec must reject g32"),
            MetalW8LmHead::from_packed(&g32)
                .err()
                .expect("legacy Metal head must reject g32"),
        ] {
            assert!(error.to_string().contains("group size 64"));
        }
    }

    #[test]
    fn packed_mlp_block_matches_the_composed_w8_cpu_oracle() {
        let hidden_size = 64;
        let intermediate_size = 64;
        let (gate, hidden) = fixture(intermediate_size, hidden_size);
        let up = gate
            .iter()
            .enumerate()
            .map(|(index, value)| value * 0.7 + (index % 7) as f32 * 0.0003)
            .collect::<Vec<_>>();
        let down = (0..hidden_size * intermediate_size)
            .map(|index| (((index * 41 + 13) % 197) as f32 - 98.0) * 0.0011)
            .collect::<Vec<_>>();

        let gate_rows = PackedW8Rows::pack_f32(&gate, intermediate_size, hidden_size).unwrap();
        let up_rows = PackedW8Rows::pack_f32(&up, intermediate_size, hidden_size).unwrap();
        let down_rows = PackedW8Rows::pack_f32(&down, hidden_size, intermediate_size).unwrap();
        let gate_scores = gate_rows.scores(&hidden).unwrap();
        let up_scores = up_rows.scores(&hidden).unwrap();
        let activated = gate_scores
            .iter()
            .zip(&up_scores)
            .map(|(&gate, &up)| gate / (1.0 + (-gate).exp()) * up)
            .collect::<Vec<_>>();
        let expected = down_rows.scores(&activated).unwrap();

        let packed =
            PackedW8MlpBlock::pack_f32(&gate, &up, &down, hidden_size, intermediate_size).unwrap();
        assert_eq!(packed.forward(&hidden).unwrap(), expected);
    }

    #[test]
    fn packed_mlp_block_buffer_ledger_is_exact_and_scoped_to_mtl_buffers() {
        let hidden_size = 64;
        let intermediate_size = 64;
        let elements = hidden_size * intermediate_size;
        let packed = PackedW8MlpBlock::pack_f32(
            &vec![0.01; elements],
            &vec![0.02; elements],
            &vec![0.03; elements],
            hidden_size,
            intermediate_size,
        )
        .unwrap();

        let ledger = packed.buffer_ledger().unwrap();

        assert_eq!(ledger.allocated_buffers, 8);
        assert_eq!(ledger.shared_buffers, 6);
        assert_eq!(ledger.private_buffers, 2);
        assert_eq!(ledger.packed_weight_bytes, 12_288);
        assert_eq!(ledger.packed_scale_bytes, 768);
        assert_eq!(ledger.activation_bytes, 1_280);
        assert_eq!(ledger.total_persistent_bytes, 14_336);
        assert_eq!(ledger.host_input_bytes_per_decode, 256);
        assert_eq!(ledger.host_output_bytes_per_decode, 256);
        assert_eq!(ledger.command_buffers_per_decode, 1);
        assert_eq!(ledger.compute_encoders_per_decode, 3);
        assert_eq!(ledger.commits_per_decode, 1);
        assert_eq!(ledger.waits_per_decode, 1);
    }

    #[test]
    fn packed_mlp_block_rejects_individually_misaligned_gate_and_up_rows() {
        let hidden_size = 64;
        let intermediate_size = 64;
        let elements = hidden_size * intermediate_size;
        let gate = vec![0.0; elements - 1];
        let up = vec![0.0; elements + 1];
        let down = vec![0.0; elements];
        let error = PackedW8MlpBlock::pack_f32(&gate, &up, &down, hidden_size, intermediate_size)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("gate has 4095 elements, expected 4096"));
    }

    #[test]
    fn packed_mlp_precision_screen_only_changes_the_down_projection() {
        let hidden_size = 64;
        let intermediate_size = 64;
        let elements = hidden_size * intermediate_size;
        let gate = vec![0.01; elements];
        let up = vec![0.02; elements];
        let down = vec![0.03; elements];
        let packed = PackedW8MlpBlock::pack_f32_with_down_group_size(
            &gate,
            &up,
            &down,
            hidden_size,
            intermediate_size,
            W8GroupSize::G32,
        )
        .unwrap();

        assert_eq!(packed.gate_up_group_size(), W8GroupSize::G64);
        assert_eq!(packed.down_group_size(), W8GroupSize::G32);
        assert_eq!(packed.gate_up_scale_bytes(), 2 * intermediate_size * 4);
        assert_eq!(packed.down_scale_bytes(), hidden_size * 2 * 4);
        let error = MetalW8MlpBlock::from_packed(&packed)
            .err()
            .expect("legacy Metal MLP must reject CPU-only g32 weights");
        assert!(error.to_string().contains("group size 64"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_mlp_block_matches_the_complete_quantized_cpu_oracle() {
        let hidden_size = 128;
        let intermediate_size = 128;
        let (gate, hidden) = fixture(intermediate_size, hidden_size);
        let up = gate
            .iter()
            .enumerate()
            .map(|(index, value)| value * -0.4 + (index % 11) as f32 * 0.0002)
            .collect::<Vec<_>>();
        let down = (0..hidden_size * intermediate_size)
            .map(|index| (((index * 43 + 23) % 211) as f32 - 105.0) * 0.0013)
            .collect::<Vec<_>>();
        let packed =
            PackedW8MlpBlock::pack_f32(&gate, &up, &down, hidden_size, intermediate_size).unwrap();
        let expected = packed.forward(&hidden).unwrap();

        let mut block = MetalW8MlpBlock::from_packed(&packed).unwrap();
        let actual = block.forward(&hidden).unwrap();
        assert_eq!(actual.len(), hidden_size);
        for (row, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2.0e-4,
                "row {row}: Metal={actual}, CPU W8={expected}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_mlp_block_rejects_invalid_inputs_without_fallback() {
        let hidden_size = 64;
        let intermediate_size = 64;
        let elements = hidden_size * intermediate_size;
        let packed = PackedW8MlpBlock::pack_f32(
            &vec![0.01; elements],
            &vec![0.02; elements],
            &vec![0.03; elements],
            hidden_size,
            intermediate_size,
        )
        .unwrap();
        let mut block = MetalW8MlpBlock::from_packed(&packed).unwrap();
        assert!(block.forward(&[0.0; 63]).is_err());
        let mut non_finite = vec![0.0; hidden_size];
        non_finite[17] = f32::NAN;
        assert!(block.forward(&non_finite).is_err());
    }

    #[test]
    fn cpu_topk_oracle_breaks_ties_by_lowest_token() {
        let scores = vec![3.0, 4.0, 4.0, -2.0, 4.0, 3.5, 4.0];
        assert_eq!(topk4_scores(&scores).unwrap(), [1, 2, 4, 6]);
    }

    #[test]
    fn qwen35_tied_lm_head_ledger_closes_to_the_official_shape() {
        let ledger = LmHeadBufferLedger::from_dimensions(248_320, 1_024).unwrap();

        assert_eq!(ledger.scope, "resident-mtlbuffer-only");
        assert_eq!(ledger.allocated_buffers, 5);
        assert_eq!(ledger.shared_buffers, 4);
        assert_eq!(ledger.private_buffers, 1);
        assert_eq!(ledger.packed_weight_bytes, 254_279_680);
        assert_eq!(ledger.packed_scale_bytes, 15_892_480);
        assert_eq!(ledger.hidden_bytes, 4_096);
        assert_eq!(ledger.partial_topk_bytes, 993_280);
        assert_eq!(ledger.output_token_bytes, 16);
        assert_eq!(ledger.total_persistent_bytes, 271_169_552);
        assert_eq!(ledger.host_input_bytes_per_call, 4_096);
        assert_eq!(ledger.host_output_bytes_per_call, 16);
        assert_eq!(ledger.command_buffers_per_call, 1);
        assert_eq!(ledger.compute_encoders_per_call, 2);
        assert_eq!(ledger.commits_per_call, 1);
        assert_eq!(ledger.waits_per_call, 1);
    }

    #[test]
    fn metal_shader_is_a_single_discoverable_source() {
        let shader = include_str!("metal_w8.metal");
        let bridge = include_str!("metal_w8_bridge.mm");
        let mlp_shader = include_str!("metal_w8_mlp.metal");
        let mlp_bridge = include_str!("metal_w8_mlp_bridge.mm");
        #[cfg(feature = "experiments")]
        {
            let matvec_shader = include_str!("metal_w8_matvec.metal");
            assert!(matvec_shader.contains("kernel void w8_rows_matvec("));
            assert!(!matvec_shader.contains("kernel void w8_mlp_"));
            assert!(bridge.contains("#include \"metal_w8_matvec_source.inc\""));
        }
        assert!(!shader.contains("kernel void w8_rows_matvec("));
        assert!(shader.contains("kernel void w8_rows_topk4("));
        assert!(shader.contains("kernel void w8_final_topk4("));
        assert!(mlp_shader.contains("kernel void w8_mlp_gate_up("));
        assert!(mlp_shader.contains("kernel void w8_mlp_silu_mul("));
        assert!(mlp_shader.contains("kernel void w8_mlp_down("));
        assert!(!shader.contains("kernel void w8_mlp_"));
        assert!(bridge.contains("#include \"metal_w8_source.inc\""));
        assert!(mlp_bridge.contains("#include \"metal_w8_mlp_source.inc\""));
        assert!(!bridge.contains("kernel void w8_rows_matvec("));
        assert!(!bridge.contains("kernel void w8_rows_topk4("));
        assert!(!bridge.contains("kernel void w8_final_topk4("));
        assert!(!mlp_bridge.contains("kernel void w8_mlp_"));
    }

    #[cfg(all(target_os = "macos", feature = "experiments"))]
    #[test]
    fn metal_matvec_matches_every_quantized_cpu_score() {
        let rows = 257;
        let columns = 128;
        let (weights, hidden) = fixture(rows, columns);
        let packed = PackedW8Rows::pack_f32(&weights, rows, columns).unwrap();
        let expected = packed.scores(&hidden).unwrap();
        let mut matvec = MetalW8MatVec::from_packed(&packed).unwrap();
        let actual = matvec.multiply(&hidden).unwrap();
        assert_eq!(actual.len(), rows);
        for (row, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "row {row}: Metal={actual}, CPU={expected}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_topk4_matches_quantized_cpu_oracle() {
        let rows = 257;
        let columns = 128;
        let (mut weights, hidden) = fixture(rows, columns);
        for (column, value) in hidden.iter().enumerate() {
            weights[173 * columns + column] += value * 3.0;
        }
        let packed = PackedW8Rows::pack_f32(&weights, rows, columns).unwrap();
        let expected = packed.topk4(&hidden).unwrap();
        let mut head = MetalW8LmHead::from_packed(&packed).unwrap();
        assert_eq!(head.topk4(&hidden).unwrap(), expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_lm_head_exposes_the_ledger_of_its_resident_handle() {
        let rows = 17;
        let columns = 128;
        let (weights, _) = fixture(rows, columns);
        let packed = PackedW8Rows::pack_f32(&weights, rows, columns).unwrap();
        let head = MetalW8LmHead::from_packed(&packed).unwrap();

        assert_eq!(
            head.buffer_ledger(),
            LmHeadBufferLedger::from_dimensions(rows, columns).unwrap()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_topk4_preserves_four_winners_from_one_eight_row_group() {
        let rows = 17;
        let columns = 128;
        let hidden = vec![1.0f32; columns];
        let mut weights = vec![0.001f32; rows * columns];
        for (row, value) in [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)] {
            weights[row * columns..(row + 1) * columns].fill(value);
        }
        let packed = PackedW8Rows::pack_f32(&weights, rows, columns).unwrap();
        assert_eq!(packed.topk4(&hidden).unwrap(), [0, 1, 2, 3]);
        let mut head = MetalW8LmHead::from_packed(&packed).unwrap();
        assert_eq!(head.topk4(&hidden).unwrap(), [0, 1, 2, 3]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_topk4_breaks_exact_ties_by_lowest_token() {
        let rows = 19;
        let columns = 128;
        let weights = vec![0.0f32; rows * columns];
        let hidden = vec![0.25f32; columns];
        let packed = PackedW8Rows::pack_f32(&weights, rows, columns).unwrap();
        let mut head = MetalW8LmHead::from_packed(&packed).unwrap();
        assert_eq!(head.topk4(&hidden).unwrap(), [0, 1, 2, 3]);
    }
}
