//! The ANE half of the LFM3 MLP column split.
//!
//! `APXINF_LFM3_ANE=<share>` splits the intermediate dimension: columns
//! `[0, share)` stay on the GPU and `[share, intermediate)` are meant for the
//! Neural Engine. This module owns the second half so the split point has one
//! implementation instead of a call site buried in `model.rs`.
//!
//! **The Core ML program is not wired in yet.** `predict` currently evaluates the
//! same two projections on the GPU, which is what `model.rs` did inline before
//! this module existed, so behaviour is unchanged. The single insertion point is
//! marked below: build the FP16 program from the #151 recipe
//! (`crates/apxinf-coreml/examples/lfm3_mlp_parity_probe.rs`) once per layer,
//! cache it, and replace `gpu_fallback` with its prediction. The contract the
//! caller relies on is exactly: `x: [T, hidden]` in, `silu(x·W1ᵀ) * (x·W3ᵀ)` for
//! the `[share, intermediate)` columns out, shape `[T, intermediate - share]`,
//! same dtype as `x`.
//!
//! Not byte-identical to the unsplit path by design: the projection's column
//! count changes, so GEMM tiling and reduction order can differ (#128). The token
//! gate in `doc/lfm3-ane-fp16-port-preregister.md` decides that, not a byte
//! comparison.

use super::weights::Linear;
use apxinf_core::Result;
use apxinf_mlx::Array;

/// One layer's ANE column share.
pub(crate) struct AneSlice {
    /// First intermediate column owned by this slice (`share`).
    start: usize,
    /// One past the last column (`intermediate`).
    end: usize,
}

impl AneSlice {
    pub(crate) fn new(start: usize, end: usize) -> Result<Self> {
        if start >= end {
            return Err(apxinf_core::Error::Other(format!(
                "ANE slice needs start < end, got {start}..{end}"
            )));
        }
        Ok(Self { start, end })
    }

    pub(crate) fn width(&self) -> usize {
        self.end - self.start
    }

    /// `silu(gate) * up` restricted to `[start, end)`, from projections the
    /// caller already computed for the full intermediate width.
    pub(crate) fn from_parts(&self, gate: &Array, up: &Array) -> Result<Array> {
        let gate = gate.slice_axis(1, self.start, self.end)?;
        let up = up.slice_axis(1, self.start, self.end)?;
        let g = gate.mul(&gate.sigmoid()?)?;
        g.mul(&up)
    }

    /// `silu(x·W1ᵀ) * (x·W3ᵀ)` over columns `[start, end)`.
    ///
    /// TODO(#160): build the FP16 Core ML program here and call it. Until then
    /// both projections run on the GPU, which keeps the split observable without
    /// changing the numerics' *source*, only its tiling.
    pub(crate) fn predict(&self, x: &Array, w1: &Linear, w3: &Linear) -> Result<Array> {
        let gate = w1.call(x)?;
        let up = w3.call(x)?;
        let gate = gate.slice_axis(1, self.start, self.end)?;
        let up = up.slice_axis(1, self.start, self.end)?;
        let g = gate.mul(&gate.sigmoid()?)?;
        g.mul(&up)
    }
}
