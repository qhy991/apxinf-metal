//! A GPU stand-in for the Neural Engine MLP share, for telling apart the two
//! things the real split changes at once (EngineTailor #236, hypothesis H2).
//!
//! The split fails the quality gate's artifacts dimension on every case that
//! renders dense text. Two candidate causes are confounded in that result:
//! splitting the MLP sum in two and carrying the offloaded half through FP16,
//! against whatever Core ML and the Neural Engine do inside the program
//! (INT8 weights, their own accumulation order and optimizations).
//!
//! This module runs the offloaded half on the GPU instead, over the same
//! rows, from the same weights, through the same scale and SiLU chain, with
//! the same tiling and the same recombination. Core ML is not involved. If
//! the artifact survives here, the cause is the split and its rounding, and
//! raising the precision at the split point should fix it. If it does not,
//! the cause is inside the part this does not reproduce.
//!
//! `APXINF_QWEN_IMAGE_ANE_SIM=f16` is the arm under test. `=bf16` is its
//! control: the same code with no FP16 narrowing, which should land on the
//! GPU-only baseline and so shows that the machinery itself is not what
//! moved the picture.
//!
//! Weights stay in their Q4 form and are dequantized per call, so the arm
//! costs GPU time rather than memory. It is measurement scaffolding, off by
//! default, and never a production path.
use super::ane::{GPU_ROWS, TILE};
use super::weights::{GateUp, Transformer, DIM, MLP, Q4};
use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType};
use half::f16;

/// What the offloaded half is carried in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Precision {
    /// The arm under test: the same narrowing the Core ML program applies.
    F16,
    /// The control: no narrowing, so only the split itself differs from
    /// GPU-only.
    Bf16,
}

impl Precision {
    pub fn dtype(self) -> MlxDType {
        match self {
            Self::F16 => MlxDType::F16,
            Self::Bf16 => MlxDType::BF16,
        }
    }
}

fn parse(value: &str) -> Option<Precision> {
    match value {
        "1" | "f16" | "fp16" => Some(Precision::F16),
        "bf16" => Some(Precision::Bf16),
        _ => None,
    }
}

/// The requested arm, or `None` when the variable is unset.
pub fn requested() -> Option<Precision> {
    parse(&std::env::var("APXINF_QWEN_IMAGE_ANE_SIM").ok()?)
}

/// A set value this module cannot honour, named so a typo cannot quietly
/// deliver the Core ML arm instead of the simulated one.
pub fn refusal() -> Option<String> {
    let raw = std::env::var("APXINF_QWEN_IMAGE_ANE_SIM").ok()?;
    parse(&raw).is_none().then(|| {
        format!("APXINF_QWEN_IMAGE_ANE_SIM={raw:?} is not an arm (expected f16 or bf16)")
    })
}

/// One block's offloaded weights: gate/up rows `[GPU_ROWS(), MLP)` and the
/// matching `out` columns, exactly the complement of what `ane::gpu_share`
/// leaves on the GPU.
struct Upper {
    gate: Q4,
    up: Q4,
    out: Q4,
}

pub struct Sim {
    precision: Precision,
    blocks: Vec<Upper>,
}

impl Sim {
    /// Clones the offloaded slices out of every block. The clones are
    /// reference counts, so they keep the originals alive once
    /// `ane::gpu_share` has narrowed the blocks to the GPU's rows.
    pub fn share(transformer: &Transformer, precision: Precision) -> Result<Self> {
        let rows = |q: &Q4| -> Result<Q4> {
            Ok(Q4 {
                weight: q.weight.slice_axis(0, GPU_ROWS(), MLP)?.contiguous()?,
                scales: q.scales.slice_axis(0, GPU_ROWS(), MLP)?.contiguous()?,
                biases: q.biases.slice_axis(0, GPU_ROWS(), MLP)?.contiguous()?,
            })
        };
        let mut blocks = Vec::with_capacity(transformer.blocks.len());
        for block in &transformer.blocks {
            let GateUp::Separate { gate, up } = &block.gate_up else {
                return Err(Error::Contract(
                    "Qwen-Image ANE simulation is prepared before the MLP weights are packed",
                ));
            };
            let o = &block.out;
            blocks.push(Upper {
                gate: rows(gate)?,
                up: rows(up)?,
                out: Q4 {
                    weight: o.weight.slice_axis(1, GPU_ROWS() / 8, MLP / 8)?.contiguous()?,
                    scales: o.scales.slice_axis(1, GPU_ROWS() / 64, MLP / 64)?.contiguous()?,
                    biases: o.biases.slice_axis(1, GPU_ROWS() / 64, MLP / 64)?.contiguous()?,
                },
            });
        }
        Ok(Self { precision, blocks })
    }

    pub fn dtype(&self) -> MlxDType {
        self.precision.dtype()
    }

    /// The offloaded half of block `index`, from the same `[tiles, DIM, TILE]`
    /// feed the Core ML arm receives, returning the same shape it returns.
    ///
    /// The chain mirrors `ane::program` op for op: gate and up over the
    /// pre-scaled input with `r` undoing the scale, SiLU as
    /// `0.5 v (1 + tanh(v / 2))` rather than a sigmoid so the two arms round
    /// the same way, the product scaled by `s`, then `out`. The Core ML arm
    /// runs one tile per prediction; tiles share `r` and `s` and no operation
    /// here mixes columns, so folding them into one matmul changes no value.
    pub fn call(&self, index: usize, feed: &Array, r: f16, s: f16) -> Result<Array> {
        let upper = &self.blocks[index];
        let stream = feed.stream();
        let dtype = self.dtype();
        let tiles = feed.shape()[0];
        let dense = |q: &Q4| -> Result<Array> {
            q.weight
                .dequantize(&q.scales, &q.biases, 64, 4)?
                .cast(dtype)
        };
        let scalar = |v: f32| Array::scalar(stream, v, dtype);
        let x = feed
            .transpose(&[1, 0, 2])?
            .contiguous()?
            .reshape(&[DIM, tiles * TILE])?;

        let r = scalar(r.to_f32())?;
        let gate = dense(&upper.gate)?.matmul(&x)?.mul(&r)?;
        let up = dense(&upper.up)?.matmul(&x)?.mul(&r)?;
        let half = gate.mul(&scalar(0.5)?)?;
        let silu = half.mul(&half.tanh()?.add(&scalar(1.0)?)?)?;
        let m = silu.mul(&up)?.mul(&scalar(s.to_f32())?)?;
        let y = dense(&upper.out)?.matmul(&m)?;

        debug_assert_eq!(y.shape(), &[DIM, tiles * TILE]);
        y.reshape(&[DIM, tiles, TILE])?
            .transpose(&[1, 0, 2])?
            .contiguous()
    }
}

#[cfg(test)]
mod tests {
    use super::super::ane::ANE_ROWS;
    use super::*;

    #[test]
    fn an_arm_name_is_either_known_or_refused_by_name() {
        assert_eq!(parse("f16"), Some(Precision::F16));
        assert_eq!(parse("fp16"), Some(Precision::F16));
        assert_eq!(parse("1"), Some(Precision::F16));
        assert_eq!(parse("bf16"), Some(Precision::Bf16));
        for typo in ["", "0", "F16", "fp32", "true", "yes"] {
            assert_eq!(parse(typo), None, "{typo:?} must not select an arm");
        }
    }

    #[test]
    fn the_two_arms_differ_only_in_the_width_they_carry() {
        assert_eq!(Precision::F16.dtype(), MlxDType::F16);
        assert_eq!(Precision::Bf16.dtype(), MlxDType::BF16);
    }

    /// The simulated half and the GPU's half have to tile the MLP exactly:
    /// every row once, no row twice. A slicing slip here would show up as a
    /// quality result rather than as a failure, which is the one way this
    /// experiment could mislead.
    #[test]
    fn the_simulated_rows_are_the_complement_of_the_gpu_rows() {
        let (gpu, ane) = (GPU_ROWS(), ANE_ROWS());
        assert_eq!(gpu + ane, MLP, "the two shares must cover the MLP exactly");
        assert_eq!(MLP - gpu, ane, "the simulation takes rows [GPU_ROWS, MLP)");
        assert!(gpu.is_multiple_of(8), "Q4 weights slice on packed columns");
        assert!(gpu.is_multiple_of(64), "scales and biases slice on groups");
    }

    #[test]
    fn a_tile_is_a_whole_number_of_feed_columns() {
        assert!(TILE > 0 && DIM > 0);
    }
}
