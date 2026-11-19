//! Fused `silu(gate) * up` for the decode step: the graph's
//! `g.mul(g.sigmoid()).mul(u)` as one element-local FP32-kernel launch.
//! Elementwise fusion is value-identical by construction (pre-registration
//! §1 / §11); the kernel performs the identical chain in the identical order.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

pub struct SiluMulFused {
    kernel: MetalKernel,
}

impl SiluMulFused {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_silu_mul_fused",
                &["gate", "up"],
                &["out"],
                include_str!("../native/silu_mul_fused.metal"),
                "",
            )?,
        })
    }

    /// `gate` and `up` must have the same BF16 shape; output preserves it.
    /// Other shapes/dtypes fail loudly (pre-registration §5.2 / test K4).
    pub fn call(&self, gate: &Array, up: &Array) -> Result<Array> {
        if gate.shape() != up.shape() {
            return Err(Error::Contract("fused silu_mul requires equal shapes"));
        }
        if gate.dtype() != MlxDType::BF16 || up.dtype() != MlxDType::BF16 {
            return Err(Error::Contract("fused silu_mul requires BF16 inputs"));
        }
        let elements = i32::try_from(gate.numel())
            .ok()
            .filter(|&n| n > 0)
            .ok_or(Error::Contract("fused silu_mul element count must fit i32"))?
            as usize;
        // SAFETY: inputs are row-contiguous. One guarded thread per element
        // reads exactly one gate/up element and writes one output element — no
        // overlap, all offsets bounded by the numel and shape checks above.
        let out = unsafe {
            self.kernel.call(
                &[gate.clone(), up.clone()],
                &[(gate.shape(), MlxDType::BF16)],
                [elements, 1, 1],
                [256, 1, 1],
                &[
                    ("T", TemplateArg::DType(MlxDType::BF16)),
                    (
                        "ELEMENTS",
                        TemplateArg::Int(
                            i32::try_from(elements)
                                .map_err(|_| Error::Contract("silu_mul elements exceed i32"))?,
                        ),
                    ),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}
