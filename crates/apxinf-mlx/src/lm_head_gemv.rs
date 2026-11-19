//! Shape-preserving lm_head GEMV for the single-row decode step.
//!
//! One FP32 accumulator per output column over ascending K — the pre-registered
//! accumulation form (`doc/lfm3-lm-head-custom-kernel.md` §1 / §11). A warp
//! maps 32 lanes to 32 consecutive columns so each weight row loads
//! coalesced. The exact input and weight shapes are enforced; no padding, no
//! K-split, no extra output rows.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

pub struct LmHeadGemv {
    kernel: MetalKernel,
}

impl LmHeadGemv {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_lm_head_gemv_bf16",
                &["x", "weight"],
                &["out"],
                include_str!("../native/lm_head_gemv.metal"),
                "",
            )?,
        })
    }

    /// `x` `[1, 1, K]` BF16, `head` `[K, V]` BF16 → `[1, 1, V]` FP32 logits.
    /// Any other geometry fails loudly (pre-registration §5.2 / test K4).
    pub fn call(&self, x: &Array, head: &Array) -> Result<Array> {
        let [1, 1, k] = *x.shape() else {
            return Err(Error::Contract(
                "lm_head GEMV requires BF16 input [1, 1, K]",
            ));
        };
        if x.dtype() != MlxDType::BF16 || head.dtype() != MlxDType::BF16 {
            return Err(Error::Contract("lm_head GEMV requires BF16 input/weight"));
        }
        let [wk, vocab] = *head.shape() else {
            return Err(Error::Contract("lm_head GEMV requires weight [K, V]"));
        };
        if wk != k || vocab == 0 || vocab > 1_000_000 {
            return Err(Error::Contract(
                "lm_head GEMV weight K must match input and V must be positive",
            ));
        }
        // SAFETY: inputs are row-contiguous. 32 threads own 32 consecutive
        // columns in one grid block; threads with col >= V return. Every
        // thread reads weight[k * V + col] for k < K (checked above) and
        // writes out[col] once, so no reads or writes overlap. The output has
        // exactly V FP32 entries and no padded columns are allocated.
        const THREADS: usize = 256;
        let blocks = vocab.div_ceil(THREADS);
        let out = unsafe {
            self.kernel.call(
                &[x.clone(), head.clone()],
                &[(&[1, 1, vocab], MlxDType::F32)],
                [blocks, 1, 1],
                [THREADS, 1, 1],
                &[
                    ("K", TemplateArg::Int(i32::try_from(k).map_err(|_| {
                        Error::Contract("lm_head K exceeds i32")
                    })?)),
                    ("V", TemplateArg::Int(i32::try_from(vocab).map_err(|_| {
                        Error::Contract("lm_head V exceeds i32")
                    })?)),
                    ("THREADS", TemplateArg::Int(THREADS as i32)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}
