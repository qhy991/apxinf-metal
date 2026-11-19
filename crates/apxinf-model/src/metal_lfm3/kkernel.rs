//! Decode-step custom kernels armed by `APXINF_LFM3_KKERNEL=1`
//! (`doc/lfm3-lm-head-custom-kernel.md` arms/switch/banner).
//!
//! The kernels are constructed once at model load from the same env switch the
//! banner reports, so the unset (baseline) arm compiles no custom pipeline and
//! the armed arm cannot silently run the default (the empty-arm trap). Both
//! kernels take the exact MLX op's input shapes (no padding, no bucketing) —
//! the shape gate is enforced in `apxinf-mlx`.

use apxinf_core::Result;
use apxinf_mlx::{fusions::LmHeadGemv, fusions::SiluMulFused, Array};

/// The empty-arm gate: the worker banner and the kernels must agree with this.
pub fn enabled() -> bool {
    std::env::var_os("APXINF_LFM3_KKERNEL").is_some_and(|v| v == "1")
}

pub struct KKernel {
    lm_head: Option<LmHeadGemv>,
    silu_mul: Option<SiluMulFused>,
}

impl KKernel {
    pub fn new(stream: &apxinf_mlx::Stream) -> Result<Self> {
        let active = enabled();
        Ok(Self {
            lm_head: active.then(|| LmHeadGemv::new(stream)).transpose()?,
            silu_mul: active.then(|| SiluMulFused::new(stream)).transpose()?,
        })
    }

    /// The lm_head single-row GEMV: `[1,1,K] × [K,V] → [1,1,V]` FP32 logits.
    pub fn project(&self, x: &Array, head: &Array) -> Result<Array> {
        match &self.lm_head {
            Some(k) => k.call(x, head),
            None => x.matmul(head),
        }
    }

    /// `silu(gate) * up` — fused when armed, graph when not.
    pub fn silu_mul(&self, gate: &Array, up: &Array) -> Result<Array> {
        match &self.silu_mul {
            Some(k) => k.call(gate, up),
            None => gate.mul(&gate.sigmoid()?)?.mul(up),
        }
    }

    /// `true` when the custom lm_head pipeline is armed (gate-0 census; used
    /// by K3's empty-arm trap).
    #[cfg(test)]
    pub fn lm_head_armed(&self) -> bool {
        self.lm_head.is_some()
    }

    /// `true` when the fused silu_mul pipeline is armed (gate-0 census; used
    /// by K3's empty-arm trap).
    #[cfg(test)]
    pub fn silu_mul_armed(&self) -> bool {
        self.silu_mul.is_some()
    }
}
