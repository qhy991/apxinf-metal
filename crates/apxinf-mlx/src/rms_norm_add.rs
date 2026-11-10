//! Residual add fused with RMS normalization: `norm(x + y) * weight` in one
//! launch, plus the BF16 sum so the caller keeps its residual path.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

pub struct RmsNormAdd {
    kernel: MetalKernel,
    hidden: usize,
}

impl RmsNormAdd {
    /// `hidden` is the row width (Qwen3-0.6B: 1024).
    pub fn new(stream: &Stream, hidden: usize) -> Result<Self> {
        if !matches!(hidden, 1024) {
            return Err(Error::Contract(
                "rms_norm_add is compiled for one row width".into(),
            ));
        }
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_rms_norm_add",
                &["x_input", "y_input", "weight_input", "eps_input"],
                &["hidden_output", "normed_output"],
                include_str!("../native/rms_norm_add.metal"),
                "",
            )?,
            hidden,
        })
    }

    /// `x` and `y` are `[1, rows, hidden]` BF16, `weight` is `[hidden]` F32.
    /// Returns `(hidden, normed)` with the same shape as `x`.
    pub fn call(&self, x: &Array, y: &Array, weight: &Array, eps: f32) -> Result<(Array, Array)> {
        let shape = x.shape().to_vec();
        if shape.len() != 3
            || shape[0] != 1
            || shape[2] != self.hidden
            || y.shape() != shape.as_slice()
            || x.dtype() != MlxDType::BF16
            || y.dtype() != MlxDType::BF16
            || weight.shape() != [self.hidden]
            || weight.dtype() != MlxDType::F32
        {
            return Err(Error::Contract(
                "rms_norm_add requires BF16 [1, rows, hidden] operands and an F32 weight".into(),
            ));
        }
        let rows = shape[1];
        let eps_tensor = Array::from_f32(&x.stream(), &[1], &[eps])?;
        let flat = [1usize, rows, self.hidden];
        let outputs = unsafe {
            self.kernel.call(
                &[x.clone(), y.clone(), weight.clone(), eps_tensor],
                &[
                    (&flat[..], MlxDType::BF16),
                    (&flat[..], MlxDType::BF16),
                ],
                // `grid` is in threads, not threadgroups (one threadgroup per row).
                [rows * 256, 1, 1],
                [256, 1, 1],
                &[
                    ("T", TemplateArg::DType(MlxDType::BF16)),
                    ("H", TemplateArg::Int(self.hidden as i32)),
                ],
            )?
        };
        let mut outputs = outputs.into_iter();
        Ok((outputs.next().unwrap(), outputs.next().unwrap()))
    }
}
