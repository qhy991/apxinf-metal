//! Causal depthwise convolution with zero history, one thread per output.
//! Portions adapted from EngineTailor (MIT), following MLX 0.32.3
//! `depthwise_conv_1d` (MIT, Apple). See NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// `out[t, c] = sum_i x[t + i - (S - 1), c] * w[c, i]` for BF16 `x [1, L, C]`
/// and weight `[C, S, 1]`, with rows before 0 read as zero, accumulated in
/// FP32 in tap order and rounded once to BF16 `[1, L, C]`. This equals a
/// depthwise convolution over `S - 1` zero rows followed by `x`.
pub struct CausalDepthwiseConv {
    kernel: MetalKernel,
}

impl CausalDepthwiseConv {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                "apxinf_causal_depthwise_conv",
                &["x", "weight", "length", "kernel_size", "history_rows", "history"],
                &["out"],
                include_str!("../native/causal_depthwise_conv.metal"),
                "",
            )?,
        })
    }

    pub fn call(&self, x: &Array, weight: &Array) -> Result<Array> {
        self.call_with_history(x, weight, None)
    }

    /// The same convolution with `taps - 1` BF16 history rows at the input's
    /// left edge, so a chunked prefix cache sees the same operator as the
    /// unfused `concat([state, chunk], axis=1).conv1d(...)`.
    pub fn call_with_history(
        &self,
        x: &Array,
        weight: &Array,
        history: Option<&Array>,
    ) -> Result<Array> {
        let [1, length, channels] = *x.shape() else {
            return Err(Error::Contract(
                "causal convolution input must be [1, L, C]",
            ));
        };
        let [wc, taps, 1] = *weight.shape() else {
            return Err(Error::Contract(
                "causal convolution weight must be [C, S, 1]",
            ));
        };
        if wc != channels
            || [x, weight].iter().any(|a| a.dtype() != MlxDType::BF16)
            || length == 0
            || channels == 0
            || taps == 0
            || length
                .checked_mul(channels)
                .is_none_or(|extent| extent > i32::MAX as usize)
            || channels
                .checked_mul(taps)
                .is_none_or(|extent| extent > i32::MAX as usize)
            || history.is_some_and(|h| {
                h.shape() != [1, taps - 1, channels] || h.dtype() != MlxDType::BF16
            })
        {
            return Err(Error::Contract(
                "causal convolution requires BF16 [1, L, C] and [C, S, 1] with extents within i32",
            ));
        }
        let scalar = |v: usize| Array::from_i32(x.stream(), &[], &[v as i32]);
        // The history tensor is `[1, taps - 1, C]`; `null` in the no-history
        // case makes the kernel's history branch unreachable, so the
        // accumulation and rounding are bit-identical to the zero-history path.
        let history_rows = history.map_or(0i32, |h| (h.shape()[1]) as i32);
        let history_buffer: Option<Array> = history.cloned();
        // The largest power of two not above L, capped at 32, rows
        // of time per threadgroup of 1024 threads.
        let time = 1usize << length.ilog2().min(5);
        // SAFETY: the grid is exactly `[C, L]`, so `c < C`; threads with
        // `t >= L` return. Reads are `x[row * C + c]` for `0 <= row <= t < L`
        // and `w[c * S + i]` for `i < S`; the store is `out[t * C + c]`, one
        // per thread. `length` and `kernel_size` are scalars equal to L and S.
        let inputs = [
            x.clone(),
            weight.clone(),
            scalar(length)?,
            scalar(taps)?,
            scalar(history_rows as usize)?,
            history_buffer.unwrap_or_else(|| {
                Array::zeros(x.stream(), &[0], MlxDType::BF16).unwrap()
            }),
        ];
        let out = unsafe {
            self.kernel.call(
                &inputs,
                &[(x.shape(), MlxDType::BF16)],
                [channels, length, 1],
                [1024 / time, time, 1],
                &[
                    ("InT", TemplateArg::DType(MlxDType::BF16)),
                    ("C", TemplateArg::Int(channels as i32)),
                ],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}
