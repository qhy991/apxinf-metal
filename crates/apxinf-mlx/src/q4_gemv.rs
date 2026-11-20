//! Fused affine-Q4 (group 64) GEMV for M=1 decode (issue #142 follow-up):
//! `x` BF16 `[1, K]`, packed weight U32 `[N, K/8]`, BF16 `scales`/`biases`
//! `[N, K/64]` -> BF16 `[1, N]`. Weights are dequantised in registers along
//! the K groups and never expanded to `[N, K]`, so per-token weight traffic
//! stays at the packed byte count. Row width `K` is substituted into the Metal
//! source (compile-time unroll), with one compiled kernel cached per `K`
//! (LFM3 needs exactly two: hidden 2048 and intermediate 7680).
//!
//! Numerics: value = `q * scale + bias` in fp32 accumulated in fp32, rounding
//! once to BF16 at the store — the MLX affine dequant expression. The K
//! reduction order (lane-strided groups, `simd_sum` ladder) differs from MLX's
//! `quantized_matmul`, so outputs are a close approximation, not bit-identical
//! (same-precision cross-engine difference, recorded by the caller's gate).
use crate::{Array, MetalKernel, MlxDType, Stream};
use apxinf_core::{Error, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Group size fixed by the MLX affine export contract (QUEUE #8 format probe).
pub const GROUP: usize = 64;

pub struct Q4GemvBf16 {
    kernels: RefCell<HashMap<usize, Rc<MetalKernel>>>,
    stream: Stream,
}

impl Q4GemvBf16 {
    pub fn new(stream: &Stream) -> Result<Self> {
        if stream.metal_index().is_none() {
            return Err(Error::Contract("Q4GemvBf16 requires a Metal stream"));
        }
        Ok(Self {
            kernels: RefCell::new(HashMap::new()),
            stream: stream.clone(),
        })
    }

    fn kernel_for(&self, k: usize) -> Result<Rc<MetalKernel>> {
        if k == 0 || k % GROUP != 0 {
            return Err(Error::Contract(
                "Q4 GEMV row width must be a positive multiple of the group size",
            ));
        }
        if let Some(kernel) = self.kernels.borrow().get(&k) {
            return Ok(kernel.clone());
        }
        let source =
            include_str!("../native/q4_gemv_bf16.metal").replace("K_PLACEHOLDER", &k.to_string());
        let kernel = Rc::new(MetalKernel::new(
            &self.stream,
            "apxinf_q4g64_gemv_bf16",
            &["x", "weight", "scales", "biases", "n"],
            &["out"],
            &source,
            "",
        )?);
        self.kernels.borrow_mut().insert(k, kernel.clone());
        Ok(kernel)
    }

    /// Whether this call can run on the fused kernel: M=1 (any leading rank,
    /// flattened to one row), exact packed/scales geometry, expected dtypes.
    pub fn eligible(x: &Array, weight: &Array, scales: &Array, biases: &Array) -> bool {
        let Some(&k) = x.shape().last() else {
            return false;
        };
        let [rows, words] = match weight.shape() {
            [a, b] => [*a, *b],
            _ => return false,
        };
        x.dtype() == MlxDType::BF16
            && weight.dtype() == MlxDType::U32
            && scales.dtype() == MlxDType::BF16
            && biases.dtype() == MlxDType::BF16
            && k > 0
            && k % GROUP == 0
            && words * 8 == k
            && scales.shape() == [rows, k / GROUP]
            && biases.shape() == [rows, k / GROUP]
            && rows > 0
            && x.numel() == k
    }

    /// Run the fused GEMV. `x` is any rank with last dim `K` and `numel == K`
    /// (e.g. `[1, 1, K]` at decode); returns BF16 with `x`'s leading shape and
    /// last dim `N` — a drop-in for `quantized_matmul` on M=1 inputs.
    pub fn call(&self, x: &Array, weight: &Array, scales: &Array, biases: &Array) -> Result<Array> {
        if !Self::eligible(x, weight, scales, biases) {
            return Err(Error::Contract("Q4GemvBf16 eligibility failed"));
        }
        let k = *x.shape().last().expect("checked");
        let rows = weight.shape()[0];
        let kernel = self.kernel_for(k)?;
        let x2 = x.reshape(&[1, k])?;
        let n_arr = Array::from_i32(&self.stream, &[], &[rows as i32])?;
        // Same leading shape as the input (e.g. [1, 1, N] for the decode step)
        // so the fused kernel is a drop-in for `quantized_matmul`.
        let mut out_shape = x.shape().to_vec();
        let last = out_shape.len() - 1;
        out_shape[last] = rows;
        // SAFETY: one 256-thread threadgroup (8 simdgroups) covers rows
        // `8 oct + simd_group < N` (guarded against `n`); each simdgroup
        // strides groups `lane + 32 t < K/64`, reading words `8 g .. 8 g + 8`
        // of the packed row as two 16-byte vectors and x `64 g .. 64 g + 64`
        // as eight 16-byte vectors. All inputs are row-contiguous
        // (MetalKernel::new ensures it) and x is 16B aligned with group
        // offsets at 128B boundaries, so every access stays in bounds; simd
        // lane 0 alone writes each `out[row]`.
        let out = unsafe {
            kernel.call(
                &[x2, weight.clone(), scales.clone(), biases.clone(), n_arr],
                &[(&out_shape, MlxDType::BF16)],
                [256, rows.div_ceil(8), 1],
                [256, 1, 1],
                &[],
            )?
        };
        Ok(out.into_iter().next().expect("one output"))
    }
}

/// Shared install (one per stream), like `qmm_bm64::SharedQmmBm64`.
pub type SharedQ4GemvBf16 = Rc<Q4GemvBf16>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eligibility_flattens_leading_dims() {
        let s = Stream::cpu().unwrap();
        let bf = |shape: &[usize]| Array::zeros(&s, shape, MlxDType::BF16).unwrap();
        let w = Array::zeros(&s, &[16, 8], MlxDType::U32).unwrap();
        let sc = bf(&[16, 1]);
        let bi = bf(&[16, 1]);
        // rank-3 decode shape [1, 1, 64] is eligible; [1, 2, 64] (M=2) is not.
        assert!(Q4GemvBf16::eligible(&bf(&[1, 1, 64]), &w, &sc, &bi));
        assert!(!Q4GemvBf16::eligible(&bf(&[1, 2, 64]), &w, &sc, &bi));
        // width must be a group multiple and match the packed/scales geometry.
        assert!(!Q4GemvBf16::eligible(&bf(&[1, 1, 32]), &w, &sc, &bi));
        let w_bad = Array::zeros(&s, &[16, 9], MlxDType::U32).unwrap();
        assert!(!Q4GemvBf16::eligible(&bf(&[1, 1, 64]), &w_bad, &sc, &bi));
        let sc_bad = bf(&[16, 2]);
        assert!(!Q4GemvBf16::eligible(&bf(&[1, 1, 64]), &w, &sc_bad, &bi));
        let w_f32 = Array::zeros(&s, &[16, 8], MlxDType::F32).unwrap();
        assert!(!Q4GemvBf16::eligible(&bf(&[1, 1, 64]), &w_f32, &sc, &bi));
    }
}
