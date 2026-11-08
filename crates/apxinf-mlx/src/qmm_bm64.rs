//! Bit-exact BM64 retile of MLX's affine quantized GEMM (`affine_qmm_t`),
//! for the image-token fallback tier (EngineTailor #237 round 24).
//!
//! The math and the FP32 K-accumulation order are MLX's own `qmm_t_impl`
//! (K chunked by BK=32, one threadgroup per output tile, per-output FP32
//! accumulation); only the compile-time tile changes from the stock
//! `(BM, BK, BN) = (32, 32, 32)` to `(64, 32, 32)`, so each dequantized
//! weight tile is reused across 64 M rows instead of 32 and weight traffic
//! halves. `qmm_t_impl` masks the M tail natively (`num_els =
//! min(BM, M - y_row)` with `load_safe`/`store_result_safe`), so ragged M
//! (e.g. the 1328x1328 tier's 6889 rows, `M % 64 = 41`) needs no split or
//! pad: one launch, bit-identical bytes to the stock op (EngineTailor rounds
//! 6/16 proved this tile-change bit-exact on the Python lane; the install
//! probe in `apxinf-model` re-verifies in this lane before enabling).
//!
//! The header inlines MLX's own kernel headers read-only, exactly as
//! `mx.fast.metal_kernel` users do; nothing in the installed MLX is edited.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};
use std::rc::Rc;

/// Compile-time tile: BM doubled, BK/BN at the round-6/7 probe winner.
const BM: i32 = 64;
const BK: i32 = 32;
const BN: i32 = 32;
const GROUP: i32 = 64;
const BITS: i32 = 4;
/// Stock dispatch group dims, mirrored from MLX's quantized gemm router.
const TG: [usize; 3] = [32, 2, 2];

/// Same body as the stock `affine_qmm_t` entry, parameterised on BM/BK/BN.
/// Attribute names must appear literally: `metal_kernel` turns the template
/// list into the generated kernel signature.
const SOURCE: &str = r"
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  uint3 tid = threadgroup_position_in_grid;
  uint lid = thread_index_in_threadgroup;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint simd_lid = thread_index_in_simdgroup;
  qmm_t_impl<T, group_size, bits, aligned_N, BM, BK, BN>(
      w, scales, biases, x, y, Xs, Ws,
      kdim, ndim, mdim, kdim,
      tid, lid, simd_gid, simd_lid);
";

/// MLX kernel headers concatenated in the order MLX itself uses for the
/// affine qmm (metal::gemm() + metal::quantized_utils() + metal::quantized());
/// the leading metal::utils() is prepended automatically by `metal_kernel`.
/// Quoted includes are inlined recursively with whole-build dedup (the
/// custom-kernel compiler has no search path for them and inlined text
/// cannot use pragma-once guards), the same text transformation as the
/// python-lane backend; angle includes resolve against the SDK.
fn mlx_kernel_headers() -> Option<String> {
    let rel = "include/mlx/backend/metal/kernels";
    let roots: Vec<std::path::PathBuf> = [
        std::env::var("APXINF_MLX_KERNEL_INCLUDE").ok().map(|v| v.into()),
        std::env::var("MLX_ROOT").ok().map(|v| format!("{v}/{rel}").into()),
        dirs_next(),
    ]
    .into_iter()
    .flatten()
    .collect();
    for root in roots {
        let mut seen = std::collections::HashSet::new();
        let mut header = String::new();
        let mut found = 0;
        for file in [
            "steel/gemm/gemm.h",
            "quantized_utils.h",
            "quantized.h",
        ] {
            seen.insert(file.to_owned());
            match std::fs::read_to_string(root.join(file)) {
                Ok(text) => {
                    header.push_str(&inline_quoted(&text, &root, &mut seen));
                    header.push('\n');
                    found += 1;
                }
                Err(_) => break,
            }
        }
        if found == 3 {
            return Some(header);
        }
    }
    None
}

/// Replace `#include "mlx/backend/metal/kernels/..."` lines with the file
/// contents (recursively, deduplicated across the whole header build);
/// quoted includes not under `root` are dropped, `#pragma once` lines are
/// removed (inlined text cannot rely on them), angle includes are kept.
/// Same transformation as the python-lane backend.
fn inline_quoted(text: &str, root: &std::path::Path, seen: &mut std::collections::HashSet<String>) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#include") {
            if let Some(start) = trimmed.find('"') {
                if let Some(end) = trimmed[start + 1..].find('"') {
                    let name = &trimmed[start + 1..start + 1 + end];
                    let rel = name.strip_prefix("mlx/backend/metal/kernels/").unwrap_or(name);
                    let key = rel.to_owned();
                    if !seen.contains(&key) {
                        seen.insert(key);
                        if let Ok(inner) = std::fs::read_to_string(root.join(rel)) {
                            out.push_str(&inline_quoted(&inner, root, seen));
                        }
                    }
                    continue; // quoted includes are expanded, not preserved
                }
            }
        }
        if trimmed.starts_with("#pragma once") {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(not(windows))]
fn dirs_next() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    [
        format!("{home}/mlx-sdk/include/mlx/backend/metal/kernels"),
        "/tmp/qwenane313/lib/python3.13/site-packages/mlx/include/mlx/backend/metal/kernels".to_owned(),
    ]
    .into_iter()
    .map(Into::into)
    .find(|p: &std::path::PathBuf| p.join("quantized.h").exists())
}

#[cfg(windows)]
fn dirs_next() -> Option<std::path::PathBuf> {
    None
}

/// `[1, T, K] x [N, K]^T -> [1, T, N]` affine Q4 (group 64) with the BM64
/// tile; byte-identical to [`Array::quantized_matmul`] on every eligible
/// call. Eligibility mirrors what the kernel template actually requires.
pub struct QmmBm64 {
    kernel: MetalKernel,
    stream: Stream,
}

impl QmmBm64 {
    pub fn new(stream: &Stream) -> Result<Self> {
        if stream.metal_index().is_none() {
            return Err(Error::Contract("QmmBm64 requires a Metal stream"));
        }
        let header = mlx_kernel_headers()
            .ok_or_else(|| Error::Contract("MLX kernel headers not found"))?;
        let kernel = MetalKernel::new(
            stream,
            "apxinf_qmm_bm64_affine_t",
            &["w", "scales", "biases", "x", "kdim", "ndim", "mdim"],
            &["y"],
            SOURCE,
            &header,
        )?;
        Ok(Self {
            kernel,
            stream: stream.clone(),
        })
    }

    /// Whether this call is safe to run on the BM64 tile.
    pub fn eligible(&self, x: &Array, w: &Array, scales: &Array, biases: &Array) -> bool {
        let shape = x.shape();
        let Some(&k) = shape.last() else {
            return false;
        };
        let [n, wk] = *w.shape() else {
            return false;
        };
        let m = x.numel() / k.max(1);
        x.dtype() == MlxDType::BF16
            && w.dtype() == MlxDType::U32
            && scales.dtype() == MlxDType::BF16
            && biases.dtype() == MlxDType::BF16
            && wk * 8 == k
            && k > 0
            && k % GROUP as usize == 0
            && scales.shape() == [n, k / GROUP as usize]
            && biases.shape() == [n, k / GROUP as usize]
            && n > 0
            && n % BN as usize == 0
            && m > 0
            && m * n <= i32::MAX as usize
    }

    /// Run the retiled kernel. `x` is `[1, T, K]` (leading dims flattened
    /// exactly as MLX's non-batched qmm path does); returns `[1, T, N]`.
    pub fn call(&self, x: &Array, w: &Array, scales: &Array, biases: &Array) -> Result<Array> {
        if !self.eligible(x, w, scales, biases) {
            return Err(Error::Contract("QmmBm64 eligibility failed"));
        }
        let shape = x.shape();
        let k = *shape.last().expect("checked");
        let m = x.numel() / k;
        let n = w.shape()[0];
        // 0-d arrays: MLX's metal_kernel passes single-element int32 inputs
        // as `constant int&`, which is what qmm_t_impl's K/N/M expect.
        let i32s = |v: usize| Array::from_i32(&self.stream, &[], &[v as i32]);
        let (k_arr, n_arr, m_arr) = (i32s(k)?, i32s(n)?, i32s(m)?);
        let mut out_shape = shape.to_vec();
        let last = out_shape.len() - 1;
        out_shape[last] = n;
        let grid = [
            (n / BN as usize) * TG[0],
            ((m + BM as usize - 1) / BM as usize) * TG[1],
            TG[2],
        ];
        // SAFETY: inputs are row-contiguous (MetalKernel::new ensures it);
        // every flattened extent of x [1,T,K], the packed w [N,K/8], the
        // scales/biases [N,K/64], and the [1,T,N] output fits i32 by the
        // eligibility check. Each 128-thread group owns the 64x32 output tile
        // at rows 64*tid.y/TG[1] .. and columns 32*tid.x/TG[0] ..;
        // qmm_t_impl masks rows >= M natively and N is a multiple of BN, so
        // stores stay in bounds and do not overlap.
        let out = unsafe {
            self.kernel.call(
                &[
                    w.clone(),
                    scales.clone(),
                    biases.clone(),
                    x.clone(),
                    k_arr,
                    n_arr,
                    m_arr,
                ],
                &[(&out_shape, MlxDType::BF16)],
                grid,
                TG,
                &[
                    ("T", TemplateArg::DType(MlxDType::BF16)),
                    ("group_size", TemplateArg::Int(GROUP)),
                    ("bits", TemplateArg::Int(BITS)),
                    ("aligned_N", TemplateArg::Bool(true)),
                    ("BM", TemplateArg::Int(BM)),
                    ("BK", TemplateArg::Int(BK)),
                    ("BN", TemplateArg::Int(BN)),
                ],
            )?
        };
        Ok(out.into_iter().next().expect("one output"))
    }
}

/// Shared install (one per stream); construction inlines MLX headers and
/// compiles nothing — MLX compiles the kernel at first launch.
pub type SharedQmmBm64 = Rc<QmmBm64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eligibility_is_shape_checked() {
        // Mirror of QmmBm64::eligible's pure shape logic (CPU-side).
        fn eligible(x: &Array, w: &Array, sc: &Array, bi: &Array) -> bool {
            let shape = x.shape();
            let Some(&k) = shape.last() else { return false };
            let [n, wk] = *w.shape() else { return false };
            let m = x.numel() / k.max(1);
            x.dtype() == MlxDType::BF16
                && w.dtype() == MlxDType::U32
                && sc.dtype() == MlxDType::BF16
                && bi.dtype() == MlxDType::BF16
                && wk * 8 == k
                && k > 0
                && k % 64 == 0
                && sc.shape() == [n, k / 64]
                && bi.shape() == [n, k / 64]
                && n > 0
                && n % 32 == 0
                && m > 0
        }
        let s = Stream::cpu().unwrap();
        let bf = |shape: &[usize]| Array::zeros(&s, shape, MlxDType::BF16).unwrap();
        let x = bf(&[1, 64, 64]);
        let w = Array::zeros(&s, &[64, 8], MlxDType::U32).unwrap();
        let sc = bf(&[64, 1]);
        let bi = bf(&[64, 1]);
        assert!(eligible(&x, &w, &sc, &bi));
        assert!(!eligible(&bf(&[1, 64, 63]), &w, &sc, &bi)); // K % 64
        assert!(!eligible(&bf(&[1, 64, 32]), &w, &sc, &bi)); // K < group
        let w2 = Array::zeros(&s, &[48, 8], MlxDType::U32).unwrap();
        assert!(!eligible(&x, &w2, &bf(&[48, 1]), &bf(&[48, 1]))); // N % 32
        assert!(!eligible(&x, &w, &bf(&[64, 2]), &bi)); // scales shape
    }
}
