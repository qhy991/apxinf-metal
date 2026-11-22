// Copyright 2026 apxinf contributors.
// Stable C ABI adapter for native SM100/SM110 BF16 GeGLU GEMM fusion.

#include "../kernels/cutlass/bf16_operators_sm100.h"

extern "C" int apxinf_static_cutlass_bf16_gemm_geglu(
    const void* activation, const void* packed_weight, const void* gate,
    void* output, int m, int n, int k, int full_n, int tactic,
    cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::bf16_gemm_geglu(
      activation, packed_weight, gate, output, m, n, k, full_n, tactic,
      stream);
}

extern "C" int apxinf_static_cutlass_bf16_dual_gemm_geglu(
    const void* activation, const void* interleaved_weight, void* output,
    int m, int n, int k, int full_n, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail::production_dual_geglu_bf16(
      activation, interleaved_weight, output, m, n, k, full_n, stream);
}
