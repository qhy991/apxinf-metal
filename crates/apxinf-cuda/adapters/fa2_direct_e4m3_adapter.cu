// Copyright 2026 ApxInf contributors.
// Stable C ABI for the exact-shape direct-E4M3 FA2 route.

#include "../kernels/cutlass/fa2_f16_e4m3_sm100.cu"

extern "C" int apxinf_static_fa2_f16_direct_e4m3_522(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float output_scale,
    cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fa2_f16_direct_e4m3_522(
      q, k, v, output, softmax_lse, batch, query_tokens, key_tokens,
      query_heads, kv_heads, head_dim, output_scale, stream);
}
