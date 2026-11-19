// Shape-preserving lm_head GEMV (QUEUE #2 / doc/lfm3-lm-head-custom-kernel.md).
// Threads are laid out across output columns: a warp reads 32 consecutive
// columns of one weight row (64 B coalesced), while every column keeps one
// FP32 accumulator updated over k = 0..K-1 in ascending order — the exact
// single-output reduction form the pre-registration freezes. Input [1,1,K]
// BF16, weight [K,V] BF16, output [1,1,V] FP32. No K-split, no extra rows.
    const uint local = simdgroup_index_in_threadgroup * 32
        + thread_index_in_simdgroup;
    const uint col = threadgroup_position_in_grid.x * THREADS + local;

    float acc = 0.0f;
    if (col < uint(V)) {
        for (int k = 0; k < K; ++k) {
            acc += static_cast<float>(x[k])
                * static_cast<float>(weight[k * V + int(col)]);
        }
        out[col] = acc;
    }
