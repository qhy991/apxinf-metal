// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
// Fused affine-Q4 (group 64) GEMV for M=1 decode: one 256-thread threadgroup
// (8 simdgroups) covers eight output rows — one row per simdgroup — and lanes
// stride the K/64 groups, dequantising in registers, so no [N, K] expansion is
// ever materialised. Per group the affine form is factored as
//   sum_k (s*q_k + b) * x_k  =  s * sum_k q_k*x_k  +  b * sum_k x_k
// so the inner loop is one FMA per element plus a two-op group epilogue.
// K is substituted into the source by the Rust wrapper (one compile per
// distinct row width); the row count arrives as a runtime constant.
    const uint oct = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint row = oct * 8u + simd_group;
    constexpr uint k = K_PLACEHOLDER;
    constexpr uint groups = k / 64u;
    constexpr uint words = k / 8u;
    const uint nrows = static_cast<uint>(n);
    float acc = 0.0f;
    if (row < nrows) {
        for (uint g = lane; g < groups; g += 32u) {
            const float s = float(scales[row * groups + g]);
            const float b = float(biases[row * groups + g]);
            const device uint32_t* wrow = weight + row * words + g * 8u;
            // One 32-byte vector load for the group's eight packed words, and
            // eight 16-byte vector loads for the matching x values (offsets
            // are 16-byte aligned: x base is 16B aligned and each group
            // starts at a 128-byte boundary).
            const vec<uint32_t, 4> w01 = reinterpret_cast<const device vec<uint32_t, 4>*>(wrow)[0];
            const vec<uint32_t, 4> w23 = reinterpret_cast<const device vec<uint32_t, 4>*>(wrow)[1];
            const device vec<bfloat16_t, 8>* xg =
                reinterpret_cast<const device vec<bfloat16_t, 8>*>(x + g * 64u);
            float qx = 0.0f;
            float sx = 0.0f;
            #define APXINF_Q4_WORD(wv, xv)                                              \
                {                                                                       \
                    vec<float, 8> xf = vec<float, 8>(xv);                               \
                    sx += xf[0] + xf[1] + xf[2] + xf[3] + xf[4] + xf[5] + xf[6] + xf[7]; \
                    qx += float((wv >> 0u) & 0x0fu) * xf[0];                            \
                    qx += float((wv >> 4u) & 0x0fu) * xf[1];                            \
                    qx += float((wv >> 8u) & 0x0fu) * xf[2];                            \
                    qx += float((wv >> 12u) & 0x0fu) * xf[3];                           \
                    qx += float((wv >> 16u) & 0x0fu) * xf[4];                           \
                    qx += float((wv >> 20u) & 0x0fu) * xf[5];                           \
                    qx += float((wv >> 24u) & 0x0fu) * xf[6];                           \
                    qx += float((wv >> 28u) & 0x0fu) * xf[7];                           \
                }
            APXINF_Q4_WORD(w01[0], xg[0]);
            APXINF_Q4_WORD(w01[1], xg[1]);
            APXINF_Q4_WORD(w01[2], xg[2]);
            APXINF_Q4_WORD(w01[3], xg[3]);
            APXINF_Q4_WORD(w23[0], xg[4]);
            APXINF_Q4_WORD(w23[1], xg[5]);
            APXINF_Q4_WORD(w23[2], xg[6]);
            APXINF_Q4_WORD(w23[3], xg[7]);
            #undef APXINF_Q4_WORD
            acc += s * qx + b * sx;
        }
    }
    const float total = simd_sum(acc);
    if (lane == 0 && row < nrows) {
        out[row] = bfloat16_t(total);
    }
