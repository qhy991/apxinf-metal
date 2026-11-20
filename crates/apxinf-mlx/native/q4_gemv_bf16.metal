// Fused affine-Q4 (group 64) GEMV for M=1 decode: one threadgroup per output row,
// 32 lanes over the K/64 groups, weights dequantised in registers.
// Requires K_PLACEHOLDER substituted with the row width (multiple of 2048).
    const uint row = threadgroup_position_in_grid.y;
    const uint lane = thread_position_in_threadgroup.x;
    const uint k = K_PLACEHOLDER;
    const uint groups = k / 64u;
    const uint words = k / 8u;
    float acc = 0.0f;
    for (uint g = lane; g < groups; g += 32u) {
        const float s = float(scales[row * groups + g]);
        const float b = float(biases[row * groups + g]);
        const uint wbase = row * words + g * 8u;
        const uint xbase = g * 64u;
        #pragma clang loop unroll(full)
        for (uint w = 0; w < 8u; ++w) {
            const uint packed = weight[wbase + w];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 4u; ++j) {
                const float q0 = float((packed >> (4u * j)) & 0x0fu);
                const float q1 = float((packed >> (4u * j + 16u)) & 0x0fu);
                acc += (s * q0 + b) * float(x[xbase + w * 8u + j]);
                acc += (s * q1 + b) * float(x[xbase + w * 8u + 4u + j]);
            }
        }
    }
    const float total = simd_sum(acc);
    out[row] = bfloat16_t(total);
