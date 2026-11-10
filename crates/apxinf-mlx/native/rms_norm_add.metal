// Residual add fused with RMS normalization: one pass over a BF16 row.
// out_hidden = bf16(x + y); out_normed = bf16(norm(out_hidden) * weight).
// The add rounds to BF16 before the statistics, matching an unfused
// `add` followed by `fast_rms_norm` on the BF16 result.
uint row = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint simd_group = simdgroup_index_in_threadgroup;
uint simd_groups = simdgroups_per_threadgroup;
const uint WIDTH = static_cast<uint>(H);
const uint ROW_OFFSET = row * H;

threadgroup float partials[32];
threadgroup float row_values[H];
float total = 0.0f;

for (uint column = lid * 4u; column < WIDTH; column += 256u * 4u) {
    #pragma unroll
    for (uint j = 0; j < 4u; ++j) {
        uint index = column + j;
        if (index < H) {
            float sum = static_cast<float>(x_input[ROW_OFFSET + index])
                + static_cast<float>(y_input[ROW_OFFSET + index]);
            bfloat16_t rounded = static_cast<bfloat16_t>(sum);
            row_values[index] = static_cast<float>(rounded);
            hidden_output[ROW_OFFSET + index] = rounded;
            total = total + row_values[index] * row_values[index];
        }
    }
}
total = simd_sum(total);
if (lane == 0u) {
    partials[simd_group] = total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_group == 0u) {
    float group_value = lane < simd_groups ? partials[lane] : 0.0f;
    group_value = simd_sum(group_value);
    if (lane == 0u) {
        partials[0] = group_value;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float scale = rsqrt(partials[0] / static_cast<float>(WIDTH) + eps_input[0]);

for (uint column = lid * 4u; column < WIDTH; column += 256u * 4u) {
    #pragma unroll
    for (uint j = 0; j < 4u; ++j) {
        uint index = column + j;
        if (index < H) {
            float normalized = row_values[index] * scale * static_cast<float>(weight_input[index]);
            normed_output[ROW_OFFSET + index] = static_cast<bfloat16_t>(normalized);
        }
    }
}
