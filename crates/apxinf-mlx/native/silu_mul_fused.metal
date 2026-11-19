// Fused silu(g) * u: gate.mul(gate.sigmoid()).mul(up), element-local FP32.
// Same elementwise chain and rounding as the unfused graph; see
// doc/lfm3-lm-head-custom-kernel.md.
    const uint i = thread_position_in_grid.x;
    if (i >= ELEMENTS) return;
    const float g = static_cast<float>(gate[i]);
    const float s = 1.0f / (1.0f + metal::exp(-g));
    const float u = static_cast<float>(up[i]);
    out[i] = static_cast<T>(g * s * u);
