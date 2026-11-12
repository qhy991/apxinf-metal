// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
// Causal depthwise convolution over [1, L, C] with an optional history buffer,
// FP32 accumulation. With zero history rows and a null history pointer this is
// bit-identical to the original zero-history kernel: the history branch is not
// taken, so the accumulation order and rounding are unchanged.
    const int c = thread_position_in_grid.x;
    const int t = thread_position_in_grid.y;
    if (t >= length) return;
    auto w = weight + c * kernel_size;
    float acc = 0.0;
    for (int i = 0; i < kernel_size; ++i) {
        const int row = t + i - (kernel_size - 1);
        const InT value = row >= 0
            ? x[row * C + c]
            : (history != nullptr ? history[(history_rows + row) * C + c] : InT(0));
        acc += static_cast<float>(value) * w[i];
    }
    out[t * C + c] = static_cast<InT>(acc);
