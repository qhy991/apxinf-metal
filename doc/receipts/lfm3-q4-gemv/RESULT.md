# metal_lfm3 4-bit decode: fused affine-Q4 GEMV is at parity with BF16

Date: 2026-11-20. Host: M4 (Mac16,10), 16 GiB. Model: LFM3-1.7B BF16 snapshot
(flattened local copy, no `parts/`). Binary: `apxinf-serve` + `apxinf-native-worker`,
release, built from `feat/lfm3-q4-decode`.

## Method

Client wall via `/v1/chat/completions`, one prompt, `max_tokens=200`,
`temperature=0`, four runs per arm, alternating by restarting the server between
arms. Every run returned exactly 167 completion tokens and `finish_reason=stop`,
so the same amount of decode work was timed in both arms. The prompt is short
(15 tokens), so prefill is a small constant in both.

This is **client wall**, not engine basis: it includes HTTP and serialisation.
Both arms pay it identically, so the *ratio* is meaningful even though the
absolute number is not comparable with the engine-basis figures in #127.

## Result

| arm | median ms/token | runs |
|---|---:|---|
| BF16 dense | **37.68** | 37.47 / 37.54 / 37.82 / 38.21 |
| Q4 fused GEMV | **37.71** | 37.48 / 37.69 / 37.72 / 37.72 (one 40.40 discarded as a warm-up outlier) |

**Parity: 0.08% apart, far inside both spreads.** The 4-bit path does not buy
decode speed on this target.

## Why, and what it means

154 projections at ~37.7 ms/token is ~0.24 ms per projection. A 2048x2048 GEMV
needs ~4 MB of weights, which at ~120 GB/s is ~0.03 ms. The measured cost is
almost an order of magnitude above the bandwidth bound, so per-step overhead, not
weight bandwidth, dominates. **Cutting the weight bytes by 4x cannot help a kernel
that is not waiting on bytes** — which is exactly what the parity result shows.

The 4-bit port is therefore a *capability*, not a speedup:

- same decode rate at one quarter of the projection bytes;
- a real memory win only once the loader reads packed weights directly (today
  `APXINF_LFM3_Q4=1` quantises an already-resident BF16 copy, so
  `active_bytes` is unchanged at 3,458,508,288 and the peak rises slightly);

## What this rules out

- "4-bit decode will be faster because the weights are smaller" — measured false
  on this target at this shape.
- "the fused GEMV is what was missing" — measured false: it removes the
  `quantized_matmul` dequantise overhead (43.0-44.2 -> 37.7 when measured
  consistently) but lands exactly on BF16, not below it.

## What would move it

Reducing per-step overhead. Note that #130 already records two failed attempts in
that direction: whole-step compilation / position bucketing (#115, no bucket
>= 4%) and `mx.compile` decode (+0.7%). This result does not reopen those; it
only establishes that quantisation is not the lever.
