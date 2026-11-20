# metal_lfm3 4-bit decode: fused affine-Q4 GEMV is at parity with BF16

Date: 2026-11-20. Host: M4 (Mac16,10), 16 GiB. Model: LFM3-1.7B BF16 snapshot
(flattened local copy, no `parts/`). Binary: `apxinf-serve` + `apxinf-native-worker`,
release, built from `feat/lfm3-q4-decode`.

## CORRECTION (2026-11-20, after PR #144)

**The mechanism section below is wrong and is kept only as history.** PR #144
measured the same fused kernel properly and found:

- MLX's M=1 `quantized_matmul` is **already a fused qmv** (register dequantise),
  so this receipt's premise -- that the kernel removes a dequantise step -- does
  not hold;
- end-to-end the fused kernel is **9.2% SLOWER**, not faster: +38% on ffn.w2
  (K=7680), **-10% on lm_head (N=128000)**, and slightly behind on the K=2048
  family that carries ~81% of the bytes;
- QUEUE #8 already measured qmm at **87.7% of roofline** on M4 (11.1 vs a 9.67 ms
  floor), consistent with there having never been room here;
- the token gate fails 1 of 3 greedy prompts (BF16-ulp boundary flips).

**What survives:** the parity result. The A/B below compared the fused GEMV
against **BF16 dense**, not against `quantized_matmul`; on that comparison both
arms sit at 37.7 ms/token, and that remains a true statement.

**What was wrong:** the sentence claiming the kernel "removes
quantized_matmul's dequantise overhead (43.0-44.2 -> 37.7)". That 43.0 -> 37.7
change was **the measurement method** (60 tokens with thinking noise versus 200
tokens with a closed thinking block), not the kernel. Comparing across those two
methods was the error, and the correct comparison (#144's) finds the opposite
sign.

**Do not reuse this kernel.** #144 carries the measured, token-gated, opt-in
implementation and the roofline evidence.

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
