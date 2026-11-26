# A multi-token forward is not the same computation as the same tokens stepped

**Status: applicable, not yet ported.** This note carries a constraint EngineTailor
measured on the LFM3 lane; it is recorded here because `crates/apxinf-model/src/metal_lfm3/`
dispatches the same way (row decode for `M=1`, GEMM for `M>1`), so any feature that
computes N tokens in one pass and expects the single-token-stepping answer inherits it.

## What EngineTailor measured

On the LFM3 engine (MLX, BF16), a verification chunk (T tokens in one forward) and the
same T tokens stepped one at a time **from the same committed state** do not agree:

| quantity | measured difference |
|---|---|
| next-token logits | up to **4.38e-01** |
| committed state after a chunk-equivalent path | **7.03e-03 (k/v) / 9.04e-03 (conv)** relative, ≈ **2 BF16 ulp** (ulp = 3.9e-03) |
| consequence | greedy argmax flips wherever the top-2 gap is below that noise |

Measured while testing prompt-lookup speculative decoding (4 prompts × 64 tokens: only
1/4 token-identical; first mismatches at tokens 29 / 63 / 34). A build whose drafter
never read the prompt passed 4/4, which is why the effect was initially missed.

## Why: the two paths call different matmul kernels

- **Metal**: `M=1` routes to `gemv`, `M>1` routes to the steel GEMM family (regular or
  SIMD split-K). Different kernels, different reduction order, different rounding.
- **CPU device**: MLX dispatches BF16 to BNNS, whose `M=1` and `M=8` algorithms differ
  (and `M=8` loses precision).

Every other operator compared from identical inputs is **bitwise identical**: RMSNorm,
QK-norm, `conv1d` vs the 7-tap window, `_conv_shift_add` vs the window, and SDPA (full
attention diverges on CPU only at KV ≥ 256; GPU through 512 is identical; the GPU sliding
band differs only on near-zero elements, ≤ 3.1e-5). The first divergence in the full
engine is layer 0's `in_proj`, with the layer input and the normed input bitwise equal.

This is therefore **not a batching law**. It is a kernel-selection split, and the same
split exists in apxinf: `q4_row_decode_f16` / `q4_transpose_decode_*` handle the
row-decode shape while `mpp_bf16_gemm_*` / `f16_gemm_bf16*` handle `M>1`.

## What this rules out

Any scheme that computes N tokens in one batching pass and expects the
single-token-stepping answer is unsound on this engine, including:

1. **Speculative decoding** (verify k drafts in a chunk, accept the matching prefix).
   Strict verification cannot make the accepted tokens the target's greedy stream,
   because the verification row is not the step row.
2. **Prefix-cache / snapshot-restore replay.** EngineTailor's earlier D2 investigation
   found that replaying a snapshot perturbed greedy trajectories and produced
   request-to-request skew (`admit_cache_len != admit_engine_pos`). The chunk-vs-step
   divergence is the most economical explanation: replay processes more than one token
   per forward where the original run stepped token by token.

**Tolerance and top-2-gap verification rules cannot restore token identity.** Even when
every argmax is judged right, the committed state is chunk-derived, so the next row's
reference has already drifted. The problem is the state, not the acceptance rule.

## What would fix it

No configuration does. EngineTailor checked the available levers:

| lever | outcome |
|---|---|
| any flag changing the per-shape kernel choice | none exists |
| FP32 accumulation | does not equalise the summation order, and doubles weight bytes |
| per-row `M=1` calls | exact, but reads the weights T times — slower than stepping |
| pre-transposed weights | disables the `b_transposed` path |
| row padding | only increases `M` |

The only path that keeps both exactness and the weight-stream amortisation is a **shared
multi-row kernel whose per-row reduction order matches the step kernel, used by both
paths**, plus per-row SDPA. That is a kernel project, not a configuration change, and it
would have to be justified by an end-to-end measurement — the same requirement the
row-decode kernels already face.

## How to check apxinf against this

Before adding chunked verification, prompt-lookup drafting or snapshot replay to
`metal_lfm3` (or any family with a row-decode/GEMM split):

1. Take a committed state, run one forward with `T` tokens, and run `T` forwards with one
   token each. Compare the last row's logits and the resulting state.
2. If they differ, the feature cannot be strict; either do the kernel project above, or
   state explicitly that the feature is approximate and bound the divergence.
3. Record the number beside the feature's evidence, not in a footnote: a 2-ulp state
   drift flips argmax at any top-2 gap under it, and that is a correctness property of
   the feature, not a measurement artefact.

## Source

EngineTailor-internal issue #548 (finding), #557 (attribution and fixability), PR #558.
Measured on Apple M4 (`Mac16,10`), MLX 0.32.3, BF16, registered cases `code-zh-fix` and
`qwen35_4b-documents-2048`.
