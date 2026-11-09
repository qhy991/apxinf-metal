# `to_v` contracted-axis split to the Neural Engine — design (#242, owner-approved continuation)

Owner protocol (issue #242, 2026-11-09): a split design first, with the ANE
side's shapes confirmed compilable; then a **single projection** A/B under the
full measurement discipline (fresh process, same window, reversed reruns,
parity gate, swap-clean, before/after dual-axis timeline). This document is
step 1; the A/B receipts land in the EngineTailor run archive.

## Which projections are candidates, and which split each can express

`to_q`/`to_k`/`to_v`/`to_out` are four separate 4096×4096 affine-Q4
projections (`weights.rs`). Offloading a *share* of one means the ANE's
partial product and the GPU's must **add** on the GPU — the only shape the
shipped MLP split has. On that constraint the design space collapses:

| projection | output-column split | contracted-axis split | status |
|---|---|---|---|
| `to_out` | concat, not a sum | sums, but its input only exists after SDPA | Phase A measured it (heads on the contraction axis): **+8.6%, STOP** |
| `to_q`, `to_k` | concat, not a sum | sums, but the partial must return **before** norm/RoPE | never measured (option B, skipped) |
| `to_v` | concat, not a sum | sums, and `v` has **no norm and no RoPE** — the partial can be added right after the GEMM | never measured — **first subject** |

`to_v` is the first subject because it is the cheapest decomposition that is
still faithful to the owner's "single projection" instruction: no norm/RoPE
re-implementation, no weight-layout risk, and the ANE predict is queued
while the GPU still has `to_q`/`to_k` to run — the one structural difference
from Phase A, whose ANE call had nothing to hide under and paid +8.6% for it.

## Boundary with softmax/KV

The ANE program ends **before** softmax: it computes only the partial
product `attn_input[:, DIM-n:] × W_v[:, DIM-n:]ᵀ → v_partial [T, 4096]`. The
GPU keeps the leading columns, adds the two partials, then performs the head
reshape, the text-prefix concat, SDPA, and everything downstream unchanged.
The sync point sits between the `to_v` GEMM and SDPA — inside attention, as
option B always does; whether the ANE's serial host time hides under the
GPU's `to_q`/`to_k` queue is the question the A/B answers.

## ANE side: shapes, weights, compile feasibility

Per block, one extra Core ML program when `APXINF_QWEN_IMAGE_ANE_V_COLS=n`
is set (default 0 = off = rollback):

```
input  x  fp16 [1, n, 1, 1152]      (trailing contraction cols, channels first)
weight   fp16 [4096, n, 1, 1]        (dequantized trailing cols of to_v)
output y  fp16 [1, 4096, 1, 1152]
```

- One FP16 1×1 convolution; no INT8, no `r`/`s` scaling pair (weights and
  input are FP16 already, so the program reproduces the GPU's arithmetic
  without a scaling round trip).
- At the A/B share (n = 2048, 50%) the program carries 2048×4096×2 B =
  16.8 MB per block, 537 MB across 32 blocks. The measured residency wall on
  this 16 GB host sits between 7.12 and 7.88 GB of resident programs; the
  shipped INT8 MLP base is 2.94 GB, so the A/B peak (~3.5 GB) is far under it.
- **Compile feasibility is already proven for this exact shape class**: the
  Phase A `to_out` share at 8 heads compiled and ran 32 of these programs
  (K=1024→4096 FP16 conv over TILE=1152; `source=compile` in the receipts).
  The A/B's own install re-confirms it (`v_programs=32`).
- Cache manifest is its own key (`qwen-image21/attn-v-contracted/v1`) with
  the column count in it (A47: a tier that hits the cache was not the tier
  that was asked for). The MLP programs' key is untouched — a v-share change
  must not invalidate 2.94 GB of cached MLP programs.

## GPU side

`gpu_share` narrows `b.to_v` to the leading `DIM - n` contraction columns
(group-64 aligned slices of the packed weight, scales, and biases); the
candidate arm computes its partial with `Q4::apply` on the sliced weight and
adds the ANE's FP16 partial back in BF16. The control arm (n unset) keeps
byte-identical wiring: `gpu_share`'s slice is then the whole weight and the
block takes the existing fused `MppGemmHeadsWithPrefix` path.

## What the arithmetic predicts (measured terms only, from the round-25 recipe)

The recipe's rule — `GPU wall recovered > work moved / (Engine rate ×
contention factor)` — with this round's measured terms:

- work moved per call at n=2048: 2048×4096×1152 = 9.66 GMAC;
- Engine rate under contention: 0.55 × the standalone ~1.9 TMAC/s for this
  shape class → ~1.0 TMAC/s → ~9.4 ms of ANE predict, plus ~4.5 ms of
  measured per-call host/copy overhead → ~14 ms of serial host time;
- GPU work queued behind the feed (to_q + to_k + the GPU's v partial):
  ~18–20 ms — so the ANE call *can* hide, unlike Phase A;
- GPU wall recovered: ~4.8 ms of `to_v` arithmetic at the GPU's ~2 TMAC/s,
  lossy at ~⅔ → ~3 ms.

Net prediction: somewhere between −1% and +1% e2e — under the KEEP bar
(≥2% with CI95 above 0) on either sign, so the A/B is a falsification probe
either way, and the mechanism the timeline shows (hidden vs exposed ANE
serial time) is the reusable result.

## Parity gate on this target

The owner's protocol names the Clef-line gate (`42/42`, `max|Δp| ≤ 0.035`) —
a teacher-forced probability-delta harness that does not exist on this image
target; its equivalent here, per the round's established reviewed-tolerance
protocol, is: share-off output **byte-identical** to the unmodified binary,
and share-on vs share-off same-seed renders passing PSNR ≥ 40 dB, mean |Δ|
≤ 0.5/255, zero pixels Δ ≥ 32 — all comfortably inside FP16-rounding
territory (Phase A measured PSNR 51.09 dB / mean |Δ| ≈ 0.2 / hottest 3.17 for
the same arithmetic shape). The mapping is flagged to the owner in the run
report so the gate can be re-specified if a different bar was meant.

## Red lines honored

Serial measurement on the 16 GB host; no new measurement when swap used
exceeds 8 GB (checked per arm); every arm records `vm_stat` swapouts and a
non-zero delta discards the arm.
