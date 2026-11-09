# Qwen3-0.6B decode: W8 weight path attribution and default switch

Date: 2026-11-10. Hosts: Apple M1 Pro (applegpu_g13s, 16 GiB) and Apple M4
(Mac16,10, 16 GiB). Model: Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, batch 1, greedy.
Branch `agent/weight-w8-path-388f52f6d7db`; probe `metal_text_probe` with
`APXINF_METAL_TEXT_MODEL=metal_qwen3` and explicit
`APXINF_METAL_TEXT_VARIANT` arms (Issue #68, W8 leg).

## TL;DR

- `mixed-w8` already had the right weight format (W8/group-64 projections at
  one input row, BF16 prefill) but its generation loop fell back to the
  synchronous host-sampled path: per token it downloaded the full
  `[1, 151936]` logits, ran a host argmax, and never overlapped steps. That
  pipeline choice, not dequantization, is what capped the historical gain at
  −13.2% instead of the device-side −28%.
- This leg admits `mixed-w8` to the device greedy path (one-step async
  lookahead) and makes `mixed-w8` the default variant; `bf16-compiled` remains
  the explicit opt-out. Measured paired TPOT (medians, 3 rounds, alternating
  arm order, Swapouts=0 on both hosts):

| Host | `bf16-compiled` | `mixed-w8` (this leg) | Δ TPOT |
|---|---:|---:|---:|
| M1 Pro | 9.809 ms | **6.702 ms** | **−31.7%** |
| M4 | 15.007 ms | **8.392 ms** | **−44.1%** |

- 1024-token consistency: free-running outputs match for the first 82 tokens
  (both hosts); teacher-forced replay (same prefix) gives **96.9% / 97.5%**
  (W8 on BF16 teacher, M1 Pro / M4) and **98.5% / 98.4%** (BF16 on W8 teacher)
  argmax agreement, with disagreements concentrated at near-tie low-confidence
  steps (median top probability there 0.31–0.42; mean |Δp| ≤ 0.0012). On M4
  the same-variant replay noise floor is already 1.4–1.5%, so cross-variant
  disagreement is at that floor; on M1 Pro both variants reproduce their own
  greedy trajectories 100% under replay.

## Where the historical −13.2% went

Decode per token (M1 Pro, serialized stage timing behind
`APXINF_QWEN3_DECODE_STAGES=1`; medians over 96 steady tokens):

| Stage | `bf16-compiled` | `mixed-w8` | Δ |
|---|---:|---:|---:|
| host graph build | 0.198 ms | 0.325 ms | +0.13 ms |
| device eval | 10.262 ms | 7.425 ms | **−2.84 ms (−27.7%)** |

Roofline bookkeeping per token (weights 1.19 GB BF16 → ≈0.62 GB W8+scales):

| Component | `bf16-compiled` | `mixed-w8` |
|---|---:|---:|
| weight-byte floor @200 GB/s | 5.95 ms | ≈3.1 ms |
| kernel-tap GPU busy (corrected) | ≈6.3 ms | ≈4.3 ms |
| kernels / token (tap) | 570 | 628 |
| command buffers / token | 21.6 | 21.1 |
| plain TPOT before this leg | 9.79 ms | 8.20 ms |
| plain TPOT after this leg | 9.78 ms | **6.67 ms** |

Attribution of the old gap:

1. **Device side worked as designed**: halving weight bytes cut device eval by
   27.7% (10.26 → 7.43 ms). Quantized GEMV runs at lower effective bandwidth
   than BF16 GEMV (≈83 vs ≈116 GB/s per weight byte), so the eval saving is
   less than the byte halving — that is the dequantization cost, and it is
   real but not dominant.
2. **The sync generation loop then gave ~1.5 ms/token back**: host argmax over
   151936 logits, ≈300 KB logits download per token, and no lookahead overlap
   (`mixed-w8` TPOT 8.20 ms vs its own device eval 7.43 ms). The same loop on
   `bf16-compiled` (`APXINF_MLX_SYNC_DECODE=1`) costs 11.2–12.5 ms vs 9.78 ms.
3. Fixing the loop (device greedy, this leg) recovers the overlap: W8 TPOT
   6.67 ms, below its serialized device eval because step N+1's eval overlaps
   step N's token readback.
4. What remains above the ≈3.1 ms W8 floor: 628 kernels / 21.1 command
   buffers per token of GPU-side scheduling (≈2.4 ms) plus qmm dequant
   throughput (≈1.2 ms). A compiled W8 decoder block is the next lever.

## Two-machine paired TPOT (acceptance)

3 rounds per host, alternating arm order, fresh process per arm, 3 repeats
each (steady repeats 1–2 reported), `vm_stat` Swapouts delta = 0 on every arm,
both Metal measurement locks held. 32-token outputs identical across repeats
and arms on M1 Pro; on M4 the two variants agree for 31/32 tokens (final-token
near tie, see the M4 table note).

### M1 Pro (applegpu_g13s)

| Round | bf16-compiled TPOT (steady) | mixed-w8 TPOT (steady) |
|---|---|---|
| g5 | 9.866 / 9.879 | 6.722 / 6.661 |
| g6 | 9.736 / 9.785 | 6.715 / 6.688 |
| g7 | 9.832 / 9.732 | 6.718 / 6.671 |
| **median** | **9.809 ms** | **6.702 ms** |

TTFT medians: 46.9 ms (bf16) vs 41.2 ms (w8). Long-run check (1024-token
parity case): 11.0 → 7.59 ms/token (−31%). Rows g5–g7 are the final PR
binary; the pre-rebase probe (identical except inter-repeat cleanup) measured
9.778 → 6.666 ms over rounds g1/g3/g4, within noise.

### M4 (Mac16,10)

| Round | bf16-compiled TPOT (steady) | mixed-w8 TPOT (steady) |
|---|---|---|
| r1 | 13.717 / 13.643 | 8.360 / 8.644 |
| r2 | 15.357 / 15.684 | 8.692 / 7.505 |
| r3 | 15.199 / 14.814 | 8.425 / 7.927 |
| **median** | **15.007 ms** | **8.392 ms** |

TTFT medians: 39.9 ms (bf16) vs 37.4 ms (w8). Staged (serialized) decode
medians on M4 (final binary): build 0.147 → 0.244 ms, device eval 13.928 →
8.137 ms (−41.6%). Long-run parity case: 14.55 → 8.27 ms/token (−43%), first
divergence at token 82, both run to 1024. The pre-rebase probe suite on the
same branch measured
13.659 → 7.641 ms with the identical paired delta (−44.1%); M4 absolute
times drift between suites more than M1 Pro's, so the paired delta is the
stable figure. The base M4's lower memory bandwidth makes the weight-byte
halving worth more than on the M1 Pro. 32-token outputs on M4 agree for
31/32 tokens, diverging only at the final-token near tie (bf16 31806 vs w8
8357); the same 1024-token near-tie behavior quantified below. M4 bf16 ids
match M1 Pro bf16 ids exactly.

## Text consistency gate (1024 tokens)

Corpus case: 44-token tutorial prompt, greedy, `max_tokens=1024`
(`devlocal/weight-w8-path/results/parity-corpus.json`). Baseline
`bf16-compiled` output stops at 960 tokens on M1 Pro (EOS) and runs to 1024 on
M4; `mixed-w8` runs to 1024 on both. The free-running pair diverges first at
token 82 (near-tie step) on both hosts. Token-identical over the full output
is therefore not claimed.

Teacher-forced replay (`APXINF_METAL_TEXT_FORCE_TOKENS`, this leg's probe
mode; each arm replays the other arm's ids through prefill + one-row forwards
and reports per-step argmax probabilities):

| Replay | M1 Pro | M4 |
|---|---:|---:|
| bf16 on its own ids (sanity) | 960/960 (100%) | 946/960 (98.5%) |
| w8 on its own ids (sanity) | 1024/1024 (100%) | 1010/1024 (98.6%) |
| **w8 on bf16 teacher** | **930/960 (96.9%)**, max \|Δp\| 0.077 | **936/960 (97.5%)**, max \|Δp\| 0.285 |
| **bf16 on w8 teacher** | **1009/1024 (98.5%)**, max \|Δp\| 0.299 | **1008/1024 (98.4%)**, max \|Δp\| 0.183 |

Mean \|Δp\| is ≤ 0.0012 over every replay. All disagreement steps across both
directions and both hosts sit at low-confidence near-ties (median top
probability 0.31–0.42), i.e. steps where either choice is defensible. On M4
the same-variant sanity replays already disagree at 1.4–1.5% of steps (device
vs host argmax tie-breaking), so the cross-variant 1.6–2.5% disagreement is at
the replay-method noise floor there; the M4 32-token final-token flip is the
same near-tie class, not a W8-specific defect. This matches the historical
acceptance shape for this family (the `bf16-compiled` rounding change was
accepted with 222/1024 free-running token differences and 221/222
external-teacher agreement): W8 quantization is intentionally lossy, and the
default switch ships with the above evidence.

## Changes (branch `agent/weight-w8-path-388f52f6d7db`)

- `perf(metal-qwen3): run mixed-w8 decode on the device greedy path` —
  `token_sequence` admits `MixedW8` to `GreedySequence`; the device decode
  path was already variant-agnostic. `APXINF_MLX_SYNC_DECODE=1` still forces
  the synchronous loop.
- `perf(metal-qwen3): make mixed-w8 the default selection` — decode defaults
  to W8 projections with BF16 prefill; `bf16-compiled` stays as the explicit
  opt-out variant.
- `feat(metal-qwen3): per-token decode stage timing behind
  APXINF_QWEN3_DECODE_STAGES` — host-build/device-eval split per one-row
  step; steps are evaluated synchronously under the flag.
- `feat(diagnostics): teacher-forced replay mode in metal_text_probe` —
  `APXINF_METAL_TEXT_FORCE_TOKENS` for the parity gate above.
- Probe env overrides (`APXINF_METAL_TEXT_MODEL` / `_VARIANT`, `none` cleanup)
  carried over from the previous leg (`e5d9a8eb`).

## Verification

- `cargo test -p apxinf-model --features mlx --release metal_qwen3 -- --test-threads=1 --include-ignored`:
  all 11 metal_qwen3 tests pass on M1 Pro, including the Metal lifecycle tests
  (capacity growth, prepared-state, prefill schedule, reference trajectory).
- 32-token greedy outputs are identical across the two variants on M1 Pro
  (all rounds); on M4 they agree for 31/32 tokens (final-token near tie).
- Kerneltap (DYLD_INSERT_LIBRARIES dispatch/CB tap from the previous leg):
  mixed-w8+greedy steady window shows 628.1 kernels/token, 21.12 CB/token,
  host inter-dispatch gaps p50 0.06 µs; TPOT under the tap 6.694 ms matches
  the untapped 6.666 ms (tap overhead < 0.5%).

## Remaining risks / next candidates

1. **Compiled W8 decoder block** (biggest lever): fold the eager per-layer
   chain (628 kernels, 21 CBs) into a `Compiled` block like `bf16-compiled`'s,
   carrying packed W8 weights as explicit inputs. Expected −1.5~2.5 ms/token
   (GPU-side scheduling ≈2.4 ms). Falsifier: if qmm inside a compiled
   callable forces materialization boundaries and CB count stays ~20, the win
   shrinks to elementwise fusion only.
2. **qmm dequant throughput**: W8 GEMV effective bandwidth ≈83 GB/s vs ≈116
   GB/s for BF16 GEMV; a better M=1 quantized kernel shape would push toward
   the ≈3.1 ms floor.
3. Memory: `mixed-w8` keeps BF16 prefill + W8 decode weights resident
   (≈1.8 GB vs 1.2 GB); documented as a decode-performance selection, not a
   memory saver.

Evidence paths (M1 Pro): `devlocal/weight-w8-path/{logs,results,scripts}` —
paired logs `probe_g*`, staged `probe_staged_*`, parity `parity_*`,
teacher-forced `force_*`, kerneltap `tap_w8_*`. M4 evidence on the M4 host's
worktree `~/Agent4Kernel/apxinf-wt/weight-w8-path-388f52f6d7db/devlocal/`.
