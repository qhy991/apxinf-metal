# d3-lite Mario demo — clean-window re-measure (relay leg, 2026-11-06)

P4 wired d3-lite into the Mario demo ([`d3-lite-metal-demo.md`](d3-lite-metal-demo.md), PR #24),
but its end-to-end window was **contaminated**: a resident `bench_local.py` harness plus its
Clef proxy and `apxinf-decide-server` held the GPU/ANE for the whole episode
(`Swapouts` +586 266 pages, swap 4 → 6 GiB, 3 foreign model processes), so every decision's
window recorded `Swapouts > 0` and the reported 732.5 ms median was a reference, not a baseline.
This leg follows the relay direction **"d3-lite demo 的干净窗复测"**: stop that harness, then
re-run the *identical* 12-decision episode in one lock window and report clean per-decision
latency and stage-split data. **Reference tier, no speedup claim.**

- Measurement host: Apple M1 Pro 16 GB, macOS 27.0.1
- Binary: P4's merged `target/release/apxinf-decide-server` (sha256
  `c744b11c704dcb753bc1fcadf9a50e73c0b9ae76a2993f95ef748ef6e8ba71d5`, worktree
  `~/src/apxinf-metal/mario-demo-adapt-a5b43c66ce96`)
- Checkpoint: `models/d3-lite-9c10703` (Qwen3.5-0.8B, BF16 + FP32 readout)
- Runs: `devlocal/notworth-audit/demo-runs/d3lite-clean-00{1,2}/`

## 1. Contaminating harness — confirmed and stopped

Confirmed by full command lines, then stopped (process group, `TERM` then `KILL`; no other
session's process touched):

| pid | command |
| --- | --- |
| 58121 | `~/Agent4Kernel/mario-harness-m1pro/mariobench/bench_local.py --repeats 3 --hold 7 --stall-limit 60 --policy single_structured --jump-hold 3 --jump-cooldown 1 --thr-gap 0.20 --union-width 112 --tag m1pro-w112-best` |
| 58124 | `mario-demo/devlocal/mario-demo-clef/scripts/decide_proxy.py --bin …/apxinf-decide-server --model …/clef-flash-4bit-b5adbe2 --variant auto --cache …` |
| 58125 | `…/clef-mario-opt-20261103/target/release/apxinf-decide-server --model …/clef-flash-4bit-b5adbe2 …` |

After the stop: no `bench_local` / decide-server / `d3_lite_bench` processes, `Swapouts` rate
0 over 3 s, both Metal locks free. The pre-existing tmux demo page on `:8766` is a 1.9 KB idle
`server.py` (no model) and is not a Metal user.

**Respawn:** the harness came back ~10 minutes after the stop — new process
group 60863 at 17:46:43, launched by `caffeinate -dimsu ./run-bench.sh
--repeats 3 --hold 7 --stall-limit 60 --policy single_structured …` under
`~/Agent4Kernel/mario-harness-m1pro/`, with the same Clef proxy + auto
decide-server children. The clean receipts above were completed before the
respawn and are unaffected, but any **future** demo/campaign window on this
host is at risk until the harness either takes the Metal locks or the
campaign's foreign-process gate waits it out.

## 2. Runner adaptations (all recorded, none hidden)

Copied P4's `real_d3lite_run.sh` to this ticket's
`devlocal/notworth-audit/scripts/real_d3lite_clean.sh` with three changes:

1. **Output directory** → this ticket's `devlocal/notworth-audit/demo-runs/<run>/` (a closed
   ticket's devlocal is not extended).
2. **Binary / sampler path** → P4's worktree (the merged binary and `vmstat_sampler.py`), so the
   re-measure evaluates the exact system P4 delivered.
3. **`pgrep`-no-match abort bug fixed**: under `set -e` + `pipefail`,
   `count=$(pgrep -f "$FOREIGN_PAT" | wc -l …)` exits non-zero when there are **zero** foreign
   processes — i.e. P4's script died precisely on the clean window we need (reproduced: run
   `d3lite-clean-001` in P4's worktree created only `lock.txt` and exited 1). Added `|| true`
   at both count sites.

Plus one addition: a **stage-attribution replay** (`run_correctness.py`, 18 cases) is run in the
same window, because the episode log carries only aggregate latency while the per-request
`stages` (`host_prep/layout/vision/forward/sync`) come from the server's responses. A failed
replay now marks the window contaminated.

## 3. Protocol

Identical to P4's: one lock window (`metal-measure.lock` → `enginetailor-metal.lock`, wait
never kill), foreign-process wait then check, `probe_load.py`, then
`evaluate.py --policy clef --max-decisions 12` with `MARIO_CLEF_PROTOCOL=separate` and
`APXINF_DECIDE_CMD` = the d3-lite decide-server, then the correctness replay, then the page
(3 browser steps on `:8767`). `vm_stat` snapshotted before/after and sampled every 200 ms
during the episode.

## 4. Window verdict — clean

Primary receipt: `d3lite-clean-002/verdict.txt` (complete: episode + replay + page).

| field | P4 (contaminated) | this leg, clean-002 |
| --- | ---: | ---: |
| `foreign_at_start` | 1 (3 processes) | **0** |
| `eval_status` / `eval_seconds` | 0 / 17 | **0 / 9** |
| `correctness_status` | — | **0** |
| `ui_status` | 0 | **0** |
| `swapouts_before` → `after` | 20 201 807 → 20 788 073 | **23 366 905 → 23 366 905** |
| `swapouts_delta` | **+586 266** | **0** |
| peak worker RSS | not claimed (sampler implausible) | **1 937 488 kB (1.94 GiB)** |
| peak swap used | 5 299 MB | **2 804 MB** |
| `contaminated` | 1 | **0** |

`d3lite-clean-001` (episode + page, no replay) is independently clean: `swapouts_delta=0`,
`foreign_at_start=0`, episode-window sampler `Swapouts` delta 0, `Swapins` +139.

## 5. Per-decision latency (clean)

Episode: 12 decisions, `decision_limit`, x=213, wall **8.72 s**, 36 local requests, $0.
Median **613.6 ms** (min 612.2, max 1470.4); decisions 2–12 are **612.2–615.6 ms**.

| # | x_before | action | latency (ms) | `Swapouts` in window | p(enemy/obstacle/gap) |
| ---: | ---: | --- | ---: | ---: | --- |
| 1 | 40 | run | **1470.4** | 0 | 0.187 / 0.170 / 0.199 |
| 2 | 41 | run | 612.5 | 0 | 0.150 / 0.131 / 0.163 |
| 3 | 47 | run | 613.6 | 0 | 0.184 / 0.170 / 0.097 |
| 4 | 56 | run | 615.6 | 0 | 0.246 / 0.126 / 0.268 |
| 5 | 70 | run | 613.6 | 0 | 0.276 / 0.114 / 0.449 |
| 6 | 87 | run | 615.0 | 0 | 0.289 / 0.114 / 0.340 |
| 7 | 105 | run | 612.4 | 0 | 0.430 / 0.114 / 0.308 |
| 8 | 123 | run | 613.4 | 0 | 0.226 / 0.205 / 0.327 |
| 9 | 141 | run | 613.8 | 0 | 0.214 / 0.254 / 0.224 |
| 10 | 159 | run | 613.3 | 0 | 0.105 / 0.278 / 0.219 |
| 11 | 177 | run | 615.4 | 0 | 0.096 / 0.255 / 0.136 |
| 12 | 195 | run | 612.2 | 0 | 0.142 / 0.224 / 0.102 |

Probabilities and the trajectory are **identical to P4's episode** (same p(…) to three decimals,
same x sequence, same `decision_limit` at x=213): window cleanliness changes latency only, not
decisions. Per-decision `Swapouts` is 0 for every decision in both clean runs.

Clean vs contaminated headline: median **732.5 → 613.6 ms** (−16.2 %), episode wall
15.36 → 8.72 s (−43 %). The contaminated median was *inflated* by the resident Clef harness,
not merely noisier.

## 6. First-frame cost (reported separately, as requested)

Decision 1 = worker start (model load) + the first request of each of the three
`(length, candidate_count, image_tokens)` shapes.

| window | decision 1 | `probe_load` start-to-ready | note |
| --- | ---: | ---: | --- |
| P4, contaminated | **6100.5 ms** | 2998.8 ms | foreign Clef process resident; swap 4 → 6 GiB |
| this leg, clean-001 | **4934.2 ms** | 1898.4 ms | clean, checkpoint cold in page cache |
| this leg, clean-002 | **1470.4 ms** | 358.5 ms | clean, checkpoint page-cache warm (run 001 just read it) |

So the demo's real first-frame cost is a **range, ≈1.5–5 s clean (up to 6.1 s under
contamination)**, driven by (a) checkpoint page-cache warmth — a ~1.5 s swing in the load probe
and ~3.5 s in the episode's first decision — and (b) whether the compiled profiles still need
their trace: the first request of each shape pays extra in `forward` (clean: 372–433 ms vs a
124 ms steady forward; P4's contaminated replay: up to **2044.6 ms**). Decisions 2–12 never pay
either cost.

**As a follow-up optimization target (not a KEEP candidate under the standing gate):** the
episode *median* is unaffected by decision 1 (it is the maximum of 12), so prewarming the
decide-server at page/server start would remove the multi-second first-frame without moving the
KEEP metric. Treat it as a demo-experience target, gated by a first-decision budget rather than
the episode median.

## 7. Stage split (clean, same window)

From the correctness replay in `d3lite-clean-002` (18 cases; **steady** excludes the first case
of each of the three shapes, n = 15; shares from means so the parts close against the request
wall).

| Stage | Steady median (ms) | Steady mean (ms) | Share | P4 contaminated |
| --- | ---: | ---: | ---: | ---: |
| host preprocess (resize/normalize/patchify + prompt render/encode) | 13.00 | 13.16 | **6.4 %** | 6.1 % |
| layout (device→host readback + multimodal layout) | 0.08 | 0.08 | 0.0 % | 0.0 % |
| vision tower (12 blocks, eager, blocking) | 59.32 | 62.43 | **30.4 %** | 34.5 % |
| forward (compiled text layers + FP32 readout) | 124.15 | 129.78 | **63.1 %** | 59.2 % |
| sync (probability readback) | 0.08 | 0.10 | 0.0 % | 0.0 % |
| base64 + PNG decode + response assembly (residual) | — | 0.16 | 0.1 % | 0.1 % |
| **request** | **197.55** | **205.70** | 100 % | 210.72 |

Accounting closes: mean stages = 205.54 ms vs mean request wall 205.70 ms (residual 0.16 ms).
Clean vs contaminated: request median 210.72 → 197.55 ms (−6.3 %); vision and forward shares
shift because contamination penalised vision more (median 69.20 → 59.32 ms) than forward
(134.18 → 124.15 ms). Derived per-decision steady cost: 3 × 197.55 ≈ **593 ms**, matching the
clean episode's 612–616 ms decisions (the demo's decision includes crop/action overhead).

Correctness is unchanged by the window (deterministic): vs the checkpoint's PyTorch reference,
MPS **18/18** agreements, max|Δp| **0.0392**; CPU 17/18 (`f0058-enemy`, Δ 0.0769), inside the
declared 3× reference self-noise budget 0.1293.

## 8. Comparison — no speedup claim

Different models, so only each side's own latency is reported.

| Side | Per-request (`separate`) | Per-decision | Window |
| --- | ---: | ---: | --- |
| **Clef reference** (GPU-only, showcase) | ≈2200 ms | **6599 ms** median (6581–6887) | clean: `Swapouts` delta 0 |
| **d3-lite** (Metal, this leg) | **197.6 ms** median | **613.6 ms** median (612.2–1470.4) | **clean: `Swapouts` delta 0, foreign 0** |
| d3-lite (P4, contaminated) | 243.5 ms | 732.5 ms median (616.2–6100.5) | contaminated: `Swapouts` +586 266, 3 foreign |

## 9. Verification

| Check | Result |
| --- | --- |
| harness confirmation + stop | **pass** — three pids matched `bench_local.py` + proxy + Clef decide-server; group empty afterwards |
| foreign-process gate | **pass** — `foreign_at_start=0`, no `foreign-at-start.txt` |
| window `vm_stat` delta | **pass** — `Swapouts` 23 366 905 → 23 366 905 (0), both runs |
| episode sampler (200 ms) | **pass** — `Swapouts` delta 0, `Swapins` delta 0 (clean-002) |
| episode | **pass** — 12/12 decisions, 36 local requests, $0, `UI REAL CHECK PASSED` |
| correctness replay (stage data) | **pass** — `correctness_status=0`, 18 rows with `stages`, MPS 18/18 |
| page (3 browser steps on `:8767`) | **pass** |
| runner syntax / `pgrep`-fix regression | **pass** — clean window now proceeds (was exit 1 with only `lock.txt`) |

## 10. Artifacts

Private evidence (ignored): this ticket's
`devlocal/notworth-audit/demo-runs/d3lite-clean-00{1,2}/` (locks, foreign check, snapshots,
200 ms `vm_stat` trace, timestamped evaluate output, probe, `steps.json`, `correctness/`,
page screenshots, `verdict.txt`); runner + sampler in `devlocal/notworth-audit/scripts/`.
The re-measure uses P4's worktree for the merged binary and
`devlocal/mario-demo-adapt/ref/format-reference.json`; P4's contaminated receipt remains at
`devlocal/mario-demo-adapt/runs/d3lite-e2e-003/`.

```bash
# stop the resident harness first (human-directed): bench_local.py group + its Clef proxy/server
bash devlocal/notworth-audit/scripts/real_d3lite_clean.sh d3lite-clean-002
# verdict must read foreign_at_start=0, swapouts_delta=0, contaminated=0
```
