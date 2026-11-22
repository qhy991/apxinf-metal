# Clef / d3-lite on M1 Pro vs M4 mini — final report

Date: 2026-11-08. Both machines run the same repository (`qhy991/apxinf-metal`,
branch `showcase/clef-vision`), the same checkpoints and the same measurement discipline
(paired arms in one Metal lock window, reverse arm order across rounds, per-arm
`vm_stat` Swapouts, 42-case parity gate).

| | M1 Pro (haiyan-mac) | M4 mini (haiyan-Mac-mini) |
|---|---|---|
| Chip | Apple M1 Pro, 16 GB | Apple M4, 16 GB |
| GPU string | `applegpu_g13s` | `applegpu_g16g` |
| Xcode | 27.0 (27A266a) | 27.0 (27A266a) |
| **Core AI instrument** | **unusable** (dyld hang with Metal; 0 % CPU + GB spill alone) | **works** — `ane-hw-intervals` + `metal-gpu-intervals` |
| ANE split qualified | column split (6656/12288) | neuron split (5632 INT8 + 1248 FP16) |

---

## 1. Headline numbers

### 1.1 Mario demo — Clef Flash 9B Q4, one 12-decision episode (3 requests per decision)

| Metric | M1 Pro (auto) | **M4 (auto, resident server)** | Ratio |
|---|---:|---:|---:|
| Per-decision median | **6 599 ms** | **3 776–3 897 ms** | **1.7×** |
| Per-decision range | 6 581–6 887 ms | 3 125–4 319 ms | — |
| First decision | ~6 100 ms (profile trace + load) | 4 129–4 245 ms (warm server) | 1.5× |
| 11–12 decisions wall clock | ~73 s (11 decisions, death) | **44.7–46.5 s** (12 decisions, cap) | **1.6×** |
| Episode start-up | ~66–69 s (ANE bring-up per process) | **21–25 ms** (resident server) | ~3000× |
| One-time ANE bring-up | 66–69 s per process | **120.3 s once** (load 100.0 + warm 20.3) | — |
| Cloud calls / cost | 0 / $0 | 0 / $0 | — |

### 1.2 Single request, steady state

| Workload | M1 Pro | M4 |
|---|---:|---:|
| Same image repeated (Tier-1/Tier-2 warm, 70 % neuron share) | — | **382.2 ms** |
| Same image repeated (default auto 56 %) | — | 441.3 ms (−13.4 % for the 70 % point) |
| New image each request (demo-shaped) | ~2.2 s | **1.27–1.7 s** |
| 42-case suite, reference tier | 2 134 ms | not re-run |
| 42-case suite, auto tier | 1 429 ms (1.56× vs MLX) | not re-run |

M4's advantage is largest when the ANE split is tuned (neuron 70 %) and the server is
resident; a fresh process pays the 120 s bring-up.

---

## 2. What each machine taught us

### 2.1 M1 Pro — the work-horse line (2026-10-25 → 2026-11-07)

| Optimization | Result | Evidence |
|---|---|---|
| Tier-1 prefix KV cache (system + chat prefix) | **+8.72 %** reference tier, default ON | PR #5 line record |
| Tier-2 image-row prefix cache (same-image repeats) | **+45.327 %** (CI95 [42.80, 47.64]) | PR #6 |
| Operator-level ANE+GPU timeline (os_signpost + Metal/POI + Perfetto) | delivered; ANE predict 46.3 %, ANE wait 38.3 %, overlap 88.6 % of GPU busy, bubbles 8.7 % | PR #7 |
| d3-lite GDN kernel port | text **+4.27 %** | PR #21 |
| d3-lite request-level concurrency | steady **+8.42 %**, all-12 **+9.40 %** | PR #28/#29 |
| d3-lite final-layer row prune | **+4.374 %** | PR #31 |
| d3-lite Mario demo integration | 613.6 ms/decision clean window (12 decisions) | PR #24/#26 |
| Clef whole-backbone `mx.compile` | **+1.338 %** (default OFF) | PR #37 |
| Clef `joint` protocol (3 images in one request) | **NOT-WORTH** — 11.6 % *slower*, parity 38/42, maxΔp 0.274 | PR #35 |
| Clef ANE dedicated thread | **NOT-WORTH** — same-layer overlap already achieved, cross-layer is a hard dependency, ceiling ≤2 % | PR #40 |
| Clef kernel-level GDN fusion | **NOT-WORTH** — measurable ceiling 1.54 % | Issue #11 |
| Clef auto-tier prefix cache | **INCONCLUSIVE** at 42 cases (swap-contaminated); auto@N12 not 3/3 clean | PR #22 |

The M1 Pro line's structural conclusion: the two engines already overlap 88.6 % of GPU
busy, the remaining time is a 5.4 % host gap plus the 8.7 % true bubble, and every
"make it faster" candidate lands at 0.1–1.5 %. The wins come from **deleting work**
(prefix caches) or from **new models**, not from squeezing the existing schedule.

### 2.2 M4 mini — the hardware-visibility line (2026-11-07 → 2026-11-08)

The M4 is valuable for one reason: the **Core AI instrument works there**, so the ANE can
be measured at hardware level (`ane-hw-intervals`: program names, start/duration/occupancy).

| Step | Result |
|---|---|
| Instrument validation | 56 MB trace from a 35 s Clef-auto window; ANE + GPU intervals exported |
| First hardware picture | GPU **86.3 %** busy, ANE **16.4 %**, GPU-busy-while-ANE-idle **73.1 %** |
| Column-split scan (M1-Pro geometry on M4) | best point C=6144 **571.4 ms**, *worse* than auto 437.3 ms; column-split engine floor 513 ms > 437 ms → dead end |
| **Neuron-split tuning** | **70 % share = 382.2 ms**, −13.4 % vs default auto 441.3 ms; parity maxΔp 0.00025; 84 % saturates (386.9 ms) and costs 4.5 GB residency |
| Shape warm-up (`--warm-shapes`) | first decision **16.7 s → 3.8 s**, one-time 16.5 s before `ready` |
| Resident decide-server (`--listen unix:…` + shim) | per-episode start **21–25 ms** (was 93–130 s); two consecutive episodes 46.5 s / 44.7 s |
| d3-lite on M4 | port + parity + baseline (placeholder; the d3 work is M1-Pro-anchored) |

---

## 3. The three mechanisms behind the M4 numbers

1. **Neuron split beats column split on M4.** M4's qualified composition splits the MLP by
   neurons (5632 INT8 + 1248 FP16 resident on the ANE), not by columns. Forcing the M1-Pro
   column geometry costs 30.7 % because the ANE's per-request cost grows 4.2×.
2. **Shape-compiled profiles are the demo's launch tax.** Every new (prompt-length, image
   shape) pair triggers `Compiled::prepare` (trace + two full evals). Per-shape first use
   measured 5.4 / 37.4 / 93.7 s on a cold kernel cache; warm shapes are 1.3–5.1 s. The
   `--warm-shapes` manifest moves that cost into `ready`.
3. **ANE bring-up is per process.** Load measures 66–69 s on M1 Pro and 93–130 s on M4 for
   `auto`, versus **2.2 s for `reference` (GPU-only)** — the cost is the ANE program build
   and placement inspection, and the Core ML cache barely helps (65.8 s fresh vs 69.3 s
   warm on M1 Pro). The only reliable answer is a long-lived process: the resident server
   pays 120.3 s once and then serves episodes in 21–25 ms.

---

## 4. What "peak performance" looks like today

| Scenario | Best measured configuration |
|---|---|
| Same-image repeat traffic (pause/analyse) | **382 ms/request** — M4, neuron 70 %, warm shapes and resident server |
| Mario demo, per decision | **3.78–3.90 s** — M4 resident server (vs 6.60 s M1 Pro) |
| Mario demo, per episode | **44.7–46.5 s** — M4 resident server (vs ~73 s M1 Pro) |
| 42-case batch, auto tier | 1 429 ms/request — M1 Pro (M4 not re-run) |
| d3-lite single-image decision | 613.6 ms/decision — M1 Pro (12 decisions) |

---

## 5. Limits and open items

- **M4 first start is 122–210 s, and the disk cache cannot remove it.** A controlled two-run
  experiment (same `APXINF_COREML_CACHE`, cold then warm) measured **209.6 s cold vs 122.1 s
  warm with no new cache files written** — i.e. the warm run hit the cache, and the remaining
  122 s is Core ML *loading and placement inspection* of the 32 compiled `.mlmodelc`
  packages, not compilation. A `sample` of the cold run puts 91.5 % of samples in
  `ProgramCache::compiled → Package::write → fs::write` (4.1 GB written), so the split is
  roughly: ~90 s compile+write on a cold cache, ~120 s instantiate+placement on every
  process. **The resident server (#48) is therefore the correct and only mitigation** — it
  pays that once and serves episodes in 21–25 ms. Compressing it further would require
  fewer ANE programs or lazy Core ML loading, both larger changes.
- **Parity budgets are tight on Clef auto** (maxΔp 0.0328 vs gate 0.035); every change
  re-runs the 42-case gate.
- **The demo protocol matters**: `separate` (3 requests/decision) is what runs today;
  `joint` was measured and rejected (slower + parity flips).
- **Two machines, one geometry per chip**: a "portable" composition must select the split
  from the chip, not hard-code it; today the column geometry is qualified on g13s and the
  neuron split on g16g.
- **Instrument availability is the gate for future ANE work**: M1 Pro cannot record Core AI
  at all, so any further ANE-op-level question must be answered on M4 and only then
  cross-checked on M1 Pro by latency.

---

## 6. Reproduction index

| Artifact | Location |
|---|---|
| Branch | `showcase/clef-vision` (`qhy991/apxinf-metal`) |
| Clef line record | PR #5 (`Mario × ApxInf optimization — experiment line record`) |
| M4 line record | Issue #41 (`M4 mini：Clef ANE/GPU 列拆分调优 — 极限性能`) |
| Operator timeline methodology | `docs/mario-opt/TIMELINE-OPERATOR.md`, skill `timeline-method` |
| M4 documents | `docs/mario-opt/M4-COLUMN-SCAN.md`, `M4-NEURON-SPLIT.md`, `M4-DEMO-WARMUP.md`, `M4-RESIDENT-SERVER.md` |
| ANE thread / GDN fusion negative results | `docs/mario-opt/ANE-THREAD-DESIGN.md`, `GDN-KERNEL-FUSION.md` |
| Joint protocol negative result | `docs/mario-opt/JOINT-PROTOCOL.md` |
