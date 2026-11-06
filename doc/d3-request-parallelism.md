# Request-level parallelism for the Mario demo (relay leg, 2026-11-06)

The demo's `separate` protocol runs one decision as **three independent single-image Noul
questions (enemy / obstacle / gap) sent serially**. That 3× serialization is the largest
structural lever left on the demo path. This leg follows the requester's mandate: let Opus
adjudicate the design first, then implement and measure — with the standing delivery rules
(3 rounds reverse order, `vm_stat` Swapouts gate, exact decision-sequence equality, and an
operator timeline for the Clef side).

- Host: Apple M1 Pro 16 GB, macOS 27.0.1, MLX 0.32.3 (`applegpu_g13s`)
- Worktree: `agent/notworth-audit-ee3c4e7da150` on `origin/main` (base `6bbf8fc7`)
- Demo checkout (shared, has pre-existing uncommitted WIP): `mario-showcase-20261103/mario-demo`,
  branch `feat/mario-decision-demo`

**Outcome — KEEP (steady-state), `auto`/reference demo path:**

- Clean campaign, 3/3 rounds, all six arms `Swapouts = 0`, decision sequences identical:
  serial **643 ms** vs pooled **590 ms** per decision → **+8.42 %**, CI **[+7.97, +8.94]**
  on decisions 2–12 (Opus's pre-declared judged set). `p_bits` bitwise-equal for every request.
- **Cost carried, not hidden:** decision 1 is **−33 %** (≈ +500 ms) because each worker thread
  independently fills the Tier-1 prefix; the all-12 CI consequently crosses zero
  ([−28.4, +8.9]). Removing the startup penalty (a shared or prewarmed fill) is the immediate
  next optimization target and would make the all-decisions gate pass too.
- Activation is opt-in: `--concurrency N` / `APXINF_DECIDE_CONCURRENCY` on the decide-server and
  `MARIO_CLEF_PARALLEL=1` on the bridge; memory cost is **N × model** (~5.7 GiB at N = 3).
- Clef auto-tier: one decision recorded; `f` = 11.5–55 % (above Opus's pursue bar) but
  `Swapouts`-contaminated → open direction, not a campaign (§7).

## 1. Measured starting point (clean window)

| Quantity | d3-lite | Source |
| --- | ---: | --- |
| Per-request steady median | **197.6 ms** | [d3-lite-demo-clean-remeasure.md](d3-lite-demo-clean-remeasure.md), same host, `Swapouts` delta 0 |
| Stage split (steady) | host_prep 13.0 / vision 59.3 / forward 124.2 / sync 0.1 ms | same |
| Per-decision (3× serial) median | **613.6 ms** | same (decisions 2–12: 612.2–615.6 ms) |
| First decision | 1470–4934 ms (worker load + first trace of the 3 compiled profiles) | same |
| Checkpoint / worker RSS | 1.6 GiB / ~1.9 GiB | same |

Clef's auto tier (GPU+ANE) also sends the same three requests serially through the same
bridge; its intra-request ANE wait is already 88.6 % masked by GPU, but the
request-**between** structure was never quantified.

## 2. Opus adjudication (step 1, mandatory)

Call 1 (brief 3035 B, `-e medium -t 240`) **timed out at the 240 s hard alarm** with no
output (exit 142; the upgraded script logs only completed calls, so this attempt does not
appear in `opus-calls.log`). Call 2 used a trimmed brief (1879 B, same effort, `-t 300`) and
**succeeded in 113 s**, producing 10 980 bytes (`opus-calls.log`:
`2026-11-06T10:32:26Z effort=medium prompt_bytes=1879 out_bytes=10980 dur=113s rc=0`).
Full text: `devlocal/notworth-audit/req-parallel/opus-design.md`.

**Opus's ranked recommendation:**

1. **(a) threaded server, one stream per thread** — expected **545–590 ms/decision**, rank 1
   *if* a measurement supports it. Each thread would own a model (shared weights were
   proposed, but `apxinf-mlx` `Array` is explicitly `!Send`, so weights cannot cross threads —
   see §4; the memory cost is therefore N × model).
2. **(c) bridge sends all three questions at once** — needed alongside (a); worthless alone
   (≈595–605 ms), because a single serial server only pipelines its stdin loop.
3. **(b) N worker processes** — rejected: same overlap limit as (a), plus cross-process
   scheduling and 3 × memory; swap risk on 16 GB.

**What it changed:** it converted an open "should we parallelize?" question into a
measurement-gated decision (measure GPU idle first), set the falsifiers (concurrent median
≥ 613.6 − 6 ms, or GPU intervals never intersect ⇒ stop), and specified the exact
measurement design (bitwise `p_bits`, bounded queue, prewarm before `ready`). It also
correctly flagged that Clef concurrency is only worth pursuing if the over-lappable fraction
`f ≥ 10 %` (and to drop it below 5 %).

**Opus's step-1 stop rule — re-adjudicated from the measurement (§3):** Opus wrote "if idle
time is under 10 ms per request, stop". The measured within-request GPU idle is ~5 ms, which
*would* fire that rule — but the rule uses the wrong quantity for this bridge. The 13 ms
`host_prep` of each request sits **outside** the request's GPU window and is paid serially
(three times per decision), so it is fully overlappable by construction once requests are in
flight. The binding term is therefore **39.4 ms/decision of serial host prep (6.4 %)**, not
the 5 ms within-window idle. The stop rule does not fire; §6 measures the result directly.

## 3. Opus step-1 measurement — one serial decision under xctrace

Recorder: `devlocal/notworth-audit/req-parallel/record_server_timeline.py`. The launched
process under `xctrace record --template "Metal System Trace" --instrument "Points of
Interest"` is the decide-server itself (`bash -c 'exec … < requests.jsonl > responses.jsonl'`),
with `APXINF_D3_SIGNPOST=1` and the three D3 gates pinned to their defaults. The checkpoint
was APFS-cloned out of `~/Documents` (TCC). Requests = the demo's own 3 request lines,
**two passes** (6 requests): pass 1 pays the compiled-profile traces, pass 2 is steady.

Window: `Swapouts` delta **0** (Swapins +10 943), wall 22.6 s, 70 command buffers.

| # | pass | question | request window (ms) | GPU busy (ms) | GPU idle (ms) | bubble |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 1 | enemy | 3877.8 | 467.8 | 3410.0 | 87.5 % |
| 1 | 1 | obstacle | 428.7 | 405.7 | 23.0 | 2.0 % |
| 2 | 1 | gap | 511.5 | 494.7 | 16.8 | 2.0 % |
| 3 | 2 | enemy | 182.9 | 177.7 | **5.2** | 0.0 % |
| 4 | 2 | obstacle | 174.9 | 170.1 | **4.9** | 0.0 % |
| 5 | 2 | gap | 217.7 | 212.7 | **5.1** | 0.0 % |

Server-side latencies for the same run (`server-responses.jsonl`): ready 705.6 ms;
pass 1 = 3892.3 / 440.8 / 524.8 ms; **pass 2 = 196.0 / 188.6 / 230.7 ms** (median 196.0 ms,
sum 615.3 ms — the established 613.6 ms decision).

**Reading:** steady requests are **GPU-bound (97.2–97.7 % busy, ~5 ms idle)**; there is no
within-request bubble to fill. But `host_prep` (13.0 / 13.6 / 12.8 ms) is **not** inside the
GPU window — the request signpost starts after preprocessing — and the serial bridge pays it
three times: 3 × ~13.1 ms = **39.4 ms/decision (6.4 %)** of purely serial host time, plus the
~15 ms of non-overlapped residual. Concurrent dispatch can hide two of the three host_prep
passes behind the first request's GPU window; the theoretical floor is the sum of GPU-busy
time (560.5 ms) plus one host_prep ≈ **574–590 ms/decision (≈ 3.4–6.4 %)**. That is above the
2 % KEEP gate, so the implementation proceeds — with the caveat that GPU queue parallelism
itself is still unverified (§4).

## 4. MLX threading facts

`devlocal/notworth-audit/req-parallel/mlx-stream-threading.md` (807 lines; checked against
upstream MLX v0.32.3 `64ea011`, the pinned SDK headers, and the shipped dylib):

| Claim | Status | Evidence |
| --- | --- | --- |
| `mx::default_stream(Device)` is thread-local | **VERIFIED** | `stream.h:30-31` "default stream of current thread"; `stream.cpp:18-27` `static thread_local`; `scheduler_tests.cpp:39-69` (4 threads → 4 distinct streams); matching TLS symbol in `libmlx.dylib` |
| `mx::compile` cache is thread-safe | **VERIFIED**, with caveats | `compile.cpp:415-418` thread-local `CompileCache` + `shared_mutex`; Metal library/kernel caches locked; maintainer: "compile cache is thread safe after 0.31.2" (ml-explore/mlx#2086). Caveats: compiled functions must be pure; no upstream test of *one* compiled object called simultaneously from multiple threads |
| `mx::new_stream` exists; no per-stream destroy | **VERIFIED** | `stream.h:39-46,63-64` — only bulk `clear_streams()`; the bridge's "no individual stream-destruction API" comment is accurate. Its stronger claim ("would retain global queues") is overstated: the registry grows, but Metal queues are owned per-encoder and reset on destruction |
| Two threads get **separate Metal command queues** | **VERIFIED** | each thread's `CommandEncoder` calls `newCommandQueue()` (`device.cpp:309-324`) |
| Actual GPU overlap of concurrent requests | **UNVERIFIED** upstream (no 0.32.3 overlap benchmark) | measured by this leg's campaign (§6) |
| Rust `Compiled` / `Array` / `Stream` are `Sync` / `Send` | **CONTRADICTED** | `Array` carries a compile_fail `!Send` doctest; `Stream` is `Rc`; `Compiled` holds `RefCell` — the safe Rust API does not expose MLX's C++-level concurrency |

**Implication:** the per-thread-model design is not a workaround — each thread's
`mx::default_stream` yields a distinct Stream and a **distinct Metal command queue**, so
independent compiled graphs are *eligible* to overlap, and the thread-safe compile cache
makes concurrent first-use traces safe. What remains unknown is whether the GPU actually
overlaps them; that is precisely the campaign's quantity.

## 5. Implementation

Two cooperating changes; the optimization only exists when both are active. Neither changes
single-request behaviour.

### 5.1 decide-server (this repo) — opt-in thread pool

`src/bin/apxinf-decide-server.rs`: `--concurrency N` (env `APXINF_DECIDE_CONCURRENCY`,
default 1). N = 1 stays on the original serial path (build → one `ready` →
read/answer/write loop), with the same response fields. N > 1 starts N worker threads;
**each builds and owns its own `Server`/`LoadedModel` entirely on that thread** — the model
is `!Send` (`Rc` handles, `RefCell` caches, thread-local MLX stream), so it is never shared
and never wrapped in a mutex (which would serialize the GPU work and defeat the point). The
main thread reads stdin lines into a bounded `VecDeque` (capacity N) guarded by a
`Mutex` + `Condvar` that is held only for queue bookkeeping; one writer emits responses
under a single stdout lock, in completion order. RAII outstanding-work accounting drains
in-flight requests on EOF; a broken stdout or a worker panic shuts the process down cleanly.

Every successful response gains a **`p_bits`** field — per question key in `answers`, the
8-hex-digit `f32::to_bits` of its reported probability — so two runs can be compared
bitwise. No existing field changes.

**Design note vs Opus:** Opus proposed sharing one set of weights across threads. That is
impossible with the current `apxinf-mlx` handle model (`Array` is explicitly `!Send`), so
the implementation pays **N × model memory** (~5.7 GiB resident at N = 3 versus ~1.9 GiB at
N = 1). This is the same total as N processes; the pool buys a single resident process and
one `ready` handshake, not memory.

### 5.2 bridge (demo checkout) — concurrent dispatch

`examples/mario_demo/clef_local.py` — untracked WIP in that checkout (pre-patch snapshot and
sha256 recorded in §9):

- `ClefLocalWorker` hardened: one dedicated reader thread is the **only** consumer of the
  child's stdout and demultiplexes replies by `id` into per-request queues; startup is
  serialized; writes are locked; a timeout or process failure fails every pending request.
  With one ask in flight the behaviour is equivalent to the old write-wait-read loop.
- `ClefLocalBackend.decide` gains a **`MARIO_CLEF_PARALLEL=1`** path: the three independent
  asks are dispatched through a `ThreadPoolExecutor` and joined in name order; `attempts`
  stays 3, and `latency_ms` becomes the **wall the player waits** (the serial path keeps
  the established sum-of-request-times convention, which equals its wall to sub-millisecond
  accuracy). The flag defaults off, so the demo's normal behaviour is unchanged.

Tests: `python -m unittest test_clef_local test_backend` in the demo env — **19/19 pass**,
including garbage-response, timeout and crash paths and `attempts == 3` for `separate`.

### 5.3 Gates exercised together

| Arm | decide-server | bridge | Effective requests in flight |
| --- | --- | --- | ---: |
| serial | `--concurrency 1` | `MARIO_CLEF_PARALLEL=0` | 1 |
| concurrent | `--concurrency 3` | `MARIO_CLEF_PARALLEL=1` | 3 |

## 6. Serial vs concurrent campaign

Runner: `devlocal/notworth-audit/req-parallel/run_concurrency_campaign.py` — 3 rounds, **literal
reverse arm order**, one exclusive dual-lock window per round, `vm_stat` before/after each arm,
foreign-process gate (`bench_local.py` / `run-bench.sh`) at every window start, per-arm timeout
360 s, an **exact decision-sequence gate** (every `(action, probabilities)` row compared with
float equality across all arms), and a pooled paired bootstrap CI (4000, seed 42) over clean
rounds only.

### Attempt 1 (19:18–19:20)

| round | arm | Swapouts | clean | decisions | median ms | steady ms (2–12) | sequence |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| 0 | serial | +0 | ✓ | 12 | 645.1 | 645.0 | identical |
| 0 | concurrent | **+0** | ✓ | 12 | **599.4** | **592.3** | identical |
| 1 (rev) | concurrent | +97 816 | ✗ | 12 | 1332.0 | 1274.9 | identical |
| 1 (rev) | serial | +0 | ✓ | 12 | 1094.6 | 1093.8 | identical |
| 2 | serial | +0 | ✓ | 12 | 1160.8 | 1157.0 | identical |
| 2 | concurrent | +28 952 | ✗ | 12 | 1083.0 | 1043.3 | identical |

- **clean rounds = [0] (1/3)** → the ≥2-clean rule is not met; `keep: false` → **INCONCLUSIVE**.
- The clean round-0 pair gives **median paired gain +6.743 %** (12 decisions, 58.3 % faster),
  CI [−48.013, +9.106]. The lower tail is the Tier-1 **miss** decisions: a chunk0 fill is paid
  on the concurrent arm's worker (up to 2381 ms vs 1515 ms serial) — the same bimodal
  hit/miss structure recorded in the clean re-measure §2. With only 12 decisions the bootstrap
  median can land in the miss mass; that is a power problem, not evidence against the effect.
- **All six arms produced the same decision sequence** (`all_decision_sequences_identical:
  true`), and the implementation's `p_bits` comparison was bitwise-equal for all six requests
  (§5.1).
- Rounds 1–2 were contaminated **only on the concurrent arm**; the serial arm was clean in all
  three rounds. The concurrent arm was itself clean in round 0 with the identical 5.7 GiB
  footprint, so the dirtiness is **temporal overlap** with the non-cooperating
  `sweep-d3-fine.sh` loop (which does not take the Metal locks — see §7), not the pool's
  memory.

### Retry (19:24–19:26, exclusive window)

The sweep session's final script (`sweep-d3-last.sh`, 5 episodes) ended at **19:24:35** with
swap rate 0; the retry launched immediately in that window in a fresh run-root.

| round | arm | Swapouts | clean | median ms | steady ms (2–12) | sequence |
| --- | --- | ---: | --- | ---: | ---: | --- |
| 0 | serial | +0 | ✓ | 642.5 | 642.4 | identical |
| 0 | concurrent | +0 | ✓ | 597.5 | 591.0 | identical |
| 1 (rev) | concurrent | +0 | ✓ | 589.1 | 588.4 | identical |
| 1 (rev) | serial | +0 | ✓ | 643.8 | 643.8 | identical |
| 2 | serial | +0 | ✓ | 645.4 | 645.4 | identical |
| 2 | concurrent | +0 | ✓ | 590.4 | 589.8 | identical |

**3/3 clean rounds** (all six arms `Swapouts = 0`), and every arm produced the same decision
sequence.

| Judged set | median gain | CI 95 % | faster | Per-round medians |
| --- | ---: | ---: | ---: | --- |
| **steady decisions 2–12 (Opus's pre-declared set)** | **+8.423 %** | **[+7.973, +8.940]** | 69.7 % | +8.008 / +8.423 / +8.434 |
| all 12 (P4/Clef reporting convention) | +8.336 % | [−28.382, +8.875] | 63.9 % | +7.034 / +8.319 / +8.405 |
| decision 1 only | **−33.2 %** | — | — | serial 1507–1517 ms vs concurrent 1983–2019 ms |

**Reading:**

- The steady-state effect is large and extremely consistent across rounds (8.0 / 8.4 / 8.4 %,
  tight CI), and it matches the mechanism measured in §3: two of the three serial
  `host_prep` passes (~13 ms each) hide behind the first request's GPU window, and MLX's
  thread-local default streams give the three requests **separate Metal command queues** (§4)
  so their GPU work is eligible to interleave.
- **Decision 1 is 33 % slower** because each worker thread owns its own prefix cache and
  independently performs the chunk0 fill; the serial arm fills once and hits thereafter. This
  is a one-time per-process startup cost, not a steady-state cost.
- The all-12 CI crosses zero for exactly that reason: ~1/3 of the 36 decisions are the three
  decision-1 rows (and the few cache misses), and bootstrap resamples that over-represent
  them drag the lower tail to −28 %. The median itself never goes negative.
- **Verdict: KEEP on the pre-declared steady-state set** (`median ≥ 2 %`, `CI lower > 0`,
  `3/3 clean`, exact decision-sequence and `p_bits` parity). The startup penalty is a material
  cost reported alongside it, not hidden; it also sets the next optimization target (a shared
  or prewarmed prefix fill would remove it and make the all-12 CI pass as well).

## 7. Clef auto-tier decision timeline

One auto-tier demo decision (the same 3 demo request lines: enemy/obstacle/gap crops), recorded
with the signpost-enabled Clef binary from the showcase lineage
(`signpost-timeline-candidates-…/target/release/apxinf-decide-server`), `xctrace record
--template "Metal System Trace" --instrument "Points of Interest"`, `APXINF_MODEL_VARIANT=auto`,
columns 6656, both locks held for warm+record. Full report and PNG:
`devlocal/notworth-audit/req-parallel/clef-timeline/{report.md,clef-decision-timeline.png}`
(1730 × 444; three request bars + host stages + `ane_wait`/`ane_predict` + Metal lanes).

**Verdict: CONTAMINATED — reference only.** `record_rc=0`, 3/3 responses, 0 foreign-process
samples, but `Swapouts` delta **+62,024 pages** ≠ 0; every timing below is labelled
`Swapouts-contaminated (reference only)`.

| Quantity | Value |
| --- | --- |
| Decision wall (end req3 − start req1) | **10,068.0 ms** (windows 4,688.7 / 2,322.3 / 3,056.8 ms; server latencies match <0.1 ms) |
| Inter-request gaps (server side) | **0.089 ms, 0.149 ms** — immediate serial dispatch |
| GPU busy union (hardware, Metal table) | 4,520.7 ms (44.9 % of wall) → idle 5,547.3 ms → **f = 0.551** |
| GPU ∪ ANE-predict (ANE = signpost SW proxy) | idle 3,996.9 ms → **f = 0.397** |
| GPU ∪ engine-active `ane_wait`∪`ane_predict` (SW proxy) | idle 1,161.5 ms → **f = 0.115** |
| Per-request exact partition (both / GPU-only / engine-only / none) | req1 1706.9 / 176.1 / 2536.6 / 269.0 (bubble 5.7 %); req2 1085.1 / 187.4 / 631.2 / 418.5 (18.0 %); req3 1138.9 / 226.2 / 1218.1 / 473.6 (15.5 %) |

**Reading against Opus's Clef rule** (pursue only if `f ≥ 10 %`, drop below 5 %): the measured
`f` is **11.5 %–55 %** depending on the busy definition — above the pursue bar on all three,
but the most conservative proxy (GPU ∪ engine-active) clears it by only 1.5 pp, the take is
Swapouts-contaminated, and it is a single decision. **Clef request-level concurrency stays an
open direction, not a campaign**: the next step would be a clean re-take (and, if the standing
rules ever relax the instrument ban, hardware ANE occupancy — this trace has no ANE/Neural
Engine schema, so all ANE figures are os_signpost software wall-clock proxies).

**Root cause found and remediated:** two earlier attempts died with `Clef runtime placement is
unknown` (xctrace rc=54). Locked loadtests isolated a **stale ANE Core ML cache** bound to a
different binary state — the exact failure mode the line's HANDOVER warns about. Deleting the
task-specific cache and letting the protocol warm recompile it made the recording succeed
(recorded load 23.0 s).

## 7.1 Before / after operator timeline (this optimization)

![d3-lite request-level parallelism — before/after operator timeline](d3-request-parallelism-timeline.png)

[`doc/d3-request-parallelism-timeline.png`](d3-request-parallelism-timeline.png) (1856 × 1280),
same recorder and same 6 file-fed requests on both arms: `before` = serial
(`--concurrency 1`, recording made before the implementation), `after` = pooled
(`--concurrency 3`). Both recordings are `Swapouts`-clean (0 / 0); the panel shows the correct
structural change — three request bars strictly sequential before, overlapping after.

**Caveat carried in the figure and here:** `xctrace`'s Metal System Trace inflates the pooled
arm (three in-flight command queues) to **~1.1 s/request versus 197 ms in the clean campaign**,
so the panel's metric table is structural evidence, **not** a latency verdict: the steady-state
regression it appears to show is tracer overhead on concurrent queues, not the optimization.
The campaign (§6, 3/3 clean, no tracer) is the verdict: serial 643 ms, concurrent 590 ms,
+8.42 % [7.97, 8.94] on decisions 2–12.

## 8. Verification

- **Opus:** call 1 (3035 B brief, `-e medium -t 240`) timed out at 240 s (exit 142, no output;
  the upgraded script logs only completed calls). Call 2 (1879 B, same effort, `-t 300`)
  succeeded in 113 s → 10 980 B; `~/apxinf-runtime/opus-calls.log` has the
  `2026-11-06T10:32:26Z … dur=113s rc=0` entry.
- **MLX research:** `devlocal/notworth-audit/req-parallel/mlx-stream-threading.md` —
  `default_stream` thread-local VERIFIED; compile cache thread-safe VERIFIED (with caveats);
  no per-stream destroy VERIFIED; separate per-thread command queues VERIFIED; GPU overlap
  UNVERIFIED upstream and measured here instead.
- **Serial decision timeline (step 1):** `recordings/d3-decision-serial/record_meta.json` —
  `Swapouts` delta **0**; per-request metrics in `per-request.json` (steady GPU busy
  177.7 / 170.1 / 212.7 ms of 182.9 / 174.9 / 217.7 ms windows).
- **Pooled decision timeline (§7.1):** `recordings/d3-decision-concurrent/record_meta.json` —
  `Swapouts` delta **0**, `export-all/metrics.json` present; tracer-inflation caveat in §7.1.
- **Campaign (§6):** `campaigns/concurrency-retry/summary.json` — 3/3 clean rounds, all six
  arms `Swapouts = 0`, `all_decision_sequences_identical: true`, steady median **+8.423 %**,
  CI **[+7.973, +8.940]**, `keep: true` on the pre-declared judged set. Attempt 1 (contended,
  1/3 clean) preserved in `campaigns/concurrency/`.
- **Implementation parity:** `cargo check --locked --features mlx` + `cargo build --release …`
  exit 0; serial vs pooled `p_bits` bitwise-equal for all six requests; serial answers match the
  pre-change smoke baseline exactly; `rustfmt --check` on the changed file clean; edge cases
  (`--concurrency 0`, bad checkpoint at N=3, env-only activation) behave as specified.
- **Bridge:** `python -m unittest test_clef_local test_backend` (demo env) — **19/19 pass**,
  incl. garbage-response, timeout and crash paths and `attempts == 3`.
- **Clef timeline (§7):** `record_rc=0`, 3/3 responses, **`Swapouts` delta +62,024 →
  CONTAMINATED (reference only)**; root cause (stale ANE cache) diagnosed by locked A/B
  loadtests and remediated; PNG produced and labelled.
- No product code outside `src/bin/apxinf-decide-server.rs`; no other crate touched; no new
  dependencies.

## 9. Artifacts

**Committed:** [`src/bin/apxinf-decide-server.rs`](../src/bin/apxinf-decide-server.rs)
(implementation), this report, and
[`doc/d3-request-parallelism-timeline.png`](d3-request-parallelism-timeline.png) (the
before/after panel).

Private evidence (ignored): `devlocal/notworth-audit/req-parallel/` —
`opus-design.md`, `mlx-stream-threading.md`, `record_server_timeline.py` +
`analyze_recorded_requests.py` + `recordings/{d3-decision-serial,d3-decision-concurrent}/`
(trace + export-all + per-request exports + `record_meta.json`), `timeline-render/` (renderer
inputs), `render_req_parallel.py` (patched panel renderer), `campaigns/{concurrency,
concurrency-retry}/` (`summary.json` + all arm receipts), `clef-timeline/` (report, PNG,
`recording-result.json`, `export-req*`, `export-decision/`, `metrics.json`, `op.trace`,
driver scripts), `bridge/clef_local.orig.py` (pre-patch snapshot of the demo worker),
`requests.jsonl`, `server-responses-c{1,3}.jsonl`, `staged/` (TCC-cloned d3 checkpoint).

The demo checkout's `clef_local.py` is modified in place (it is untracked WIP in that checkout);
the original is preserved at `bridge/clef_local.orig.py`, its sha256
(`125c0dae…9fac3c`) is recorded, and the change is reproducible from this report. The Clef
auto-tier cache used by the timeline lives at `/tmp/clef-ane-cache-notworth-reqpar` (task-specific,
regenerable).
