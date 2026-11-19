# QUEUE #6 — batched decode throughput（batch 1/2/4/8 总 tok/s）

Source: `QUEUE.md` v2 item 6 (campaign `lfm3-dig`, base `feat/lfm3-metal`).
Baseline: PR #127 pipelined decode, `apxinf-pipe` 36.07 / 36.40 ms/token
(sync `apxinf-sync` 37.49 / 37.79), M4 (`haiyan-mac-mini`), `feat/lfm3-metal`
tip `1ed3f2f3` carried here at `origin/feat/lfm3-metal`. PR target:
`feat/lfm3-metal`.

The single-request latency roofline is sealed: 3.4576 GB/token ÷ 100.6
GB/s = **34.36 ms/token** floor, observed 36.07/36.40 ⇒ 95%/94% of
streaming-bandwidth ceiling. The remaining question is **batch throughput**:
can one request's idle decode cycles carry other requests' weight traffic?

This is the first ticket since #128 that deliberately changes the device
data shape, so it inherits QUEUE #5's frozen lesson verbatim:

> **代数等价 ≠ token 等价；任何结构改动先跑 token 门，再谈速度。**

The hard gate is therefore **not** a timing property. It is:

> **批内每个请求的贪心输出必须与 batch=1 时逐 token 相同。**

## Quantification before pre-registration (QUEUE #6 / #7 "先量化" discipline)

| candidate | quantitative prior | position |
| --- | --- | --- |
| decode batching (batch 1/2/4/8) | 3.4576 GB/token ÷ 100.6 GB/s = 34.36 ms/token single-request floor; weights are 100% shared ⇒ upper bound on batch win ≈ B× before attention/conv state traffic grows | **this ticket** |
| tokeniser / detokeniser host cost | already ~0.004% (0.34/0.39 ms gateway round trip) | not first — no quantitative residual |
| load/startup, gateway RTT | known small, does not touch the generation path | queue item 7, later |
| launch-gap drain | QUEUE #4 **INVALID** (empty banner arm + A/A 6.95%) ⇒ prior remains unmeasured; not re-opened here | not this ticket |

**Architecture fact discovered before writing this gate** (verified by reading
`apxinf-serving`, not assumed):

- `crates/apxinf-serving/src/gateway.rs:329` advertises
  `"batching":false`; `supervisor.rs` capabilities carry
  `max_active_sequences:1` (`native_worker.rs:614`); the coordinator's
  `execute()` (supervisor.rs:2548) pops **one** job, sends one `submit`
  frame, and waits for its terminal frame before the next `queue.recv()`.
- `Lfm3Model` (model.rs:1077 `DeviceDecode::decode`) rejects any device
  token tensor whose shape is not exactly `[1]`, and `build_graph` reshapes
  embeddings to `[1, len, hidden]`; KV rows are `[1, kv_heads, capacity,
  head_dim]`, conv windows `[1, hidden, kernel-1]`. There is **no batched
  execution entry today**.

⇒ This ticket's pre-registered decision is therefore not "measure an
existing switch". It is a **new implementation** whose first obligation is
to prove, before any device run, that a batched forward can be added
**without** disturbing the single-request path. If that cannot be shown by
construction (shape/vectorised-accumulation audit), §5.2 forces an
INVALID verdict *before* any window, and we record "no batched entry was
safe to add" rather than ship a token-divergent batch.

Status: **pre-registered (this commit, before any device run).** The
decision rule and thresholds below are frozen from here; amendments are
dated append-only at the end.

---

## 预登记 (pre-registration)

### 1. Arms

| arm | change | switch |
| --- | --- | --- |
| `apxinf-pipe-b1` | none; current pipelined decode, one request (the batch=1 oracle) | unset |
| `apxinf-pipe-bB` | same binary; B = 2/4/8 requests admitted and decoded together, per-request KV/conv state kept in a leading batch axis, greedy selection per row | `APXINF_LFM3_BATCH=B` |

Same binary, same weights, same per-kernel code for the single-request
path. The batch arm is armed only when `B > 1`; `B = 1` (or unset) is the
unchanged path (empty-arm gate, §5.4).

Mechanism (frozen here):

- `Weights::project` on `[rows, hidden] × [hidden, vocab]` is the only
  matmul whose input is already multi-row-capable: for batch decode the
  rows are `[B, hidden]` stacked from `B` distinct requests, so the
  **identical** `[B, vocab]` MLX matmul runs with an unchanged weight
  shape. `KKernel`'s custom lm_head is therefore *not* armed on the batch
  path (it requires `[1,1,K]`; §11).
- Conv (`conv_step`, model.rs:514) and attention (`attention_step`,
  model.rs:681) currently hardcode the single-row geometry
  (`[1, 1, heads, dim]`, `window` `[1, L, C]`, KV row `[1, kv_heads, …]`).
  Batched variants must **either** (a) keep the existing kernels and
  iterate B times (weights shared once, launches ×B) — a pure orchestration
  change that cannot alter any kernel's arithmetic — **or** (b) add
  batched kernels whose per-row accumulation order is audited
  shape-preserving. This ticket **starts with (a)**: it is the only option
  that QUEUE #5's rule certifies *before* measurement.
- Supervision: `PendingQueue::recv` (pending_queue.rs:104) currently pops
  one job; a batch dispatcher pops up to B jobs whose prompts are ready
  and submits one batched frame (or B frames dispatched in one step if
  (a) is chosen), then demultiplexes per-request token events to their
  existing `Ticket.output` channels. `max_active_sequences` / ready
  capabilities and the banner gain a `batch=B` field from the same env.

### 2. Cells / order

| cell | meaning |
| --- | --- |
| C-B1 | batch=1 total tok/s (oracle, no batch arm) |
| C-B2 / C-B4 / C-B8 | total tok/s, prefill tok/s, per-request p50/p99 inter-token latency at batch 2 / 4 / 8 |

- **Order** `A B B A A B` where A = B1 and B = the batch size under test,
  **three segments** each, seeds 20261119 / 20261120 / 20261121;
  segment medians, then cell median. Same discipline as #127 / QUEUE #4.
- A/A: run two identical B1 segments *and* two identical B=B segments in
  the same window; worst paired spread **≤ 5%** or the window is void
  (RUNNER §4). This ticket's A/A is its own — **#121's S1 short-generation
  cell (5.02%) is not citable** (lead note).
- Engine-reported timings (`--engine-timings`); client wall-clock is void
  (RUNNER §3, #120 SSE buffering).
- Window ≤ 45 min on M4 under both Metal locks (waited, never stolen),
  `pmset displaysleepnow` + `caffeinate -si`.

### 3. Mechanism accounting (what the win must explain)

Per decode step, bytes split into **shared** and **per-request**:

```
shared:  every Linear/RMSNorm weight ≈ 3.4576 GB − per-row KV/conv I/O
per-req: KV slice read + conv window read + logits row write/read
```

- Batch win upper bound: shared weight bytes amortised ×B ⇒
  speedup ≤ B (perfect), with a realistic roofline at
  `shared_bytes / (B · shared_bytes/B + per_req_bytes · B)`
  ⇒ **compute the expected GB/s and ms/token for B = 2/4/8 in §Results**;
  if the measured speedup is far below this roofline, the gap is either
  per-request traffic (report it) or launch overhead (机制未知 if not
  observable — `StreamCounters` has no launch counter, `xctrace`
  unavailable; QUEUE #4 §8 honesty note applies).
- Explicit byte budget for the 16 GB machine: KV rows at ctx 4096
  = `11 attention layers × 2 × 4 heads × 4096 × 128 × 2 B ≈ 92 MB` per
  request; conv windows `13 × 2048 × 6 × 2 B ≈ 0.32 MB`; logits
  `128000 × 4 B = 0.51 MB` per active row. Report measured
  `peak_memory_bytes` (worker `terminal` event) for B = 1/2/4/8 at ctx
  1024 and 4096; **B is capped at the largest value whose peak stays
  within the declared memory budget** (`--memory-limit-bytes`); otherwise
  the cell is INCONCLUSIVE with the measured peak, not extrapolated.

### 4. Falsifier

- **No value**: batch > 1 total tok/s improvement **< 1.5×** versus B1
  (measured, same window) ⇒ FALSIFIED.
- **Token gate**: any request in any batch differs from its B1 greedy
  sequence ⇒ INVALID (§5.1) — hard, not a timing judgement.
- **A/A > 5%** ⇒ window void (INVALID for this ticket, re-run allowed
  only with a dated amendment).
- **Effect proof**: banner must show `batch=B` on every armed segment;
  absent ⇒ INVALID (empty-arm trap, same failure mode as QUEUE #4).

### 5. Verdict rules

| verdict | condition |
| --- | --- |
| **WIN** | batch arm **runs** (banner proves it), token gate **passes**, A/A ≤ 5%, and total tok/s at some B ≥ 2 is ≥ 1.5× B1 with the mechanism arithmetic reconciling within 15% of the shared/ per-request byte model |
| **PARTIAL** | token gate passes and speedup is ≥ 1.2× but < 1.5× at some B, or ≥ 1.5× at one B only (not monotone), with A/A clean |
| **PARITY** | token gate passes, A/A clean, but speedup < 1.2× at every B (or B=1 is the only safe B — memory ceiling, measured peak reported) |
| **FALSIFIED** | token gate passes and A/A clean, but speedup < 1.5× at every B **and** the byte model predicted ≥ 1.5× (mechanism wrong: per-request traffic or launches dominate — name which, or write 机制未知) |
| **INCONCLUSIVE** | batched forward could not be proven shape/numerics-preserving **before** any run (§5.2 audit fails), or memory ceiling forces B=1 on the 16 GB machine (measured peak reported), or A/A void once and the re-run was not authorised |
| **INVALID** | any hard gate in §5 fires (token mismatch, shape drift, effect proof missing, A/A > 5%) |

### 6. Hard gates (any one ⇒ INVALID)

1. **Token consistency (the queue's stated hard gate).** For every batch
   size B and every request in the batch, the greedy output equals that
   request's batch=1 output **token for token**. Harness: B prompts (the
   same 4 pseudo-prompts, repeated/truncated as needed for B), 128 output
   tokens each; run once at B1 to get `expected[i]`, once at B=B and
   assert `got[i] == expected[i]` per request. **Entry and exit both
   clear `APXINF_LFM3_BATCH`** so a prior panic cannot poison the oracle
   arm (the empty-arm trap).
2. **Shape / accumulation audit (QUEUE #5 rule, before the device).**
   Any batched kernel whose per-element reduction order over K differs
   from the single-request kernel's order, any padding/bucketing/wider
   matmul, any change to `build_graph`'s op sequence for the **B=1**
   path, or any change to `Embedding`/`head` weight shapes ⇒ INVALID.
   The B=1 path must be **byte-identical in graph construction** to the
   pre-ticket tree (assert with `trace_callbacks` / `eval_calls`
   deltas equal to #127's baseline values).
3. **Effect proof**: banner `batch=<B>` present on every armed segment
   (same env the dispatcher reads); missing ⇒ INVALID.
4. **A/A ≤ 5%** for this ticket's own segments (B1 and B=B).

### 7. Default rule

This ticket is a **throughput measurement with a new execution entry**.
The result (WIN/PARTIAL/PARITY/FALSIFIED/INCONCLUSIVE/INVALID) is
recorded in §Results and in the PR. A WIN authorises keeping the batched
path **behind the switch** (`APXINF_LFM3_BATCH`, default 1) — it is a
capacity feature, not a latency feature, and the switch stays explicit
because it changes the device data shape. A non-WIN verdict **reverts the
batched path** (dispatcher + batched kernels + banner field) and keeps
only this record, exactly like QUEUE #4 §7. No silent default flip.

### 8. Gate-0 census (before any timing window)

Same locked M4 session, one warm request per arm:

- **B1 regression census**: `StreamCounters` deltas and stage
  decomposition on the B1 arm must match #127's baseline within 5%
  (this ticket's own A/A); any B1 drift ⇒ §5.2 INVALID *before* the
  batch arm runs.
- **Batch arm**: banner `batch=B`; `eval_calls` per step expected
  **×1 per request** for orchestration-style (a) or **×1 total** for a
  fused batched forward (b) — either is admissible *only if* the token
  gate passes; report which. `trace_callbacks` must be equal to the
  graph's op count (no new compile). `uploads/downloads` unchanged.
- **Kernel launches**: `StreamCounters` does not expose them; `xctrace`
  unavailable ⇒ **机制未知**; the window still runs (QUEUE #4 §8).
- **Per-kernel GB/s**: not observable ⇒ **机制未知**; only request-total
  tok/s and the byte model are used for §5.

### 9. Tests (before device runs)

Metal-locked, `#[ignore]`, real BF16 checkpoint, `--test-threads=1`,
`APXINF_LFM3_TEST_MODEL=~/lfm3-spec/weights-local/LFM3-1.7B` (same
harness as T1/K3/P1/E1):

- **B1 token gate (hard)** — the §6.1 assertion, for B = 2, 4, 8.
- **B2 empty-arm** — unset ⇒ dispatcher never groups >1 job, banner
  `batch=1`, `Lfm3Model` B1 path untouched (field assert).
- **B3 continuation / state isolation** — two requests in one batch,
  after 64 tokens each, per-request KV/conv windows must equal the
  corresponding B1 runs' bytes; a 16-token continuation of each must
  match token-for-token.
- **B4 negative control** — batched forward with a padded row count
  (B′ ≠ B) or a mixed dtype row must `Err` loudly (shape gate
  load-bearing, K4 style).
- **B5 memory ceiling** — at ctx 4096, measured `peak_memory_bytes` for
  B = 1/2/4/8; assert the reported peak is monotone in B and within the
  declared budget; if B=4 already exceeds it, the test asserts the
  dispatcher *refuses* B=4 with a capacity error rather than OOMing.
- The four `cpu_pipeline_*` tests and the existing T1/T2/T3 suite must
  pass unchanged (the schedule driver is not modified).

### 10. Window discipline

M4 only, both Metal locks (wait, never steal), `SDKROOT=
/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`, `MLX_ROOT=
~/mlx-sdk`, `DYLD_LIBRARY_PATH` includes mlx lib; `pmset displaysleepnow`
+ `caffeinate -si` before each segment; engine timings only; swap /
thermal / load recorded before and after; M1 is not measured; nothing is
extrapolated across hosts; cross-window absolute numbers are not
comparable (#127 §11 caveat 5) — the verdict rests only on within-window
paired medians.

### 11. Implementation limits (what this ticket must NOT do)

- **No B=1 path change**: the single-request graph construction, kernel
  choices, schedule and banner fields other than the additive `batch`
  field are frozen (QUEUE #5: shape-preserving is necessary but not
  sufficient — so also *graph*-preserving for B=1).
- **No custom lm_head / silu kernels on the batch path** (QUEUE #2
  INVALID; `KKernel` requires single-row shapes; the batch path uses
  the stock MLX matmul on stacked rows).
- **No KV/conv cache sharing between requests** — each request owns its
  own state; the batch only shares weight reads.
- **No quantisation / tiling / compiled-graph change** — this ticket
  changes the *number* of rows per step, not the arithmetic of any row.
- **No default-on**: `APXINF_LFM3_BATCH` defaults to 1; the dispatcher
  groups only when the switch is set and the queue has ≥2 ready jobs.

---

## Relation to QUEUE #1–#5, #128, #127, #115

- **#128 (merged GEMV)** — INVALID because concatenating weights
  changed matmul tiling ⇒ accumulation order ⇒ token. §6.2 of this
  ticket is that lesson operationalised: batched rows must not change
  any *per-row* reduction order, and B=1 must be graph-identical.
- **QUEUE #1 (compiled decode)** — INCONCLUSIVE; no window, no census.
  Not re-opened; this ticket's batch dispatcher is unrelated to step
  compilation.
- **QUEUE #2 (custom kernels)** — INVALID at K3 token gate; the batch
  path therefore uses stock MLX matmul (§11).
- **QUEUE #3 (prefill graph cache)** — INVALID at P1 token gate; the
  stale-graph lesson (cached logits not rebound to new ids) is directly
  relevant: a batched prefill must bind *per-request* ids, not reuse one
  request's graph. §6.1 covers it.
- **QUEUE #4 (eval/sync split)** — INVALID (empty banner arm + A/A
  6.95%); §6.3 copies its effect-proof requirement, and §8 copies its
  mechanisms-unknown honesty note.
- **QUEUE #5 (structural-reorg negative closeout)** — the governing
  rule for this ticket; §6.2 is its operational form. QUEUE #5
  explicitly left batch throughput (#6) as the *only* remaining
  structure-changing direction, because its gate is per-request token
  equality rather than per-kernel bit equality.
- **#127 (pipelined decode)** — the B1 oracle and the latency baseline
  whose byte model (3.4576 GB/token) this ticket amortises.
- **#115 (compile/bucketing)** — closed; this ticket does not compile
  or bucket (§11).

---

## Results

*(appended after the window; verdict + mechanism arithmetic + receipts)*
