# QUEUE #6 — pre-registration review receipt（round-82，2026-11-22）

Scope: the one minimal, no-M4-lock step frozen at `STATUS.md` §626 — review
the QUEUE #6 batched-throughput pre-registration for drift and for the
missing "each request vs batch=1 token-identical" hard-gate harness. Offline
only; no lock, no device, no new gate, no implementation.

Base: `feat/lfm3-metal` tip `11a8fa7d`. Receipt branch: `agent/lfm3-dig`.
The pre-registration itself is unchanged (`doc/lfm3-batch-throughput.md`,
verdict PARITY / NOT-WORTH, Lead 09:31 ruling of 2026-11-20; unchanged vs
receipt branch `agent/lfm3-dig-batch` — `git diff` is empty).

## Findings

### F1 — Frozen gates are still intact (PASS)

| gate | frozen location | status on `feat/lfm3-metal` |
| --- | --- | --- |
| §6.1 token consistency (every request in the batch token-for-token equals its batch=1 greedy output) | `doc/lfm3-batch-throughput.md` §6.1 | intact, no amendment weakened it |
| §6.2 shape / accumulation audit (B=1 graph byte-identical) | §6.2 | intact |
| §6.3 effect proof (`batch=<B>` banner on every armed segment) | §6.3 | intact; production `/readyz` is fail-closed: `gateway.rs:329` hardcodes `"batching":false`, `native_worker.rs:628` `max_active_sequences:1` — the receipt branch's advertisement is not in base |
| §6.4 A/A ≤ 5% (own segments) | §6.4 | intact |
| §4 falsifier (≥1.5× total tok/s at some B, else FALSIFIED/PARITY per §5) | §4 | intact |
| §7 default rule (non-WIN ⇒ revert; switch stays explicit) | §7 | intact; dispatcher never landed in base, so nothing to revert |

Base tip and origin agree (`11a8fa7d`, `git fetch` + `rev-parse`), working
tree clean.

### F2 — The batch token-gate harness does NOT exist in the tree (OPEN, no action this round)

§9 of the pre-registration specifies the harness as a Metal-locked `#[ignore]`
Rust test ("B1 token gate (hard)"), but no such test exists on base or on the
receipt branch:

- `crates/apxinf-model/src/metal_lfm3/model.rs:1594` `t1_pipelined_greedy_matches_sync_token_for_token`
  is the **schedule-vs-schedule** oracle (4 pseudo-prompts × 128 tok, sync vs
  pipelined). It is the correct *pattern* to copy, but it has no batch dimension:
  `Lfm3Model`'s `DeviceDecode::decode` still rejects any device token tensor
  whose shape is not exactly `[1]` on base.
- No test or script in-tree references `APXINF_LFM3_BATCH` against a model
  (only `native_worker.rs:448` env parsing + `native_worker.rs:1496` banner
  field on the receipt branch).
- The only batch tests shipped were the serial dispatcher CPU tests
  (`crates/apxinf-serving/src/supervisor/lifecycle_regressions.rs:1065`
  `batch_two_pops_two_pending_jobs…`), which assert FIFO admission and
  completion, **not** token equality — consistent with Amendment 1 §A (mechanism
  (a) ships a serial pop window with per-request weight re-reads ⇒ 1.000× by
  construction).

⇒ §9's hard-gate test is a **placeholder** relative to the delivered tree. It
cannot be filled in without a batched forward entry (`Lfm3Model` B>1 decode),
which Amendment 1 §D option (1) never shipped — and Lead 09:31 closed #6 as
NOT-WORTH, forbidding that implementation. Consequence for the frozen record:
the token gate was **correctly not run** (Amendment 1 §B: no window on the
shipped tree can produce a WIN; §6.1 not run by design), but a future ticket
that re-opens batch execution must land its B1-oracle harness in the same PR
as the batched entry, not as a later step — otherwise §6.1 is unfalsifiable.

### F3 — Numbers in the pre-registration are stale relative to shipped assets (NIT, no correction)

The arithmetic receipt uses `128000 × 2048 × 2 = 0.524 GB` for `lm_head`
(§A) and `3.4576 GB` BF16 weights — both frozen Lead-cited values from the
original BF16 snapshot. They are internally consistent with §3's byte budget
and were the basis of the NOT-WORTH ruling, so they are **not amended**. Note
for the record: the current M4 assets are BF16 `135922ca…` and 4-bit
`96b021b3…` (`doc/lfm3-campaign-summary.md` §2); QUEUE #6 was never re-run on
4-bit and its verdict does not cover it.

## Verdict

**PARITY / NOT-WORTH stands (no change).** The pre-registration review
produces no new verdict tier and no amendment: the frozen gates are intact,
the fail-closed advertisement is correctly absent from base, and the one
implementation gap (F2, missing token-gate harness) is load-bearing only for
a direction Lead has closed. No device, no lock, no source change.

**Receipts:** this file (commit on `agent/lfm3-dig`); `doc/lfm3-batch-throughput.md`
§Results + Amendment 1/2; receipt branch `agent/lfm3-dig-batch` `119091b6..edd804d8`;
`cargo` tests not re-run this round (docs-only change).

**Next minimal step (no M4 lock):** either (a) QUEUE #8 quality档 GSM8K /
suite 冻结口径的离线复核（Lead §626 二选一中的另一项）, or (b) a Lead decision
on `lfm3-ane-load` (pid 9884/24582,双 Metal 锁连续 21+ 轮 HELD) — P2 W1
remains blocked until then.
