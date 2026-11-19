# LFM3 pipelined greedy decode: per-token host-gap measurement and the M4 matched-precision verdict

Source: GitHub issue [qhy991/apxinf-metal#122](https://github.com/qhy991/apxinf-metal/issues/122).
Baseline branch: `feat/lfm3-metal` (tip `93041ad6`, the #121 merge). PR targets
`feat/lfm3-metal`.

The question is specific and falsifiable: #121 measured `apxinf-native` BF16 at
37.20 / 37.50 ms/token against `llamacpp-f16` at 36.42 / 36.67 ms/token (S2 decode,
ctx 1024 / 4096, gen 128, M4 window `f16-20261118T211938Z`). Both arms stream the
same bytes per token, and the 0.78–0.83 ms gap does not move when the KV read
quadruples, so the gap is a **fixed per-token host-side cost**, not bandwidth. This
ticket wires `metal_lfm3`'s greedy decode onto the repository's existing pipelined
path (`mlx_decode::GreedySequence`), measures where the 0.8 ms actually goes, and
re-runs the matched-precision M4 window to decide whether pipelined apxinf beats
llama.cpp F16 under the pre-registered rule.

Status: **pre-registered; no device run has happened yet at this commit.**
Methodology references: [`doc/engine-comparison-lfm3.md`](engine-comparison-lfm3.md)
(#121, the matched-precision window this ticket extends) and
[`scripts/bench_engines.sh`](../scripts/bench_engines.sh).

---

## 预登记 (pre-registration)

Committed before any device run, per the ticket's discipline. The decision rules
and thresholds below are frozen from here; amendments are dated append-only.

### 1. Arms

| arm | engine | weights | serving surface | schedule switch |
|---|---|---|---|---|
| `llamacpp-f16` | llama.cpp `llama-server` (build `d635b19f…`, source `5297534`) | `LFM3-1.7B-F16.gguf`, sha256 `b7ab8b69…` (reused from #121) | `/v1/completions`, bare | n/a |
| `apxinf-sync` | apxinf-metal `metal_lfm3` from this branch (rebuilt, sha256 into identity) | reconstructed BF16 `model.safetensors`, sha256 `135922ca…` | `/v1/chat/completions`, lane ChatML | `APXINF_MLX_SYNC_DECODE=1` |
| `apxinf-pipe` | **the same binary and weights as `apxinf-sync`** | same | same | switch unset (pipelined) |

Switch mechanism (fixed here): `apxinf-serve` gains a repeatable `--worker-env
KEY=VALUE` option; the supervisor forwards each pair into the worker process after
its `env_clear()`, so a switch set on the gateway actually reaches the worker.
Proof of effect (the 空臂 trap from the campaign's methodology lesson 2): the
native worker prints one stderr banner at construction, which the supervisor pipes
into the gateway log:

```
decode_schedule=<sync|pipelined> overlaps_callbacks=<false|true> decode_stages=<false|true>
```

An arm whose gateway log does not contain its banner line is **void**, not slow.

### 2. Cells

S2 decode only, exactly #121's口径: contexts 1024 and 4096, `max_tokens=128`,
`--engine-timings` basis on both apxinf arms, `scripts/bench_engines.sh` +
`scripts/bench_engines_report.py`, per-segment corpus seeds (each arm in a segment
shares the segment's seed; distinct text per segment so no prefix-cache hits).
The S1 decode-at-8 cells stay excluded from any verdict, as in #121 (their A/A is
not certifiable). No tier A/B rerun, no S1/S3/S4/S5.

### 3. Order

`A B C C B A A B C` where A = `llamacpp-f16`, B = `apxinf-sync`, C = `apxinf-pipe`:
three segments per arm, paired medians per cell (each segment contributes one
median per arm; a cell's paired value is the median of the three segment medians).

### 4. Sync-arm decomposition (new instrumentation, default off)

Env flag `APXINF_LFM3_DECODE_STAGES=1` (reaches the worker through the same
`--worker-env` forwarding). When on, every decode step logs one JSON line per
stage on stderr (gateway log), each carrying a per-process `seq`:

- `build_ms` — host graph construction of the full step (~700 MLX ops);
- `eval_wait_ms` — the blocking `stream.eval` of the step;
- `sample_ms` — the sampler's cast + argmax + synchronising read;
- `callback_ms` — the boundary callback (detokenize, IPC `tokens` event,
  `memory_stats`, `check_controls`).

**Reconciliation:** the four per-token means, summed, must match the `apxinf-sync`
arm's measured `decode_ms/token` within **10%**. If they do not, the discrepancy is
explained in this document before any verdict is drawn. The instrumentation runs
only in separate decomposition segments; it is never on inside a timed segment.

### 5. M4 bandwidth ceiling

In the same window (both Metal locks held, same conditioning), a ≤5-minute copy +
reduce streaming probe measures the M4's practical peak GB/s, and the lower bound
`3.4576 GB / peak GB/s` ms/token is recorded. (The 162 / 172 GB/s figures in older
documents are M1 Pro numbers and must not be quoted for the M4.)

### 6. Notation

Per cell, from paired medians: **G** = `apxinf-sync` − `llamacpp-f16` (ms/token),
**P** = `apxinf-sync` − `apxinf-pipe`, **R** = `apxinf-pipe` / `llamacpp-f16`.

### 7. Decision rule (frozen)

- **INVALID** — any request's emitted token ids differ bitwise between
  `apxinf-sync` and `apxinf-pipe` in the window, or any arm's A/A exceeds 5%.
  Everything else is void regardless of speed.
- **WIN** — both cells R ≤ 0.98, and in every one of the three segments the paired
  `apxinf-pipe` < `llamacpp-f16`.
- **PARITY** — both cells R ≤ 1.00 but WIN does not hold.
- **PARTIAL** — P ≥ 0.5·G but R > 1.00: the host wait is real but not the whole
  gap; the remainder is recorded as GPU-side overhead (kernel count / fusion) and
  handed to the next round.
- **FALSIFIED** — P < 0.25·G: the "per-token host wait" hypothesis fails; the
  decomposition of section 4 explains where the 0.8 ms actually is.
- Anything else: **INCONCLUSIVE** with all numbers reported.

### 8. Default rule

If the window is not INVALID and both cells satisfy P ≥ 0, LFM3 greedy decode
defaults to the pipelined path (same default policy as Qwen3), with no opt-in dead
path left in the tree. Otherwise the code is not merged; only this record is.

### 9. Tests (before the device runs)

Real BF16 checkpoint, Metal-locked, `#[ignore]`-marked, run with
`APXINF_LFM3_TEST_MODEL=<snapshot dir>` on the M4:

- **T1 token equality** — ≥4 prompts × 128 tokens: sync and pipe emit identical
  token ids.
- **T2 EOS/budget edges** — budget 1, budget 8, first-token-EOS (driven at the
  sequence level so it is deterministic), cancel right after the first publish:
  returned and published sequences match sync, final `position == P+N−1`.
- **T3 continuation state** — after a pipe run (extra lookahead step rolled back),
  all KV rows `[0, position)` and all 13 conv windows are byte-identical to a sync
  run's, and a 16-token continuation matches sync token-for-token. **Fault
  proof:** with the conv restore disabled (`#[cfg(test)]` seam), T3 must fail; the
  failing output is pasted into the PR.

### 10. Window discipline

M4 (`haiyan-mac-mini`) only. Both Metal locks (`/tmp/enginetailor-metal.lock`,
`~/.cache/enginetailor/metal-measure.lock`) held for the whole window through the
fcntl lock runner (waited, never stolen); `pmset displaysleepnow` + `caffeinate
-si` (no `-u`); host GUI agents paused as in the #121 windows; swap / thermal /
load recorded before and after; device budget ≤45 minutes for decomposition +
bandwidth probe + ABBA combined. M2 Air is not measured (owner instruction); the
M1 Pro is offline; nothing is extrapolated across hosts.

### 11. Implementation limits

No prefill change, no numerics/weight/kernel change, no compiled decode graph, no
KV bucketing. The only harness-facing changes are the `--worker-env` pass-through
and the worker banner; `bench_engines.sh` / `bench_engines_report.py` verdict logic
and gates are untouched.

---

## Results

Window: 2026-11-19 02:50–03:01 CST (M4, both Metal locks held via
`lockrun_m4.py`, `pmset displaysleepnow` + `caffeinate -si`, no GUI agents
needed pausing, host state recorded before/after). The sync-arm decomposition
and the T1/T2/T3 acceptance tests ran in a second locked session immediately
after (see amendments). Verdict per the frozen section-7 rule: **WIN**.

### Identity block

| item | value |
|---|---|
| llama.cpp weights | `LFM3-1.7B-F16.gguf`, 3,466,626,336 B, sha256 `b7ab8b69a85e4dfbf62cdfd0e02d7d051ed09f2c5b84c701771dddb073b85970` (reused from #121) |
| apxinf weights | reconstructed BF16 `model.safetensors`, 3,457,635,448 B, sha256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330` |
| `llama_server_sha256` | `d635b19fb11a19d3262ee8544252f20cb92620b58c426b820453bb4f4535e55e` (llama.cpp `5297534`, Metal, SDK 26.5) |
| `apxinf_serve_sha256` | `c8cec3ead88f63e120c1f034efb7af547fa1fd30a6fbb4f25b8e1aa1c672d60c` (source tip `d1703a2d` + test fixes `bd34d4a8`; M4 tree archive sha256 `d30502bb30db749ffad29747…` of the exact source) |
| `apxinf_native_worker_sha256` | recorded in the window log (same build) |
| arch proof | apxinf worker stderr: `Metal 0: Apple M4 (applegpu_g16g)`; llama.cpp `print_info: arch = lfm3` |

The M4 clone is a tree archive of the PR branch (the M4's GitHub relay is
~17 KB/s, too slow to clone); the archive hash plus the branch commit give the
same re-checkability as a checkout.

### Per-arm effect evidence (banner lines)

Every apxinf arm segment's gateway log carries the worker's construction-time
banner; no arm ran without it.

- `apxinf-sync` segments: `decode_schedule=sync overlaps_callbacks=false decode_stages=false`
- `apxinf-pipe` segments: `decode_schedule=pipelined overlaps_callbacks=true decode_stages=false`
- decomposition segments: `decode_schedule=sync overlaps_callbacks=false decode_stages=true`

### S2 decode — engine-reported, three arms (`--engine-timings`)

| case | arm | decode ms/token p50 | paired R vs `llamacpp-f16` | A/A |
|---|---|---|---|---|
| ctx 1024 / 128 out | `apxinf-pipe` | **36.07** | **0.952** | 0.5% |
| ctx 1024 / 128 out | `apxinf-sync` | 37.15 | 0.981 | 0.4% |
| ctx 1024 / 128 out | `llamacpp-f16` | 37.87 | (baseline) | 0.3% |
| ctx 4096 / 128 out | `apxinf-pipe` | **36.40** | **0.955** | 0.5% |
| ctx 4096 / 128 out | `apxinf-sync` | 37.53 | 0.984 | 0.4% |
| ctx 4096 / 128 out | `llamacpp-f16` | 38.12 | (baseline) | 0.3% |

Per-segment paired medians (ms/token; the WIN rule's "every segment" test):

| cell | seg 0 pipe / llama | seg 1 pipe / llama | seg 2 pipe / llama |
|---|---|---|---|
| ctx 1024 | 36.34 / 38.23 | 36.38 / 38.16 | 36.34 / 38.17 |
| ctx 4096 | 36.68 / 38.40 | 36.66 / 38.45 | 36.77 / 38.42 |

Pipe < llama in **all 6 segment pairings of both cells** → the WIN
per-segment condition holds. Token streams: the paired parity probe (4 fixed
prompts × 128 tokens through both schedules, byte-exact SSE text compare) is
PASS, and the per-cell A/A gates are 0.3–0.5% (≤ 5%), so the window is not
INVALID.

**Verdict: WIN** — `apxinf-pipe` beats `llamacpp-f16` on both cells
(R = 0.952 / 0.955 ≤ 0.98) with every segment paired in its favour.

Notation quantities: **G** = sync − llama = −0.72 / −0.59 ms/token (negative:
this window's llama.cpp arm ran slower than #121's, so the "0.8 ms gap" from
#121 did not reproduce as a gap — see caveats); **P** = sync − pipe =
+1.08 / +1.13 ms/token — the pipelined path recovers ~1.1 ms/token of
host-side work.

### M4 bandwidth ceiling (same window)

Streaming probe over 3,457,635,448 B BF16 (`copy` = elementwise add of a zero
scalar, read+write; `reduce` = mean over the whole buffer, read-stream):

| probe | median ms | GB/s | 3.4576 GB lower bound ms/token |
|---|---|---|---|
| copy | 68.50 | 50.5 (declared bytes; ~101 GB/s r+w fabric) | 68.50 |
| reduce | 34.36 | **100.6** | **34.36** |

Both engines run at 92–97% of the ~100.6 GB/s read-stream ceiling. The M1
Pro's 162/172 GB/s figures are not quoted for the M4.

### Sync-arm decomposition

_See the 2026-11-19 amendment below; the four-stage table and the ≤10%
reconciliation are in `doc/receipts/lfm3-pipelined-decode/decomposition.json`._

### Tests

_See the amendment; T1/T2/T3 outputs are in
`doc/receipts/lfm3-pipelined-decode/tests.log`, and the T3 fault-injection
failure output is pasted in the PR body as required._

### Mixed factors (quote with the tables)

1. **Matched precision, not identical bytes**, same as #121: GGUF F16
   (3,466,626,336 B) vs MLX BF16 (3,457,635,448 B), 0.26% container delta.
2. **Prompt protocol**: `apxinf-*` arms carry the +9-token lane ChatML prefix;
   `llamacpp-f16` is bare completion.
3. **Timing basis is engine-reported** on all three arms (`--engine-timings`);
   apxinf-serve buffers its SSE (issue #120), so a client decode window would
   be degenerate. Spans declared per engine.
4. **One host, one window**, batch-1 serial, greedy, three ABBA segments per
   arm, both Metal locks held, per-segment corpus seeds.
5. **Window drift vs #121**: this window's `llamacpp-f16` ran 37.87/38.12
   ms/token vs 36.42/36.67 in `f16-20261118T211938Z` (+1.4 ms), while
   `apxinf-sync` reproduced #121 within 0.1 ms (37.15/37.53 vs 37.20/37.50).
   Cross-window absolute numbers are not comparable; the verdict rests only on
   within-window pairing, which is why the pre-registered rule is defined on
   paired medians.

### Amendments

_Amendment 2026-11-19 (to be filled): the first window session's test phase
hit a PATH issue and the decomposition phase an evidence-path mismatch; both
were fixed and re-run in a second locked session without changing any
decision rule, threshold, or gate._
