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

_Not yet measured. This section is filled after the window; the verdict uses
exactly the section-7 vocabulary._
