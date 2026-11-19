# LFM3 pipelined-decode receipts (issue #122)

Every number in `doc/lfm3-pipelined-decode.md` recomputes from these files.

| file | what | how to recompute |
|---|---|---|
| `summary.json` | `bench_engines_report.py` machine summary (A/A, S2 rows, gates) | `python3 scripts/bench_engines_report.py --raw-dir <this dir>/raw --stage s2 --baseline llamacpp-f16 --apxinf-arms apxinf-sync,apxinf-pipe --json-out summary.json` |
| `raw/*.json` | 9 harness arm files (3 arms × 3 ABBA segments), engine-reported timings, corpus hashes, host state, artifact identity | produced by `scripts/bench_engines.sh` |
| `per_segment_paired.json` | per-segment paired medians pipe vs llamacpp-f16 and the WIN-rule check | regenerate from `raw/` with the snippet in `../../doc/lfm3-pipelined-decode.md` |
| `logs/status.log` | orchestrator phase log with identity sha256s | window runner |
| `logs/host-{before,after}.json` | load, swap, thermal, top-CPU around the window | `scripts/host_state.py` |
| `logs/bandwidth.json` | M4 copy/reduce streaming peak over 3,457,635,448 B | `m4_bw_probe` example (M4-only tool) |
| `logs/apxinf-parity-{sync,pipe}-diag.jsonl` | gateway diagnostics of the paired parity probes (banner + worker stderr) | grep `decode_schedule=` |
| `parity/verdict.json` | sync-vs-pipe token-stream equality (INVALID gate) | `scripts/parity_check.py` |
| `decomposition.json` | sync-arm four-stage sums vs measured decode ms/token | `scripts/stage_decomposition.py` |
| `tests.log`, `tests-fault-injection.log` | T1/T2/T3 acceptance runs and the T3 fault-injection failure | cargo test on the M4 |

Window tool copies (orchestrator, probes, analyzer) live in the git-ignored
feature dir `devlocal/lfm3-pipelined-decode/scripts/` on the M1 worktree and
on the M4 host under `~/lfm3bench/lfm3-pipelined-decode/scripts/`.
