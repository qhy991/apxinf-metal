# Receipts: LFM3 launch-gap harvest (issue #130 item 1)

Window: 2026-11-20 13:42–13:55Z on M4, one session (runner
`~/launch-gap-run/run-all.sh`), both Metal locks held throughout the
device phases.

Contents:

- `session.log` — the runner's phase/gate/bench journal (locks, build,
  banner asserts, bench exits, PAIRED/CELL lines).
- `census.json` — gate-0 census summary (banner proof, retrace gate,
  host-submission timing, what is 机制未知).
- `c4.log` / `c1.log` / `c2.log` / `c3.log` — gate outputs (token equality,
  retrace census, edge and state gates).
- `raw/` — per-segment bench JSON (`apxinf-{pipe,pipe-compiled}-s2-*.json`).
- `logs/` — census gateway stdout + diagnostics (banner and
  `decode_stage` lines) and the stage-decomposition failures (that
  instrument parses only the sync schedule; pipelined arms are N/A).

Verdict (doc/lfm3-launch-gap.md §"2026-11-20 13:57"): **PARTIAL** —
(1024,128) C=0.1851 ms/token, (4096,128) C=0.2511, A/A 0.70%/1.07%,
token-identical, no retrace. The compiled step stays behind
`APXINF_LFM3_COMPILED_DECODE`.

Attempts before the clean run (harness bugs fixed before any device run,
recorded for the audit trail): (1) an inverted lock-peek while loop;
(2) a wrong release-build feature list (`--features mlx,coreml` on
`apxinf-serving`, which owns neither); (3) a gate false-negative on a
module-qualified test name. No gateway started and no device ran in any
of them.
