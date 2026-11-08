# EngineTailor promoted config vs apxinf native — gap matrix

Source of truth for the EngineTailor (ET) side: `enginetailor/models/d3lite/engine.py`
`PROMOTED` (the only model in ET that declares a promoted point). apxinf side:
`crates/apxinf-model/src/metal_d3_lite/` on `main`.

## d3-lite: 4 of 6 promoted items are already ported

| ET option | ET promoted value | apxinf native | Status |
|---|---|---|---|
| `gdn_core` | `apxinf_eight_lanes` | `GatedDeltaRecurrence` with `APXINF_D3_GDN_CORE` **defaulting to `eight_lanes`** (image path stays `four_values` for parity headroom) | ✅ ported (PR #21) |
| `conv1d` | `apxinf` | `CausalDepthwiseConv`; `APXINF_D3_CONV1D=stock` restores the composed MLX `conv1d` | ✅ ported (PR #21) |
| `rope` | `apxinf` | `HalfSplitRope`, FP32 angles, one rounding to the input dtype | ✅ ported (PR #12) |
| `compile` | `model` | `Compiled::prepare` per admitted length, replay on every request | ✅ ported (PR #12) |
| `gemm_dtype` | `f16` | BF16 only — the variant contract admits `auto`/`bf16`/`bf16-compiled` and **rejects f16** (`metal_d3_lite/mod.rs`) | ❌ missing (blocked by the variant contract, not by a kernel) |
| `gdn_block` | `fused_prep` | the GDN operand chain runs as separate ops (conv, silu, q/k l2-norm, recurrence, gated norm, gate) | ❌ missing |

So the remaining migration surface for d3-lite is **two items**, not six.

### What that means for the "≈24 % gap"

The review's `EngineTailor promoted 2808 ms vs apxinf native 3693 ms` predates PR #21
(the promoted-config port). Any current gap claim has to be re-measured on the same
suite with the same warm-up; the two numbers in the review were not taken at the same
point in the port's history and are not a like-for-like pair.

### The two open items, ordered by risk

1. **`gdn_block=fused_prep`** — in ET this runs the GDN operand chain as one kernel
   (`gdn_fused_prep`). In apxinf it is 6–7 separate dispatches per GDN layer. The
   mechanism is proven on the ET side; the port needs a new Metal kernel and a parity
   pass on all 24 GDN layers (the 42-case gate plus the image path).
2. **`gemm_dtype=f16`** — ET measured a roofline band of 0–7 % for F16 weight copies.
   apxinf rejects the variant today, so the port also means extending the variant
   contract and its parity evidence. F16 changes accumulation rounding, so this one
   carries a real parity risk on the 0.0022 headroom of the Clef-style gates; for
   d3-lite the budget is wider (48-case receipt, 0.15 tolerance) but still needs proof.

## Other families

ET declares no `PROMOTED` for `clef_flash`, `jev`, `minicpm`, `qwen3`, `flux`, `kimi` or
`llama`; their wins live in campaign recipes rather than a promoted point. For those the
migration is per-win, not per-config, and each needs its own evidence trail.

## Recommended first move

Re-measure apxinf native d3-lite against ET promoted **on the ET gate suite** before
porting either open item: if the gap is now small, porting `fused_prep` is the only
remaining item worth doing; if it is still large, the difference is not in the option
list and needs a stage-level comparison instead.

## Measured: the M1 Pro gap is now 8.7 %, not 24 % (2026-11-08, issue #52 step 1)

Re-measurement requested by the issue #52 follow-up (2026-11-08 17:45 +08:00). This
section is additive evidence; the matrix above is unchanged.

### Setup

- Host: **Apple M1 Pro**, 16 GB, macOS 27.0.1. Native: `apxinf-metal@a6a1fa2c`
  (main after PR #53/#54), release `d3_lite_bench`, MLX SDK 0.32.3
  (`applegpu_g13s`). ET: `EngineTailor@5a41fa1`, python-mlx 0.32.0.
- Model: `d3-lite-9c10703` (`config.json` sha256 `a9071195384e…`, identical to the
  2026-10-23 Graft receipts). Workload: ET `testdata/workload.json` sha256
  `68201d7ec0fc…` (48 gate + 8 timing cases).
- Native arm = current main defaults: `gdn_core=eight_lanes` (text), `conv1d=apxinf`,
  `HalfSplitRope`, whole-request compiled callable, `last_row=full`,
  `gdn_block=separate`, `gemm_dtype=bf16`.
- ET arms = `stock` and `PROMOTED` (all six knobs incl. `gemm_dtype=f16` +
  `gdn_block=fused_prep`).
- Every model process ran under the two standing Metal locks
  (`/tmp/enginetailor-metal.lock`, `~/.cache/enginetailor/metal-measure.lock`) via
  `orchestrate.lock_run`.

### Decision agreement (48 gate cases)

| Pair | Decisions | Worst max\|Δp\| | Mean KL |
| --- | ---: | ---: | ---: |
| native vs FP32 oracle (2026-10-23 receipt) | **48/48** | 0.0232 | 2.02e-4 |
| native vs ET promoted (fresh collect) | **48/48** | 0.0342 | 3.21e-4 |
| ET promoted (fresh) vs FP32 oracle | **48/48** | 0.0231 | 2.63e-4 |
| ET promoted (fresh) vs ET promoted (2026-10-23) | **48/48** | 0.0266 | 2.35e-4 |

Native's own gate is byte-identical across repeats and passes the standing budget
and the stricter 0.035 bound. Cross-harness probability deltas up to 0.034 reflect
kernel-order and F16-GEMM differences; **no decision flips anywhere**.

### Latency (8 timing cases, fast machine state)

Native: per-case medians over 2 clean process visits × 3 timed calls (both visits
Swapouts +0). ET: per-case paired medians from a 12-block ABBA/BAAB bench
(`--calls 5`, 12/12 valid blocks, all stable; suite medians reproduced the
2026-10-23 campaign within 0.3 %).

| Case | Bucket | native ms | ET stock ms | ET promoted ms | native vs promoted | native vs stock | ET promoted gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b0150_engines_kind | 150 | 70.2 | 78.9 | 65.8 | **+6.6 %** | −11.0 % | +19.8 % |
| b0150_quickstart_route | 150 | 54.4 | 61.7 | 49.7 | **+9.3 %** | −11.9 % | +24.1 % |
| b0400_license_kind | 400 | 161.1 | 184.9 | 146.0 | **+10.3 %** | −12.9 % | +26.2 % |
| b0400_kitreadme_topic | 400 | 166.2 | 191.2 | 151.3 | **+9.9 %** | −13.1 % | +26.5 % |
| b1000_suite_kind | 1000 | 362.1 | 419.4 | 331.8 | **+9.1 %** | −13.7 % | +26.4 % |
| b1000_readme_topic | 1000 | 365.3 | 425.0 | 334.7 | **+9.1 %** | −14.1 % | +26.6 % |
| b2400_suite_kind | 2400 | 894.7 | 1049.4 | 827.3 | **+8.1 %** | −14.8 % | +26.9 % |
| b2400_readme_topic | 2400 | 896.1 | 1049.3 | 825.5 | **+8.6 %** | −14.6 % | +27.1 % |
| **suite** | — | **2970.0** | **3459.8** | **2732.1** | **+8.71 %** | **−14.16 %** | **+21.0 %** (sums); **+26.83 %** paired, p 2.4e-4 |

Per bucket the native-vs-promoted gap is uniform: **+7.8 % (150), +10.1 % (400),
+9.1 % (1000), +8.4 % (2400)** — no length regime dominates, so the residual is a
per-layer cost, consistent with the two knobs native does not enable
(`fused_prep`, `f16`), not a pipeline or shape effect.

### What the 24 % claim means now

The review pair (`ET promoted 2808 ms` vs `native 3693 ms`) predates PR #21. On the
current main, native sits **between** ET stock and ET promoted: **14.2 % faster
than ET stock**, **8.7 % behind the full promoted point**. On suite sums the
promoted point buys 21.0 % over stock; native already captures 14.2 of those
points (GDN eight-lanes + apxinf conv + rope + compile, plus its own last-row
pruning that ET lacks), and the remaining native→promoted step (8.7 %) tracks the
two knobs still opt-in on main (`APXINF_D3_GDN_BLOCK=fused_prep`,
`APXINF_D3_GEMM_DTYPE=f16`). The gap is **significant enough to justify step 2**
(an M1 Pro paired campaign for `fused_prep`, then `f16` separately — the M4
verdicts do not transfer: M4 measured `fused_prep` +3.55 % KEEP and `f16`
−3.68 % NOT-WORTH, but ET's promoted point, which includes F16, was promoted on
this M1 Pro).

### Validity notes

- **Machine bimodality**: the host shifted to a ~22 % slower state during
  21:53–22:00 (ET bench block 11, and native visits B2/C2 at suite 3582.7 ms,
  thermal level 0, no swap). Headline numbers use only the fast-state windows,
  which reproduced across 2.5 h (native visits A/A2: 361.5 vs 361.5 ms overall
  median; ET promoted suite 2735 ms at 19:32 vs 2731 ms at 21:20). Slow-state
  rows are retained in `analysis.json`, not used for the comparison.
- **Contention incidents**: an unlocked competing `d3_lite_bench` campaign
  (`/tmp/m1-camp`, 19:41–19:47) overlapped the first ET bench — its blocks 3–7
  are excluded (that run was killed and redone; the competing campaign's own
  r1–r3 numbers are likewise contaminated by my locked bench and should be
  discarded). The second ET bench was SIGKILLed at 20:13 (no jetsam record;
  cause unknown) during warmup and produced no data. The third run completed
  12/12 clean blocks.
- Cross-harness absolute latency is like-for-like (same synced per-request wall
  time after warmup), but the harnesses differ: native replays one compiled
  callable per `(length, count)`; ET `compile=model` retraces per shape in one
  Python process holding both arms.
- Evidence (not committed): `devlocal/et-m1pro-gap/out/` —
  `native-gate/`, `et-promoted-gate/`, `native-timing-{A,A2,B2,C2}/`,
  `et-bench3/bench.json`, `analysis.json`, `campaign.log`.
