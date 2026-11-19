# LFM3 merged GEMV projections — pre-registration (written before any device measurement)

Status: registered 2026-11-19, before any build or measurement of this change.

## Why

The matched-precision comparison (#121) puts apxinf-native at **37.20 / 37.50 ms/token** vs
llama.cpp F16 at **36.42 / 36.67** (ctx 1024 / 4096, gen 128) — an apxinf deficit of
**0.78 / 0.83 ms per token** that is constant across context length, i.e. fixed per-token cost,
not bandwidth or attention.

Two independent sources of that fixed cost were identified in issue #122:

1. **Host-side gaps**: the decode loop synchronises per token (graph build, `stream.eval` wait,
   sample + sync, boundary callbacks) — being addressed by issue #122's pipelined decode;
2. **Kernel count / fusion**: ~700 launches per step, with every launch carrying a barrier. This
   change attacks one part of it: the four attention projections (q, k, v, per-head gate) read the
   **same** `n` and the MLP's `w1`/`w3` read the **same** `ffn`, yet each is a separate matmul.
   Merging them removes **3 launches per attention layer** and **1 per layer** for the MLP:
   11 attention layers × 3 + 24 × 1 = **57 fewer launches per token**.

An MLX-lane precedent exists: the EngineTailor lane runs `_merge_mode=both` (qkv + w13 merged) by
default; this brings the same algebraic idea to the native Metal path.

## Change

Behind the cargo feature `merged-gemv` (default off; a feature, not an env switch, because the
gateway spawns the worker with `env_clear()` — see the campaign's "empty arm" lesson):

- `weights.rs`: at load time build `qkvg = concat(q, k, v, gate)` along the output axis into one
  `[hidden, q + 2*kv + heads]` "linear", and `w13 = concat(w1, w3)` into `[hidden, 2*inter]`.
  Original projections are kept, so the feature-on binary carries the extra merged copies
  (≈ +0.3–0.5 GB) — a cost to report, not to hide.
- `model.rs`: both the chunked path (`attention_layer`) and the single-token path
  (`attention_step`, the one that matters for ms/token) take one matmul and slice the four
  outputs; the gated-MLP takes one matmul and slices `gate` / `up`.

## Industrial gate (fixed now, before measuring)

- **Correctness (hard)**: greedy output must be **token-identical** to the feature-off binary on
  ≥ 4 prompts × 64 tokens. A single differing token ⇒ **INVALID**, no matter the speed. (Note:
  concatenating weights can change matmul tiling and therefore accumulation order, so this is not
  assumed — it is tested.)
- **Performance**: M4, both metal locks, display asleep, same window, ABBA `A B B A A B`,
  ≥ 3 segments/arm, S2 decode ctx 1024 and 4096, gen 128, engine-reported timing
  (`--engine-timings`; client timing is unusable until issue #120 is fixed), paired medians,
  A/A ≤ 5 % per arm per cell.
- **Decision rule**:
  - both cells ≥ **+3 %** faster and tokens identical ⇒ **KEEP**, enable the feature by default;
  - 0–3 % faster ⇒ record, stay opt-in, do not flip the default;
  - slower in either cell ⇒ **NOT-WORTH**, revert, record verbatim.
- **Report**: per-cell ms/token for both arms, paired delta, A/A, effective GB/s (bytes identical
  in both arms — only launch count changes), and the launch-count reduction as arithmetic.

## Out of scope

Pipelined decode (#122), RoPE/fusion of the element-wise chains, compile/bucketing (#115 closed
those), any change to prefill or numerics.

## Measurement result — 2026-11-19, M4 (`haiyan-mac-mini`)

Status: **INVALID** — the hard correctness gate failed. No speed measurement was
taken, and the decision rule was not evaluated (its KEEP branch requires token
identity, which does not hold).

### Builds (same tree, one feature apart)

| binary | default | `merged-gemv` |
|---|---|---|
| `apxinf-serve` sha256 | `f3ff54661cc8565d183038eb725c9dcdc89dc1f9a9722562241f29b189e75f24` | `011d65fb63f83d426f0d00dbcabb5d24977ce1f4bed96b30eb9be2311181c136` |
| `apxinf-native-worker` sha256 | `ffd9826f7b3f0e596bc3878545cf8bae0f23f98748e947ef583423670c5ab511` | `ab93b0a4daee4f29fa10b6422232cefdffd47dc34cb6c0708d5465082760cd6d` |

Built on the M4 from branch tip `05e352c7` (off `c9c5dd2a`) with
`cargo build --release -p apxinf-serving --features native-minicpm5 --bins`
(default) and `--features "native-minicpm5,apxinf-model/merged-gemv"` (treatment);
`SDKROOT=…/MacOSX26.5.sdk`, `MLX_ROOT=~/mlx-sdk`. Provenance check: the default
build reproduced `f3ff5466…` bit-for-bit, the exact binary measured in
#119/#121, so the control is the shipped path. The full source tarball transfer
stalled on the relay, so the M4 tree (verified to be exactly `fbd1e2e9` for the
touched files) received the branch's exact Rust delta: `git diff fbd1e2e9 05e352c7
-- crates/` is precisely the three files `crates/apxinf-model/Cargo.toml`,
`…/src/metal_lfm3/model.rs`, `…/src/metal_lfm3/weights.rs`, hash-verified after
copy; everything else in that diff is docs/harness.

### Correctness (hard gate)

Reconstructed BF16 checkpoint (sha256 `135922ca…`), chat surface, greedy
(`temperature=0`), `max_tokens=64`, five fixed prompts:

- **Control**: the default build ran twice; the two runs were byte-identical on
  all five prompts.
- **Treatment**: `merged-gemv` ran three times; all three identical to each other.
- **Result**: `merged-gemv` **deterministically differs** from default on prompt 0
  (`What is the capital of France?`):
  - default: `The capital of France is **Paris**.` — 60 completion tokens,
    `finish_reason: stop`;
  - merged: `The capital` — 64 completion tokens, `finish_reason: length` (the
    rest of the trajectory sits inside an unterminated structured block, which is
    why the visible text stops there; the token streams differ).
- Prompts 1-4 were text-identical, which is exactly the "mostly equal" outcome
  the gate exists to reject.

**Token-level check**, independent of the gateway's structured-block rendering:
`/v1/replay` with the same `prompt_token_ids`, `steps=64`, greedy, on both
binaries, four raw prompts:

| prompt | first differing token index |
|---|---|
| `The capital of France is` | **38** |
| `The first five prime numbers are` | none (64/64 identical) |
| `The sky appears blue because` | **15** |
| `A Python function that sums a list` | **58** |

The trajectories agree for the first tokens and then diverge deep in the
sequence — the signature of a numerical perturbation (concatenating the weights
changes matmul tiling and therefore accumulation order), not a gross
shape/indexing bug. The pre-registration anticipated exactly this ("this is not
assumed — it is tested").

Gate as pre-registered: *"greedy output must be token-identical … A single
differing token ⇒ INVALID, no matter the speed."* → **INVALID**. No speed window
was run, so this change has **no timing number**.

### Cost actually measured (worker bytes, from the gateway's worker metrics)

| build | active bytes | peak bytes |
|---|---|---|
| default | 3,458,508,288 (3.46 GB) | 3,472,468,304 |
| `merged-gemv` | 5,107,590,656 (5.11 GB) | 5,120,485,712 |
| delta | **+1.649 GB** | **+1.648 GB** |

That is ~3.3x the pre-registration's "+0.3-0.5 GB" estimate. The arithmetic
explains it — the merged copies are full BF16 matrices:
`w13`: 24 × 2048 × (2 × 7680) × 2 B = 1.510 GB;
`qkvg`: 11 × 2048 × (2048 + 2×512 + 16) × 2 B = 0.139 GB;
total 1.649 GB, matching the measured +1.649 GB. The originals are kept, as the
design says; the estimate was wrong, not the measurement.

### Launch-count arithmetic (unchanged, for the record)

11 attention layers × 3 + 24 layers × 1 = **57 fewer launches per decode step**,
identical bytes. The matched-precision window (`f16-20261118T211938Z`) put the
apxinf deficit at 0.78 / 0.83 ms/token of fixed per-token cost; if all of that
were launch cost it would imply ~13.7 µs per launch. This change cannot be
credited with any share of it, because it does not preserve the token stream.

### Verdict

**INVALID under the pre-registered gate; do not enable the feature.** The
implementation is a clean expression of the idea and its memory cost is now
measured, but the gate's hard requirement fails deterministically.

Options for whoever picks this up:
1. keep the feature default-off with this record (recommended — the shipped path
   is unchanged and no default flips on invalid evidence);
2. re-tile the merged matmul so that the accumulation order matches the four
   separate projections bit-for-bit; only if that succeeds does the launch saving
   deserve a speed window;
3. relax the gate to "approximately equal" — **not** recommended: the campaign's
   pre-registration makes numeric drift a hard fail, and this is the case it was
   written for.

Host state during the correctness runs: load 1.9-2.9, swap used 1.05-1.09 GB. The
correctness runs made no timing claim and therefore did not take the Metal
measurement locks; the host was idle.
