# LFM3-1.7B engine comparison: apxinf vs llama.cpp vs ollama

Status: **method + pre-registered gate; measurements pending (stage 2).**
Design: GitHub issue [qhy991/apxinf-metal#111](https://github.com/qhy991/apxinf-metal/issues/111)
(the working copy under `devlocal/engine-comparison-lfm3/DESIGN.md` is git-ignored).
Harness: [`scripts/bench_engines.sh`](../scripts/bench_engines.sh),
report: [`scripts/bench_engines_report.py`](../scripts/bench_engines_report.py).

The question is specific and falsifiable: on one M1 Pro, with one weight family and
one measurement discipline, how does apxinf's `metal_lfm3` batch-1 serving compare
with llama.cpp and ollama for (1) single-request decode, (2) prefill/TTFT on long
prompts, and (3) end-to-end agent work? A result of "apxinf decode lags" is a
useful result; the point of this document is that the comparison is re-checkable
rather than that apxinf wins.

## Arms

| arm | engine | weights | serving surface | declared prompt protocol |
|---|---|---|---|---|
| `apxinf-native` | apxinf-metal `metal_lfm3` (Rust + Metal) | **MLX BF16** (`bf16-eager`, the only implemented variant) | `apxinf-serve` (`/v1/chat/completions`) | lane ChatML (family-rendered) |
| `apxinf-mlx-lane` | EngineTailor MLX engine + Python worker (current agent stack) | MLX packed4, affine group 64 | `apxinf-serve` (Python provider) | lane ChatML |
| `llamacpp` | llama.cpp `llama-server` | `LFM3-1.7B-Q4_K_M.gguf` | `/v1/completions` | bare completion |
| `ollama` | ollama (llama.cpp runtime + its own scheduler) | the same Q4_K_M GGUF, imported by Modelfile | `/v1/completions` (primary), `/api/generate` with `raw: true` (S3 decomposition, template-free) | bare completion |
| `llamacpp-f16` | llama.cpp `llama-server` | `LFM3-1.7B-F16.gguf` (16-bit tier) | `/v1/completions` | bare completion |

The 16-bit tier (`apxinf-native` BF16 vs `llamacpp-f16`) is the quantization-neutral
pair the decision rule needs; the 4-bit tier is `llamacpp` vs `ollama`. `apxinf-mlx-lane`
is the only packed4 intersection between the two engines, and it could not be
provisioned in this window (see the 2026-11-18 amendment).

`apxinf-serve` publishes no bare-completion route (its declared text surface is
`/v1/chat/completions` plus `/v1/replay`), so the apxinf arms are measured on the
chat surface while llama.cpp and ollama are measured on the completions surface.
That is a declared mixed factor (see below), not an accident of the harness.

## Pre-registered gate and decision rule

Copied from the campaign design before any number existed. This text is the gate;
if it turns out to be wrong it is amended with a date, never re-picked after
seeing the data.

```
Gate (comparability): same-arm repeat (A/A) <= 5%; otherwise that arm's data is
                 void and is re-measured.
Gate (S2 decode): report the paired median; if apxinf < 0.8 x llamacpp, write the
                 verdict as "apxinf decode lags" and open a follow-up (do not
                 write "parity").
Gate (S4 end-to-end): same solved set and wall delta > 10% => engine-level
                 difference; different solved set => investigate template and
                 quantization before drawing a conclusion.
Decision rule: the difference may be attributed to the "engine implementation"
                 only when the 16-bit tier reproduces it; otherwise the
                 attribution is "quantization tier difference".
```

The report script applies this text mechanically: it prints the A/A verdict per
arm, the paired S2 ratio against the gate's 0.8 threshold, and the attribution
line, and it exits non-zero when the A/A gate, the evidence requirement, or the
minimum-segment requirement is not met. It never prints a verdict that ignores a
blocker.

## Controls

1. **Prompt shape.** S1-S3 use bare completion (no chat template) on the
   completions surface. The apxinf arms are the one exception, declared and
   measured: their `prompt_tokens` include the family's lane ChatML prefix, and
   the report prints that offset in the corpus table. S4 goes through each
   engine's own template and records the rendered prompt that was actually sent.
2. **Decoding.** `temperature=0`, fixed `seed`, fixed `max_tokens`, same stop
   condition. Greedy only.
3. **Prompt corpus.** One corpus seed, a fixed generator, and the same text for
   every arm; `corpus.sha256` is recorded in every raw file. Nominal lengths are
   128 / 512 / 2048 / 8192 tokens for S1, context 1024 / 4096 for S2. Each arm
   reports its own `prompt_tokens` and the report compares them; the stage-2
   corpus is frozen with llama.cpp's `/tokenize` so the nominal and actual counts
   agree for every bare-completion arm. Distinct text per repeat, so no engine's
   prompt/KV cache can turn a repeat into a cache hit.
4. **Machine state.** `pmset displaysleepnow` + `caffeinate -si` (never `-u`),
   both Metal locks held for the whole window
   (`/tmp/enginetailor-metal.lock`, then `~/.cache/enginetailor/metal-measure.lock`)
   through `lockrun.py --tag ... --`, one warm-up request per arm, thermal /
   swap / disk state recorded in every raw file, and no other campaign running.
5. **Ordering and repetition.** Interleaved ABBA (`A B C D D C B A`), at least
   three segments per arm, medians only. A single-order number is not a result
   and is not reported.
6. **Artifact identity.** `sha256` of the engine binary and of the weight file,
   commit and branch, plus the GGUF revision, are recorded per run. An arm whose
   binary or weights differ from the recorded hash is not the same arm.

## Stages

**S1 prefill / TTFT.** Prompts of 128 / 512 / 2048 / 8183 tokens, `max_tokens=8`,
three repeats per segment, plus one 8192-token rejection probe per arm (see the
2026-11-18 amendment). Metric: client TTFT (first SSE content delta), prefill tok/s
from each engine's own reported prompt tokens, with the engine-reported prefill as a
cross-check.

**S2 decode.** `max_tokens` 128 and 512 at context 1024 and 4096, batch 1. Metric:
client decode window per generated token (first to last content delta), decode
tok/s, with the engine-reported ms/token as a cross-check.

**S3 serving-layer overhead.** The same `max_tokens=8` request issued 50 times per
segment. Metric: client wall per request, and server overhead = wall minus the
engine-reported prefill and decode (reported n/a where an engine does not
publish both).

**S4 end-to-end.** The 12-task ANEWeave agent suite through each engine's
OpenAI-compatible endpoint. Run only if S1-S3 are clean and the metal locks
allow it; it is a separate tool and is not driven by `bench_engines.sh`.

**S5 quality sanity.** Fixed 16 prompts, 4-bit tier against 16-bit tier per arm;
report token agreement and solved-set differences, not cross-quantization
bit-exactness.

### The 8K S1 point

The LFM3 family's `MAX_CONTEXT` is 8192 and its capacity check counts prompt plus
generation (`request_capacity(MAX_CONTEXT - 1, 1)` is the largest permitted
request), so an 8192-token prompt with `max_tokens=8` is **rejected** by the native
family by construction; llama.cpp and ollama serve it when launched with a larger
context window. Per the 2026-11-18 amendment the measured 8K row uses a
**8183-token prompt** on every arm (8183 + 8 = 8191), and the exact 8192-token
request is still sent once per arm so the raw JSON records which engines accept it.
That rejection is reported as a serving-contract capability finding, not as a
measurement.

## Running it

One invocation drives one arm, one stage, one ABBA segment:

```sh
# llama.cpp, bare completion, S1 segment 0
scripts/bench_engines.sh \
  --arm llamacpp --stage s1 --segment 0 \
  --endpoint http://127.0.0.1:8081 --model LFM3-1.7B-Q4_K_M \
  --binary-hash  "<sha256 of llama-server>" \
  --model-hash   "<sha256 of LFM3-1.7B-Q4_K_M.gguf>" \
  --quantization gguf-q4_k_m \
  --evidence-log devlocal/engine-comparison-lfm3/logs/llama-server.log \
  --evidence-pattern 'n_past|prompt eval time|eval time'

# apxinf-native, chat surface, S1 segment 0
scripts/bench_engines.sh \
  --arm apxinf-native --stage s1 --segment 0 --protocol openai-chat \
  --endpoint http://127.0.0.1:8099 --model LFM3-1.7B \
  --binary-hash "<sha256 of apxinf-serve>" \
  --model-hash   "<sha256 of model.safetensors>" \
  --quantization mlx-packed4-affine-g64 \
  --request-log devlocal/engine-comparison-lfm3/logs/apxinf-requests.jsonl \
  --evidence-log devlocal/engine-comparison-lfm3/logs/apxinf-serve.log \
  --evidence-pattern 'request_settled'
```

The script writes `devlocal/engine-comparison-lfm3/raw/<arm>-<stage>-<ts>.json`
and exits `3` when no request produced a token and no evidence line was found: a
no-op arm is not a result. `--dry-run` prints the case plan and the corpus hash
without sending anything.

### Launching the arms

`llama-server` (LFM3 support: `llama-cli --version` plus `lfm3` in the load log
is the engagement check; `-c 16384` leaves room for the 8192-token S1 prompt):

```sh
llama-server -m LFM3-1.7B-Q4_K_M.gguf -c 16384 -ngl 99 --host 127.0.0.1 --port 8081
```

ollama (import the local GGUF; bake `num_ctx`/sampling into the model because the
OpenAI-compatible endpoint does not accept them per request):

```sh
printf 'FROM %s\nPARAMETER num_ctx 16384\nPARAMETER temperature 0\nPARAMETER seed 42\n' \
  "$PWD/LFM3-1.7B-Q4_K_M.gguf" > Modelfile
ollama create lfm3-q4km -f Modelfile
ollama serve            # 127.0.0.1:11434
```

`apxinf-serve` (built on the M1 host from `feat/lfm3-metal`; the native worker is
behind the `native-minicpm5` feature; `--request-log` is what lets the report
attach apxinf's own prefill/decode durations to each request):

```sh
cargo build --release -p apxinf-serving --features native-minicpm5 --bins
export DYLD_LIBRARY_PATH="$MLX_ROOT/lib"
./target/release/apxinf-serve \
  --model "$SNAPSHOT" --model-id LFM3-1.7B --provider native --model-name metal_lfm3 \
  --native-worker ./target/release/apxinf-native-worker \
  --metal-library "$MLX_ROOT/lib/libmlx.dylib" \
  --host 127.0.0.1 --port 8099 \
  --max-context 8192 --max-output-tokens 512 \
  --host-pressure-policy disabled \
  --request-log devlocal/engine-comparison-lfm3/logs/apxinf-requests.jsonl
```

The snapshot directory may contain only regular files (symlinks to the same
checkpoint files are fine): the identity manifest hashes every artifact and
rejects subdirectories.

### Reporting

```sh
python3 scripts/bench_engines_report.py \
  --raw-dir devlocal/engine-comparison-lfm3/raw \
  --baseline llamacpp --apxinf-arms apxinf-native \
  --json-out devlocal/engine-comparison-lfm3/summary.json
```

The report prints the arms inventory, the corpus/token-count comparability
table, the S1/S2/S3 tables with paired ratios, the mixed-factors block, and the
pre-registered gate evaluation. Exit status `0` means the data meets every
pre-declared gate; `2` an A/A violation; `3` an arm with no generation evidence;
`4` an arm below the minimum segment count. Any non-zero status means no verdict
may be claimed from that data set.

A/A has two forms and both are enforced. Per-arm: within one arm, the spread of
the segment medians for every case (ABBA gives at least three segments). And
replicate: the same configuration run under `<arm>-aa`, which the report pairs
against `<arm>` and compares with the same 5% tolerance.

**Evidence of engagement.** Every raw file records per-request token counts, a
response identifier, and the engine-reported durations when the surface exposes
them, plus the matching line from the engine's own log (`--evidence-log` +
`--evidence-pattern`). A file whose requests produced no token and whose log
marker never appeared is a no-op arm: the script exits `3` and the report blocks
it. For the llama.cpp arm the marker is the per-request `n_past` line; for
apxinf it is the gateway's `request_settled` record in its diagnostic log. ollama's
OpenAI-compatible response carries no durations, so at least one ollama run uses
`--protocol ollama-generate` (S3), which stores ollama's own
`prompt_eval_count`/`eval_count` JSON as the engagement evidence.

## Mixed factors (must be quoted with every result table)

1. **Quantization is not the same between arms.** The three tiers in this
   report are MLX **BF16** (`apxinf-native`), GGUF **Q4_K_M** (`llamacpp`,
   `ollama`, ~4.5-5 bits per weight) and GGUF **F16** (`llamacpp-f16`). An
   absolute tok/s gap between tiers is expected even from two perfect engine
   implementations, so cross-tier rows are context, not evidence. Per the
   pre-registered decision rule, the engine-implementation comparison is the
   16-bit pair (`apxinf-native` BF16 vs `llamacpp-f16`); anything measured only
   across tiers is attributed to the **quantization tier**. The packed4 lane
   (`apxinf-mlx-lane`) would have been the 4-bit apxinf comparison and is
   incomplete in this window (2026-11-18 amendment).
2. **Prompt protocol is not uniform.** S1-S3 use bare completion for llama.cpp
   and ollama. apxinf has no bare-completion route, so its arms go through the
   lane ChatML prefix; their `prompt_tokens` include it and the corpus table
   shows the offset. Long prompts make this negligible, short prompts do not.
3. **Measurement basis.** S1 TTFT and S2 decode are client-observed (SSE first
   and last content delta), which is the only basis available for every arm.
   Engine-reported prefill/decode durations are printed next to them and cover
   different spans per engine: llama.cpp reports prompt-eval and predicted time;
   apxinf reports state reset + capacity preparation + prefill as `prefill_ms`.
   S3 subtracts the engine-reported durations from client wall; where an engine
   publishes no durations, S3 overhead is n/a and wall alone is not serving
   overhead.
4. **One host, one window.** All arms run on the M1 Pro acceptance host in one
   interleaved window with both metal locks held. Absolute numbers do not cross
   windows; comparisons are paired inside a segment.
5. **Concurrency.** All arms are batch-1 serial. llama.cpp/ollama defaults that
   add request batching or prompt caching are disabled or made cache-inert by
   using distinct prompts per repeat; apxinf's serial worker has neither.

## Results

Every table below is produced by `bench_engines_report.py`; empty cells mean
"not measured", not "zero". Arm set in this window: `llamacpp` (Q4_K_M),
`ollama` (Q4_K_M), `apxinf-native` (BF16), `llamacpp-f16` (16-bit tier);
`apxinf-mlx-lane` is incomplete (2026-11-18 amendment).

### Artifact and host identity

| arm | engine version / commit | binary sha256 | weights sha256 | quantization |
|---|---|---|---|---|
| `apxinf-native` | | | | MLX packed4 affine g64 |
| `apxinf-mlx-lane` | | | | MLX packed4 |
| `llamacpp` | | | | GGUF Q4_K_M |
| `ollama` | | | | GGUF Q4_K_M |

Host: M1 Pro, macOS `<version>`, window `<start>`-`<end>`, display asleep,
`caffeinate -si`, both metal locks held; thermal state before/after `<...>`,
swapouts before/after `<...>`, disk free `<...>`.

### A/A (same-arm repeatability)

| arm | stage | segments | worst group spread | A/A |
|---|---|---|---|---|
| | | | | |

### S1 prefill / TTFT (client-observed, paired)

| case | arm | TTFT p50 (ms) | prefill tok/s p50 | engine prefill p50 (ms) | vs `llamacpp` | n | segments |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

### S2 decode (client-observed, paired)

| case | arm | decode ms/token p50 | decode tok/s p50 | engine ms/token p50 | vs `llamacpp` | n | segments |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

### S3 serving-layer overhead

| arm | wall p50 (ms) | TTFT p50 (ms) | overhead p50 (ms) | vs `llamacpp` | n | segments |
|---|---|---|---|---|---|---|
| | | | | | | |

### Corpus / token-count comparability

| stage | case | `apxinf-native` prompt tokens | `llamacpp` prompt tokens | `ollama` prompt tokens |
|---|---|---|---|---|
| | | | | |

## Verdict

Pending stage 2. The verdict block will be written in the form required by
[`skills/perf-verdict-gate`](../skills/perf-verdict-gate/SKILL.md):

```
## <statement, dated YYYY-MM-DD>
Status: KEEP | NOT-WORTH | INCONCLUSIVE | REVERTED
Claim: <one falsifiable sentence>
Evidence: <commit + raw JSON paths>
Gate: <pre-declared gate and the measured value against it>
Arms: <segments, order, paired delta + spread>
Artifact: <commit, binary hash, flags>   Host: <top CPU, swapouts>
Caveats: <what this does not establish>
```

Raw JSON, host state and hashes stay under
`devlocal/engine-comparison-lfm3/` (git-ignored); the report command and the
corpus hash in each file are what make the numbers re-checkable.

## Amendments

None yet. Date any change to the gate, the stage plan, or the corpus here; do not
silently retune the gate after seeing the data.


### 2026-11-18 - precision of the apxinf arms, the 8K row, and the MLX lane (recorded before any stage-2 number)

1. **`apxinf-native` runs BF16, not packed4.** `crates/apxinf-model/src/metal_lfm3/README.md`
   declares "BF16 only; `auto` and `bf16-eager` (the only implemented variant)"; the
   served replay echoes `quant: bf16`; the resident set is 3.46 GB. The packed4
   (affine group 64) tier belongs to the **MLX lane** checkpoint
   (`~/models/LFM3-1.7B-MLX-4bit`, `quantization {group_size: 64, bits: 4, mode: affine}`).
   The design's arm table said the native arm ran packed4; that was wrong and is
   corrected here. Consequence: the quantization-neutral engine comparison is the
   **16-bit tier** (`apxinf-native` BF16 vs `llamacpp-f16`), and the S2 decode gate is
   evaluated on that pair; the 4-bit tier is `llamacpp` vs `ollama`.
2. **The 8K S1 row uses an 8183-token prompt** on every arm (8183 + 8 = 8191, the
   largest request the native family admits), while the exact **8192-token** request
   is sent once per arm so the raw JSON records which engines accept it. The native
   family counts prompt + generation against `MAX_CONTEXT` (8192) and rejects it;
   llama.cpp and ollama serve it with a larger context window. This is a real
   serving-contract difference and is reported as a capability finding.
3. **`apxinf-mlx-lane` is incomplete, not measured.** The serving contract pins
   `mlx-lm 0.31.3`, whose upstream model registry has no `lfm3` model type
   (`ValueError: Model type lfm3 not supported`), and the repo's Python text worker
   rejects the checkpoint's `auto_map` ("requests custom code"). No LFM3 MLX
   integration exists in this repository's serving path, so the packed4 lane could
   not be served through `apxinf-serve` in this window. It is reported as
   incomplete; no 4-bit apxinf comparison is claimed, and the absence is visible
   in the arms table rather than silently dropped.
4. **ollama is measured with `raw: true`** on `/api/generate` (and its
   OpenAI-compatible `/v1/completions` as a control surface), because ollama applies
   the model chat template by default and S1-S3 are bare completion. Without `raw`,
   the same 128-token corpus text is evaluated as 138 prompt tokens (the template
   prefix), which would break the corpus identity check.
