# LFM3-1.7B engine comparison: apxinf vs llama.cpp vs ollama

Status: **method + pre-registered gate; three stage-2 windows complete on the M4
(2026-11-18/19).** Tier A: `llamacpp` vs `ollama` on identical Q4_K_M weights —
every gate met, no measurable serving-layer difference. Tier B cross-precision:
`apxinf-native` BF16 vs `llamacpp` Q4_K_M — every gate met, and the 2.95x decode
gap is explained by the 3.14x weight-byte ratio (**precision** statement).
Tier B matched precision: `apxinf-native` BF16 vs `llamacpp` F16 — every gate met
on the quoted cells; apxinf is **within 2%** of llama.cpp at 128 generated tokens,
which is the pre-registered **"same efficiency class"** verdict for the
engine-implementation comparison.
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

Host amendment 2026-11-18 (stage 2): the M1 Pro is offline. The reported stage-2
tier-A window ran on the **M4 mini** (`haiyan-mac-mini`: Apple M4, 10 cores, 16 GB,
macOS 27.0.1). The earlier M2-Air window is reference-only and void; no number in
this document is compared across hosts.

## Arms

| arm | engine | weights | serving surface | declared prompt protocol |
|---|---|---|---|---|
| `apxinf-native` | apxinf-metal `metal_lfm3` (Rust + Metal) | **MLX BF16** (`bf16-eager`, the only implemented variant) | `apxinf-serve` (`/v1/chat/completions`) | lane ChatML (family-rendered) |
| `apxinf-mlx-lane` | EngineTailor MLX engine + Python worker (current agent stack) | MLX packed4, affine group 64 | `apxinf-serve` (Python provider) | lane ChatML |
| `llamacpp` | llama.cpp `llama-server` | `LFM3-1.7B-Q4_K_M.gguf` | `/v1/completions` | bare completion |
| `ollama` | ollama (llama.cpp runtime + its own scheduler) | the same Q4_K_M GGUF, imported by Modelfile | `/api/generate` with `raw: true` (the M4 stage-2 tier-A surface: template-free bare completion; `/v1/completions` applies the model chat template and is therefore not bare) | bare completion |
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

Client behaviour is fixed and identical for every arm: stdlib `http.client` only,
one fresh connection per request, response always closed, a streamed response
abandoned at `[DONE]` (or ollama's `done: true`), and one retry on a transport
error with the retry count recorded per request. This matters for portability:
reusing a connection after an abandoned SSE response raises
`http.client.ResponseNotReady: Request-sent` on newer CPythons (observed on 3.14
against llama-server). Tested on CPython 3.9 and 3.14; anything older is not
supported. The same client is used for every arm, so the loopback/TCP setup cost
is common mode and does not enter a paired delta.

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
# Fixed-context arm (used by the M4 stage-2 window): a per-request num_ctx makes
# ollama reload its runner for every distinct context and the reload lands inside
# TTFT. Pin one context for the whole arm, matching llama.cpp's -c:
#   scripts/bench_engines.sh --protocol ollama-generate --ollama-num-ctx 16384 ...
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

<!-- BEGIN RESULTS -->

## Stage-2 tier-A result — same weights behind two servers (2026-11-18, M4)

Status: **KEEP** (the measurement stands; no configuration change follows from it).
Claim: with the identical Q4_K_M GGUF in one M4 window, ollama 0.40.16's serving
stack and a source-built llama.cpp `llama-server` are indistinguishable at this
precision and batch-1: paired decode medians agree within 0.4% (12.55 vs 12.51
ms/token at context 1024; 12.82 vs 12.80 at context 4096, 128 generated tokens)
and prefill TTFT within 2% (118.5 / 414.6 / 1533.4 ms vs 120.9 / 417.3 / 1543.4 ms
for 97 / 392 / 1562-token prompts), ollama ~1-2% slower.
Evidence: `devlocal/engine-comparison-lfm3/results/tiera-20261118T151518Z/`
(12 raw JSON files, the report, artifact identity, host state — git-ignored per
`AGENTS.md`); report exit `0`; window 2026-11-18 23:15:27-23:16:31 CST.
Gate: the pre-registered comparability gate passed on **every** group — A/A
`llamacpp` s1 1.4%, s2 0.3%; `ollama` s1 1.8%, s2 1.2% (all <= 5%, three ABBA
segments per arm); a generated-token and engine-log evidence line for both arms;
corpus prompt-token identity exact on all five cases (97/392/1562, 785/3116).
Decision rule: ollama embeds llama.cpp as its runner, so this is a **same-runtime
comparison**: it measures the serving layer and scheduler, never the engine
implementation. No matched-precision pair exists in this window, so the decision
rule's "engine implementation" branch is not taken and must not be quoted as one.
Caveats: batch-1 serial; one host; Q4_K_M only (4-bit); greedy; S1 `max_tokens=8`,
S2 `max_tokens=128`; S3, S4 and S5 were not run in this window, and the 2026-11-26
stage sections below carry them; two different HTTP surfaces (declared
below); ollama `num_ctx` pinned to 16384.
Supersedes: the `m2-quiet` window below (void; retained for history).

### Artifact and host identity

| artifact | value |
|---|---|
| `gguf_sha256` / bytes | `3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4` / 1,099,627,808 |
| `llama_server_sha256` | `d635b19fb11a19d3262ee8544252f20cb92620b58c426b820453bb4f4535e55e` |
| llama.cpp source | ggml-org `llama.cpp` commit `5297534`, built on the M4: `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON`, `SDKROOT=…/MacOSX26.5.sdk` (the 27.0 SDK fails `ld`: `tapi error: malformed file`) |
| llama.cpp version | `version: 1.0.0-dev (build 0)` with AppleClang 21.0.0; `print_info: arch = lfm3`, `file type = Q4_K - Medium`, 24 layers |
| `ollama_binary_sha256` / version | `5c3cf935be8d896433a5caefefc8857bc2bb9e32c77c35e7c3d38b7c5919822f` / `ollama 0.40.16` (official darwin build) |
| ollama model | `lfm3-q4km` from a Modelfile over the same GGUF; its runner loads `/Users/haiyan-mini/.ollama/models/blobs/sha256-3793e1e6…` (byte-identical to the llama.cpp file) |
| host | `haiyan-mac-mini`: Apple M4, 10 cores, 16 GB, macOS 27.0.1 — not the M1 Pro the design was written for |
| window discipline | both EngineTailor Metal locks held by `lockrun_m4.py` (fcntl, waited, never stolen); `pmset displaysleepnow` + `caffeinate -si`; greedy decode (`temperature=0`, fixed seed) |

Host state (recorded before and after by the window script):

| | load average (1/5/15) | swap used / free | Swapouts | busy >= 20% CPU |
|---|---|---|---|---|
| before | 2.55 / 2.57 / 2.69 | 793.69 MB / 1254.31 MB | 45960175 | Logitech GUI agent (paused for the window) |
| after | 1.82 / 2.33 / 2.59 | 785.69 MB / 1262.31 MB | 45960175 | `ollama`'s own runner (the engine under test) |

Swapouts did not move and swap use moved by 8 MB, so the window is not a
contaminated reference-only window (`skills/perf-verdict-gate` rule 2).

### Corpus and prompt-token identity

The same deterministic text was sent to both arms (one distinct text per segment;
identical text for the two arms inside one segment). Prompt tokens are the
engines' own reported counts for byte-identical input:

| stage | case | `llamacpp` | `ollama` |
|---|---|---|---|
| s1 | 128-token target | 97 | 97 |
| s1 | 512-token target | 392 | 392 |
| s1 | 2048-token target | 1562 | 1562 |
| s2 | ctx 1024 / 128 out | 785 | 785 |
| s2 | ctx 4096 / 128 out | 3116 | 3116 |

### S1 prefill / TTFT

Client-observed time to first content delta, engine-reported prefill as the
cross-check; ratio is paired per segment against `llamacpp` (>1 = this arm faster).

| case | arm | TTFT p50 (ms) | prefill tok/s p50 | engine prefill p50 (ms) | vs baseline | n | seg |
|---|---|---|---|---|---|---|---|
| 128-token prompt | `llamacpp` | 118.5 | 821.9 | 117.3 | (baseline) | 3 | 3 |
| 128-token prompt | `ollama` | 120.9 | 802.2 | 118.8 | 0.98x | 3 | 3 |
| 512-token prompt | `llamacpp` | 414.6 | 945.7 | 413.4 | (baseline) | 3 | 3 |
| 512-token prompt | `ollama` | 417.3 | 939.3 | 415.0 | 0.99x | 3 | 3 |
| 2048-token prompt | `llamacpp` | 1533.4 | 1018.6 | 1531.4 | (baseline) | 3 | 3 |
| 2048-token prompt | `ollama` | 1543.4 | 1012.1 | 1540.2 | 0.99x | 3 | 3 |

### S2 decode

Client-observed decode window (first to last content delta) per generated token;
engine-reported ms/token next to it.

| case | arm | decode ms/token p50 | decode tok/s p50 | engine ms/token p50 | vs baseline | n | seg |
|---|---|---|---|---|---|---|---|
| ctx 1024 / 128 out | `llamacpp` | 12.55 | 79.7 | 12.45 | (baseline) | 3 | 3 |
| ctx 1024 / 128 out | `ollama` | 12.51 | 80.0 | 12.41 | 1.00x | 3 | 3 |
| ctx 4096 / 128 out | `llamacpp` | 12.82 | 78.0 | 12.72 | (baseline) | 3 | 3 |
| ctx 4096 / 128 out | `ollama` | 12.80 | 78.1 | 12.70 | 1.00x | 3 | 3 |

### Engine evidence (one distinctive line per engine)

`llamacpp` (its own request log):

```
print_info: arch                  = lfm3
slot print_timing: id 0 | task 10 | prompt eval time =    118.60 ms /    97 tokens (   1.22 ms per token,   817.89 tokens per second)
slot print_timing: id 0 | task 10 |        eval time =     91.31 ms /     8 tokens (  13.04 ms per token,    76.66 tokens per second)
```

`ollama` (its own runner log on the same bytes, and the request that was served):

```
starting llama-server cmd="…/Ollama.app/Contents/Resources/llama-server --model /Users/haiyan-mini/.ollama/models/blobs/sha256-3793e1e6… --c 16384 …"
llama_model_loader: loaded meta data with 45 key-value pairs and 287 tensors from …/blobs/sha256-3793e1e6… (version GGUF V3 (latest))
[GIN] 2026/11/18 - 23:15:31 | 200 | 789.34ms | 127.0.0.1 | POST     "/api/generate"
```

### Mixed factors (quote with the tables)

1. **Precision is uniform**: both arms read the same Q4_K_M file; the hash above is
   the same bytes for both.
2. **HTTP surface differs, prompt text does not.** `llamacpp` is measured on
   `/v1/completions`; `ollama` on `/api/generate` with `raw: true`, because
   ollama's `/v1/completions` applies the model chat template — the same 5-token
   text comes back as `prompt_tokens: 15` there, and the completion starts as a
   chat answer. The raw surface reports `prompt_tokens: 5` for the same text and
   exposes ollama's own `prompt_eval_duration`/`eval_duration`. This is why both
   engines can report identical prompt-token counts.
3. **Context pinning.** ollama is given the same `num_ctx 16384` on every request
   (new harness flag `--ollama-num-ctx 16384`) as llama.cpp's `-c 16384`. Without
   the pin, ollama reloads its runner for each distinct context and the reload
   lands inside TTFT (one `starting llama-server` line per request was observed).
4. **Corpus is distinct per segment.** One repeat per segment would otherwise
   replay the identical prompt three times; llama.cpp's slot prefix cache then
   serves the 2nd/3rd prefill from KV (`prompt eval time = 21.02 ms / 4 tokens`
   for a 1562-token prompt). Each segment therefore uses its own corpus seed, the
   two arms in a segment sharing that seed.
5. **Measurement base.** TTFT/decode are client-observed (first/last SSE content
   delta) for both arms; the engine-reported durations are printed next to them
   and cover similar spans here (llama.cpp prompt-eval; ollama prompt-eval).
6. **One host, one window, both engines resident.** No absolute number crosses a
   window or a host. Batch-1 serial, one request at a time.

### Void windows and their fixes (kept visible)

Three earlier windows in this campaign are void and are **not** part of the data
set above. They are recorded because each one looked plausible until its own A/A
gate was applied (the git-ignored ledger keeps the full detail):

| window | what looked wrong | A/A | fix applied before the next window |
|---|---|---|---|
| `tiera-20261118T145912Z` | identical prompt text in all three segments + `repeats=1`; llama.cpp's prefix cache answered the 2nd/3rd prefill from KV (`prompt eval time / 4 tokens` for a 1562-token prompt) | `llamacpp` s1 **8162.5%**; `ollama` s1 8.3% | per-segment corpus seed |
| `tiera-20261118T150153Z` | ollama ran `/api/generate` with a per-request `num_ctx`, so it restarted its runner for every request (19 `starting llama-server` lines in one window) and the reload landed inside TTFT | `ollama` s1 **5.9%** | `--ollama-num-ctx 16384` (one runner, fixed context) |
| `tiera-20261118T150358Z` | protocol clean, but one 2048-token prefill outlier: a 10-repeat probe of that exact case spanned 1539-1645 ms (**6.4%**), i.e. that group could not pass a 5% gate by chance | `ollama` s1 **7.0%** | pause the user's Logitech GUI helper processes (`SIGSTOP`, PIDs recorded, `SIGCONT` in teardown): the same probe spans 1539-1570 ms (**2.0%**) |

The desktop-agent pause is a **measurement-condition change**, not a corpus or
gate change: it is stated here, the resumed PIDs were verified after the window,
and the reported window's host table is the state with those agents paused.

### Efficiency arithmetic (not a cross-host speed ranking)

The decode arm streams 1.0996 GB of weights per generated token, so 12.55 ms/token
is ≈88 GB/s effective. The campaign's packed4 MLX lane measured 11.8 ms/token on
the M1 Pro with a 0.973 GB checkpoint ≈82 GB/s by the same arithmetic (≈79 GB/s on
the campaign's own effective-byte count). On two different hosts, that puts both
engines in the same ~80 GB/s **bandwidth-efficiency class** for LFM3 at 4 bits —
consistent with batch-1 decode being weight-stream bound. This is an arithmetic
ratio between the checkpoint's bytes and the measured wall, **not** a measured
memory ceiling and **not** a speed comparison between the M1 Pro and the M4.

### Tier B — `apxinf-native`: builds, but the arm is still not measurable (2026-11-18)

> **Superseded 2026-11-18 23:51 CST** by the cross-precision section below: PR #118
> fixed the truncated-structured-block rejection, `apxinf-serve` was rebuilt from
> `fbd1e2e9`, and the arm now runs (report exit 0). The text below is retained
> because it is the exact failure that blocked the previous window.


`apxinf-serve` does build on the M4 from `feat/lfm3-metal` (source commit
`5104123b`: `cargo build --release -p apxinf-serving --features native-minicpm5
--bins`, `SDKROOT=…/MacOSX26.5.sdk`, `MLX_ROOT=~/mlx-sdk`), and its native worker
loads the reconstructed BF16 checkpoint (`model.safetensors` 3,457,635,448 bytes,
sha256 `135922ca…`, `Metal 0: Apple M4 (applegpu_g16g)`, 3.46 GB resident). The arm
is still not measurable, and the blocker is the serving contract, not the build:

- Every harness request (S1 `max_tokens=8`, S2 `max_tokens=128`, the word-salad
  corpus) is answered with
  `{"type":"error","error":{"type":"api_error","code":"internal_error","message":"The model output contains an incomplete structured block."}}`
  while the worker itself completed it (`native_request_settled` … `output_tokens: 8`,
  `ttft_ns` 48 ms, 364 ms elapsed). The family emits a `<think>` block as its first
  token; a short `max_tokens` truncates it and the gateway's structured-block
  validation turns a completed generation into `internal_error`. A direct question
  with `max_tokens=120` succeeds (57 completion tokens, `finish_reason: stop`,
  "The capital of France is **Paris**."), so this is protocol/content dependent,
  not a load or build failure.
- Consequence: there are **no tier-B numbers** in this document and the decision
  rule's matched-precision pair still does not exist. A fix belongs to the server
  (allow a truncated structured block when `finish_reason=length`) or to a
  benchmark protocol that never truncates structured output.
- Matched-precision path: `convert_hf_to_gguf.py` at llama.cpp commit `5297534` has
  **no `lfm3` support** (zero case-insensitive `lfm` matches), so the BF16
  checkpoint cannot be converted to an F16 GGUF locally; the pair needs the
  official `LFM3-1.7B-F16.gguf` (3.47 GB). Even then, the cross-tier caveat
  applies: `apxinf-native` BF16 (3.46 GB) vs llama.cpp Q4_K_M (1.10 GB) differ
  ~3x in precision, so a delta across them is a **precision statement, not an
  engine statement**; only BF16-vs-F16 would be an engine-implementation
  comparison.



## Stage-2 tier B — cross-precision: `apxinf-native` BF16 vs `llamacpp` Q4_K_M (2026-11-18, M4)

Status: **KEEP** (measurement accepted) — this is a **precision statement, not an
engine statement**. The matched BF16-vs-F16 pair that the decision rule needs is
measured in the last subsection of this section ("same efficiency class, apxinf
within 2%").
Claim: in one ABBA window on the M4, apxinf's native `metal_lfm3` path at MLX BF16
decodes at **36.60 / 36.93 ms per token** (27.3 / 27.1 tok/s) while llama.cpp at
Q4_K_M does **12.39 / 12.71 ms** (80.7 / 78.7 tok/s): apxinf is **2.95x slower per
token**. That gap is explained by the **3.14x weight-byte ratio** (3.458 GB BF16
vs 1.0996 GB Q4_K_M): effective streaming bandwidth is **94.5 vs 88.7 GB/s**, i.e.
the same class. Prefill is at parity: apxinf 142.7 / 430.9 / 1462.8 ms vs
llama.cpp 116.5 / 412.6 / 1531.2 ms for 106/401/1571 vs 97/392/1562 prompt tokens
(apxinf 17% slower at 128 tokens, 4% slower at 512, 4.5% **faster** at 2048).
Evidence: `devlocal/engine-comparison-lfm3/results/tierb-20261118T155745Z/`
(12 raw JSON, report, identity, request log, host state); report **exit 0**;
window 2026-11-18 23:58:13-00:00:31 CST.
Gate: A/A `apxinf-native` s1 **2.7%**, s2 **0.4%**; `llamacpp` s1 **2.1%**, s2
**0.9%** (all <= 5%, three ABBA segments per arm); engine-log evidence on both
arms; matched spans declared below.
Decision rule: cross-tier by construction, so **no engine-implementation
attribution** is made. The pre-registered matched-precision decision is applied
only to the BF16-vs-F16 pair (last subsection).
Caveats: engine-reported timing basis; +9-token ChatML prefix on the apxinf arm;
apxinf `prefill_ms` includes state reset + capacity preparation; one host, one
window; batch-1; Q4_K_M vs BF16 only.
Supersedes: the `### Tier B — apxinf-native: builds, but the arm is still not
measurable` subsection above (the truncated-structured-block bug it reports was
fixed by PR #118; that text stays visible, this section is the measurement).

### What unblocked the arm

`apxinf-serve` was rebuilt on the M4 from `feat/lfm3-metal` tip **`fbd1e2e9`**
(PR #118; source tarball of that tree sha256 `da5b6cb9…`; `cargo build --release -p
apxinf-serving --features native-minicpm5 --bins`, `SDKROOT=…/MacOSX26.5.sdk`,
`MLX_ROOT=~/mlx-sdk`): `apxinf-serve` sha256 `f3ff54661cc8565d183038eb725c9dcdc89dc1f9a9722562241f29b189e75f24`,
`apxinf-native-worker` sha256 `04f4692a4d0bb4923dac6765c0bb87b2fa4709c2d184987e55fae727ca4633a5`.
Sanity check on the exact failure shape of the previous window (word-salad
prompt, `max_tokens=8`, streaming): **HTTP 200**, one content chunk, then
`finish_reason: "length"` and `[DONE]`; the request log records
`"status":"completed","cause":"length", "completion_tokens":8`.

### Artifact and host identity

| artifact | value |
|---|---|
| `apxinf-serve` / worker sha256 | `f3ff5466…` / `04f4692a…` (built on the M4 at source tip `fbd1e2e9`) |
| apxinf weights | reconstructed BF16 HF checkpoint, `model.safetensors` 3,457,635,448 B, sha256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`; served identity `quant: bf16`, `ane: off` |
| `llama_server_sha256` | `d635b19fb11a19d3262ee8544252f20cb92620b58c426b820453bb4f4535e55e` (llama.cpp `5297534`, Metal) |
| `gguf_sha256` / bytes | `3793e1e6…` / 1,099,627,808 |
| arch proof | llama.cpp `print_info: arch = lfm3`; apxinf `Metal 0: Apple M4 (applegpu_g16g)` + 3.46 GB resident |
| window discipline | both Metal locks via `lockrun_m4.py`; `pmset displaysleepnow` + `caffeinate -si`; host GUI agents paused (same conditioning as tier A, PIDs recorded) |

Host state: load **2.73/2.94/2.70 -> 2.15/2.57/2.58**; swap used **649.7 -> 609.7 MB**;
`Swapouts` **46055875** before and after (unchanged), so the window did not move
materially.

### Corpus and prompt-token identity

| stage | case | `apxinf-native` (chat, +9 prefix) | `llamacpp` (bare) |
|---|---|---|---|
| s1 | 128-token target | 106 | 97 |
| s1 | 512-token target | 401 | 392 |
| s1 | 2048-token target | 1571 | 1562 |
| s2 | ctx 1024 / 128 out | 794 | 785 |
| s2 | ctx 4096 / 128 out | 3125 | 3116 |

The offset is exactly the lane ChatML prefix every apxinf text request carries;
it is a declared mixed factor, not a corpus mismatch (bare-completion arms agree
exactly with each other, as the tier-A window showed).

### S1 prefill (engine-reported, `--engine-timings`)

| case | arm | prefill p50 (ms) | prefill tok/s p50 | vs `llamacpp` | n | seg |
|---|---|---|---|---|---|---|
| 128-token prompt | `apxinf-native` | 142.7 | 745.0 | 0.83x | 3 | 3 |
| 128-token prompt | `llamacpp` | 116.5 | 836.2 | (baseline) | 3 | 3 |
| 512-token prompt | `apxinf-native` | 430.9 | 937.8 | 0.96x | 3 | 3 |
| 512-token prompt | `llamacpp` | 412.6 | 950.0 | (baseline) | 3 | 3 |
| 2048-token prompt | `apxinf-native` | 1462.8 | 1078.4 | 1.05x | 3 | 3 |
| 2048-token prompt | `llamacpp` | 1531.2 | 1020.8 | (baseline) | 3 | 3 |

### S2 decode (engine-reported, `--engine-timings`)

| case | arm | decode ms/token p50 | decode tok/s p50 | weight bytes / token | effective GB/s | vs `llamacpp` | n | seg |
|---|---|---|---|---|---|---|---|---|
| ctx 1024 / 128 out | `apxinf-native` | 36.60 | 27.3 | 3.458 GB (BF16) | ≈94.5 | 0.34x | 3 | 3 |
| ctx 1024 / 128 out | `llamacpp` | 12.39 | 80.7 | 1.0996 GB (Q4_K_M) | ≈88.7 | (baseline) | 3 | 3 |
| ctx 4096 / 128 out | `apxinf-native` | 36.93 | 27.1 | 3.458 GB (BF16) | ≈93.6 | 0.34x | 3 | 3 |
| ctx 4096 / 128 out | `llamacpp` | 12.71 | 78.7 | 1.0996 GB (Q4_K_M) | ≈86.5 | (baseline) | 3 | 3 |

Attribution arithmetic (the point of labelling this a precision statement):
weight-byte ratio 3.458 / 1.0996 = **3.14x**; measured decode-time ratio is
36.60 / 12.39 = **2.95x** at ctx 1024 and 36.93 / 12.71 = **2.91x** at ctx 4096.
The BF16 arm therefore streams at *slightly higher* effective bandwidth than the
Q4 arm (94.5 vs 88.7 GB/s). A 2.95x decode gap whose byte ratio is 3.14x is what
a weight-stream-bound pair at different precisions looks like; it is **not**
evidence that apxinf's engine is 3x slower. Likewise the prefill near-parity
(apxinf 4.5% faster at 2048 tokens) cannot be read as an engine win, because
apxinf's `prefill_ms` spans state reset + capacity preparation + prefill while
llama.cpp's `prompt_ms` is prompt evaluation only — the wider span is the
conservative one.

### Why the client-observed basis is unusable on this pair

`apxinf-serve` buffers its streamed answer: the harness observed
`client_ttft_ms` ≈ `client_wall_ms` (7742 ms against an engine `prefill_ms` of
783 ms) and a client "decode window" of **0.0004 ms**, i.e. every content delta
arrives at completion. The window therefore ran with the new
`--engine-timings` harness flag for **both** arms, and each raw file records
`"timing_primary": "engine"`; the client-observed values are preserved as
`client_*_ms_observed`. This is a real serving-contract difference (a client
cannot stream apxinf output token-by-token) and is recorded as such; it is why
the per-engine durations above, not TTFT, are the comparable metric.

### Mixed factors (quote with the tables)

1. **Precision differs: BF16 (3.458 GB/token) vs Q4_K_M (1.0996 GB/token), 3.14x
   in weight bytes.** Any decode delta here is a precision statement.
2. **Prompt protocol differs**: apxinf is chat-only (lane ChatML prefix, +9
   prompt tokens on every case); llama.cpp is bare completion.
3. **Timing basis is engine-reported for both arms**, with declared span
   differences (llama.cpp prompt-eval/predicted; apxinf prefill incl. state
   preparation, decode). Neither number is a client-observed TTFT.
4. **One host, one window**, batch-1 serial, greedy, 3 ABBA segments per arm.

### Matched-precision result: `apxinf-native` BF16 vs `llamacpp` F16 (2026-11-19, M4)

> **Amended 2026-11-19 05:19 CST:** the pair is measured. The pre-registered
> protocol paragraph below is kept verbatim; the tables after it are window
> `f16-20261118T211938Z` (report exit 0). The earlier hand-processed run
> `f16-20261118T194957Z` is cited only where it agrees, as a reproduction check.

Status: **KEEP** — this is the engine-implementation comparison the decision rule
applies to.
Claim: at 16-bit on both sides and batch-1 on one M4 window, `apxinf-native` (MLX
BF16) and `llamacpp-f16` (GGUF F16) decode LFM3 at the **same rate**: 37.20 vs
36.42 ms/token at context 1024 and 37.50 vs 36.67 ms/token at context 4096
(128 generated tokens), i.e. **apxinf within 2% of llama.cpp** — apxinf/llamacpp
= 1.021 / 1.023, inside the pre-registered 0.8-1.25x "same efficiency class"
band. Effective bandwidth: 92.9 / 92.2 GB/s (apxinf) vs 95.2 / 94.5 GB/s
(llama.cpp). Prefill (the pre-registered S1 metric): apxinf 11.2% slower at 128
prompt tokens, 8.6% slower at 512, 0.6% slower at 2048.
Evidence: `devlocal/engine-comparison-lfm3/results/f16-20261118T211938Z/`
(12 raw JSON, report, identity, request ledger, host state); report **exit 0**;
window 2026-11-19 05:19:38-05:23:01 CST.
Gate: per-cell A/A is in every table below; every quoted cell is <= 5%. The
request ledger joined **15/15** apxinf requests by `request_id`
(`request_log: {matched: 15, by_request_id: 15, by_order: 0}`), so the apxinf
timings are the engine's own, not a fallback guess.
Decision rule: apxinf/llamacpp decode ratio 1.021 / 1.023 lies in the 0.8-1.25x
band → **"same efficiency class"**; no "apxinf decode lags" verdict, and no win is
claimed.
Caveats: engine-reported timing basis on both arms (see mixed factor 3); +9-token
ChatML prefix on the apxinf arm; S1 decode-at-8 excluded (below); one host, one
window; batch-1; greedy.

> Pre-registered protocol (from the campaign design, unchanged):
>
> ```
> Gate: same-arm repeat (A/A) <= 5%, engine-reported basis for both arms,
>       matched spans declared.
> Decision: apxinf decode < 0.8 x llamacpp  -> "apxinf decode lags" (no softening);
>           0.8-1.25x                       -> "same efficiency class";
>           > 1.25x                         -> report the win with the effective
>                                              GB/s arithmetic for both arms.
> ```

**Arms.** `llamacpp-f16`: the official `LFM3-1.7B-F16.gguf`, GGUF F16, bare
completion on `/v1/completions`. `apxinf-native`: the reconstructed BF16 HF
checkpoint served natively on Metal, chat surface (lane ChatML), engine-reported
durations. Both are 16-bit weights of the same model; this is the pair the
decision rule admits.

### Identity block

| item | value |
|---|---|
| llama.cpp weights | `LFM3-1.7B-F16.gguf`, **3,466,626,336 B**, sha256 `b7ab8b69a85e4dfbf62cdfd0e02d7d051ed09f2c5b84c701771dddb073b85970` (equals the upstream HF LFS hash) |
| apxinf weights | reconstructed BF16 `model.safetensors`, **3,457,635,448 B**, sha256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330` |
| `llama_server_sha256` | `d635b19fb11a19d3262ee8544252f20cb92620b58c426b820453bb4f4535e55e` (llama.cpp `5297534`, Metal, SDK 26.5) |
| `apxinf_serve_sha256` | `f3ff54661cc8565d183038eb725c9dcdc89dc1f9a9722562241f29b189e75f24` (source tip `fbd1e2e9`, includes #118) |
| `apxinf_native_worker_sha256` | `04f4692a4d0bb4923dac6765c0bb87b2fa4709c2d184987e55fae727ca4633a5` |
| arch proof | llama.cpp `print_info: arch = lfm3`; apxinf `Metal 0: Apple M4 (applegpu_g16g)`, `quant: bf16` |

Host state: load **2.64/2.51/2.64 -> 2.90/2.72/2.70**; swap used
**901.4 -> 893.4 MB**; `Swapouts` **46284047** before and after (unchanged).
Two system daemons (`mediaanalysisd`, `spotlightknowledged`) appeared in the top-CPU
list during the window; A/A stayed clean (<= 3.5% per cell, below), so the window
is kept, with that recorded as a caveat rather than silently dropped.

### S2 decode — engine-reported, matched precision

| case | arm | decode ms/token p50 | decode tok/s p50 | A/A (this cell) | weight bytes / token | effective GB/s | apxinf/llamacpp |
|---|---|---|---|---|---|---|---|
| ctx 1024 / 128 out | `apxinf-native` | 37.20 | 26.9 | 3.3% | 3.4576 GB (BF16) | ≈92.9 | **1.021x** |
| ctx 1024 / 128 out | `llamacpp-f16` | 36.42 | 27.5 | 2.4% | 3.4666 GB (F16) | ≈95.2 | (baseline) |
| ctx 4096 / 128 out | `apxinf-native` | 37.50 | 26.7 | 0.9% | 3.4576 GB (BF16) | ≈92.2 | **1.023x** |
| ctx 4096 / 128 out | `llamacpp-f16` | 36.67 | 27.3 | 1.0% | 3.4666 GB (F16) | ≈94.5 | (baseline) |

The two arms stream the same number of bytes (0.26% apart, container difference)
at 92-95 GB/s effective, so the 2% difference is the implementation, and it is
below the pre-registered "same efficiency class" boundary in both directions.

### S1 prefill — engine-reported (supporting data, not part of the decode verdict)

| case | arm | prefill p50 (ms) | prefill tok/s p50 | A/A (this cell) | apxinf/llamacpp |
|---|---|---|---|---|---|
| 128-token prompt | `apxinf-native` | 141.8 | 747.8 | 1.9% | 1.112x |
| 128-token prompt | `llamacpp-f16` | 127.4 | 761.1 | 3.5% | (baseline) |
| 512-token prompt | `apxinf-native` | 435.1 | 921.7 | 1.9% | 1.086x |
| 512-token prompt | `llamacpp-f16` | 400.4 | 978.9 | 1.0% | (baseline) |
| 2048-token prompt | `apxinf-native` | 1461.5 | 1074.3 | 0.8% | 1.006x |
| 2048-token prompt | `llamacpp-f16` | 1453.1 | 1074.3 | 0.4% | (baseline) |

### S1 cells excluded as non-quotable

`max_tokens=8` gives a seven-interval generation window, and the apxinf arm's
same-arm spread on that window is not certifiable: the 512-token cell is
**5.02% A/A** (over the pre-registered 5% gate: per-segment medians
37.855 / 38.349 / 39.781 ms/token), and the earlier hand-processed run measured
**9.7-9.9%** on the same cells. **No S1 decode cell is quoted as a result**, and
the hand-processed s1 rows are not carried into this document. The S1 *prefill*
table above is the pre-registered S1 metric and is quotable (all cells
0.4-3.5%); it is supporting data only, because the verdict is defined on decode.

### Mixed factors (quote with the tables)

1. **Matched precision, not identical bytes.** Both arms are 16-bit, but one is
   GGUF F16 (3,466,626,336 B) and the other MLX BF16 (3,457,635,448 B): 0.26% more
   bytes on the llama.cpp side from container layout. Weights and precision are
   matched for the purpose of the decision rule.
2. **Prompt protocol.** `apxinf-native` is chat-only: every case carries the lane
   ChatML prefix, **+9 prompt tokens** (106/401/1571 vs 97/392/1562 prefill;
   794/3125 vs 785/3116 decode).
3. **Timing basis is engine-reported for both arms** because `apxinf-serve`
   buffers its SSE (issue [#120](https://github.com/qhy991/apxinf-metal/issues/120):
   client TTFT 7.7 s against engine prefill 0.78 s, client decode window
   ~0.0004 ms). Raw files record `timing_primary: "engine"` and keep
   `client_*_ms_observed`; spans are declared (llama.cpp prompt-eval/predicted;
   apxinf prefill incl. state preparation, decode).
4. **One host, one window**, batch-1 serial, greedy, three ABBA segments per arm,
   both Metal locks held, host GUI agents paused as in tier A.
5. **Reproduction against the earlier hand-processed run
   `f16-20261118T194957Z`:** s2 ctx1024 36.78 vs 36.42 (llama) and 37.45 vs 37.20
   (apxinf); ctx4096 37.13 vs 36.67 and 37.80 vs 37.50; s1 short-gen
   34.3-34.6 vs 34.9-35.5 (llama) and 37.1-38.2 vs 38.1-39.2 (apxinf) — every
   number reproduces within ~1-2%. The one value that moved is the apxinf s1 A/A:
   9.7-9.9% in the hand run (its 512-token prefill had one 505.7 ms segment against
   ~431 ms on the others) versus 1.8-5.0% here; that is window noise, not a
   systematic difference, and the certified s1 decode still stays excluded as
   above.

## Stage-2 S3 serving-layer overhead (2026-11-26, M4)

S3 had no data in this document. It now has three windows on the same M4, both Metal
locks held and host state recorded before and after each: one per engine pair, and a
repeat of the second pair on the other timing basis. The stage metric is client
`wall_ms`, decomposed as `wall_ms` minus the engine's own reported prefill and decode.

### Same weights behind two servers (`llamacpp` against `ollama`)

Status: **KEEP**. The measurement stands and no configuration change follows from it.

Claim: on the S3 primary metric these two serving layers are indistinguishable, and on
the residual the stage exists to expose they separate reproducibly.

On client wall, the S3 metric, 150 requests per arm across three ABBA segments gave
216.95 ms for `llamacpp` against 217.46 ms for `ollama`, a paired ratio of 0.998. The
gap is 0.5 ms, or 0.23% of the request, an order of magnitude below the window's own
A/A floor of 1.5% to 2.2%, so it is reported as no measurable difference and not as a
ranking.

On the residual, client wall minus each engine's own reported prefill and decode, the
arms separate and stay separated. Per segment `llamacpp` is 1.09 / 1.10 / 1.19 ms and
`ollama` is 2.17 / 2.31 / 2.32 ms. The ranges do not overlap and the split holds in all
three segments, so `ollama`'s serving path costs about 1.1 ms more per request than a
directly run `llama-server`. That is 0.5% of a 217 ms request. It is read as a
serving-path difference between these two binaries, because both spans are
self-reported, and not as a property of either engine family.

Limiter: the request is bound by the model, not the serving layer. The S3 case is a
128-token target prompt that the engines report as 90 to 105 prompt tokens, median 98,
plus an 8-token generation. Prefill p50 is 119.4 ms for `llamacpp` and 119.6 ms for
`ollama`, against 118.5 ms for the published 97-token S1 point. Engine-reported decode
is 11.99 and 11.86 ms/token against the published 12.45 ms/token at context 1024. Both
agree with tier A within about 4%, so the 217 ms wall is 119 ms of prefill and 96 ms of
decode, and the residual above is the only part left for a serving layer to explain.

| case | arm | wall p50 (ms) | TTFT p50 (ms) | overhead p50 (ms) | vs baseline | n | seg |
|---|---|---|---|---|---|---|---|
| 128 tok in / 8 tok out (x N) | `llamacpp` | 217.0 | 119.4 | 1.1 | (baseline / unpaired) | 150 | 3 |
| 128 tok in / 8 tok out (x N) | `ollama` | 217.5 | 119.6 | 2.2 | 1.00x | 150 | 3 |

Evidence: `devlocal/engine-comparison-lfm3/results/s3-20261126T141306Z/` holds the 6
raw JSON files, the report, the summary, artifact identity and host state before and
after; report exit `0`; window 2026-11-26 22:13:06 to 22:14:26 CST.
Gate: passed on every arm. A/A `llamacpp` s3 2.22%, `ollama` s3 1.52%, both under the
5% ceiling across three ABBA segments per arm; 150 of 150 requests generated on both
arms with an engine-log evidence line each; no failed request; corpus prompt-token
identity exact.

### Cross-precision pair (`apxinf-native` against `llamacpp`)

Status: **MEASUREMENT**. The residual is reproducible; what it is made of is not
identified, and no attribution follows from this window.

Claim: on this request shape `apxinf-native` takes 710.4 ms of client wall against
`llamacpp`'s 212.0 ms, and 310.2 ms of apxinf's request sits outside apxinf's own
reported prefill and decode, against 0.9 ms for `llamacpp`.

This is a cross-tier and cross-protocol pair, and no part of the ratio above is an
engine attribution. `apxinf-native` serves MLX BF16 through `/v1/chat/completions`
while `llamacpp` serves GGUF Q4_K_M through `/v1/completions`, and the pre-registered
decision rule permits an engine-implementation claim only when the 16-bit arms
reproduce its sign. The per-arm residual is far less exposed to that confound than the
ratio is, because each arm's own reported prefill and decode are subtracted from its
own wall, and that is why this stage was worth running for apxinf while tier B's S1 and
S2 attribution is not available.

The 0.30 wall ratio is not the finding. It is what the published tier-B arithmetic
already predicts, since BF16 decode against Q4_K_M decode is a weight-stream-bound pair
at different precisions. Engine-reported decode here is 32.49 ms/token for
`apxinf-native` against 11.65 for `llamacpp`, a 2.79x gap against the published 2.94x
at context 1024 and against a weight-byte ratio of 3.14x. The finding is the residual,
which is 44% of apxinf's wall and has no counterpart on the other arm.

What the residual is made of is not identified. A per-request IPC round trip to the
separate `apxinf-native-worker` process is one candidate, since the window's host
snapshot shows that worker alive, and a per-request serve-layer setup that apxinf does
not count in its reported spans is another. Nothing in this window distinguishes them,
so the residual is recorded as a reproducible finding that needs its own
instrumentation, not as an explanation.

| case | arm | wall p50 (ms) | TTFT p50 (ms) | overhead p50 (ms) | vs baseline | n | seg |
|---|---|---|---|---|---|---|---|
| 128 tok in / 8 tok out (x N) | `apxinf-native` | 710.4 | 138.4 | 310.2 | 0.30x | 150 | 3 |
| 128 tok in / 8 tok out (x N) | `llamacpp` | 212.0 | 117.8 | 0.9 | (baseline / unpaired) | 150 | 3 |

Corpus comparability: `apxinf-native` reports 107 prompt tokens against `llamacpp`'s 98
for the same text, the declared lane-ChatML prefix offset.

Evidence: `devlocal/engine-comparison-lfm3/results/s3c-20261126T142007Z/` holds the 6
raw JSON files, the report, the summary, artifact identity and host state; report exit
`0`; window 2026-11-26 22:20:07 to 22:23:02 CST. Window
`devlocal/engine-comparison-lfm3/results/s3b-20261126T141608Z/` is the same measurement
on the engine-reported basis and reproduces it, with `apxinf-native` at 708.8 ms wall
and 309.76 ms residual and `llamacpp` at 213.5 ms and 0.96 ms.
Gate: passed on every arm. A/A `apxinf-native` s3 0.7%, `llamacpp` s3 2.6%, both under
the 5% ceiling across three ABBA segments per arm; 150 of 150 requests generated on both
arms with an engine-log evidence line each; no failed request.

Caveats: batch-1 serial; one host; greedy; `max_tokens=8`; non-streaming, so `wall_ms`
is client-observed while prefill and decode are engine-reported, which is what the S3
definition asks for; two different HTTP surfaces and two different weight formats, both
declared; the residual compares two engines' self-reported spans, so it is a
serving-path difference between these two binaries.

## Stage-2 S4 end-to-end agent suite (2026-11-26, M4)

Status: **BLOCKED**. No S4 number is claimed and the pre-registered S4 gate is not
applied to anything.

The 12-task suite is frozen in the ANEWeave repository at
`receipts/AG2-m1/suite/AG2-suite-v1.json`, sha256
`660ad1c07c4f111ed1879c12eec9260d45547068951718d2c889a11f1bb75722`: 12 tasks,
`math_tool` 6, `code` 4, `math_single` 2. The runner that drives it,
`receipts/AG2-m1/harness/ag2_run.py`, cannot run on the M4 host as it stands.

It starts its own gateway rather than accepting an endpoint. `ag2_run.py:184-202`
builds a gateway command and `Popen`s it, and its required argument surface
(`ag2_run.py:433-441`) is `--gateway-bin`, `--worker`, `--engine-root`, `--model-root`,
`--mini-bin`, `--mini-config`, `--harness-venv` and `--uv-python`. There is no flag
that points any of it at an existing server. Its worker `lfm3_worker.py` is not present
on the M4, and its host wiring still names M1 Pro paths (`ag-run/harness/in_lock.sh:8-14`
points at `/Users/haiyan/...` trees). Its bash agent is `mini-swe-agent` 2.4.6, which
the ANEWeave record pins in `receipts/AG0-m1/mini-swe-agent-freeze.txt` but which is
installed nowhere on the M4: `pip show mini-swe-agent` finds nothing, no `minisweagent`
module imports, and no `mini` entrypoint exists under any of the host's virtualenvs.

One slice is engine-agnostic and reachable. `receipts/eval/agents/code_loop.py` takes an
arbitrary base URL through `--gateway` and posts only to `/v1/chat/completions`
(`code_loop.py:219` and `:523`), and it filters the suite to `kind == "code"` at
`:576-577`. That is 4 of the 12 tasks. Running it would give a code-task solved set for
two engines and say nothing about the other 8, so it is not the stage and it is not
reported as one.

What S4 needs, exactly, is one of two things. Either install `mini-swe-agent==2.4.6`
with its pinned dependency set on the M4 host, or edit `ag2_run.py` so a base URL can be
supplied instead of starting the AG1 gateway, and supply the 2 `math_single` tasks from
a short script of their own. The second path needs no install and no M1 host, and it is
the one this campaign would take.

## Stage-2 S5 quantized-tier agreement (2026-11-26, M4)

Status: **MEASUREMENT**, with its scope stated. Two arms are measured and one is
blocked. The half of this stage that reads solved-set differences is not obtainable in
this window, and the reason is below rather than left implied.

The stage compares an arm's own 4-bit and 16-bit tier on the same weights, greedy, seed
42, `max_tokens=64`, bare completion, one tier at a time, since two 16-bit models plus
two 4-bit models at once do not fit the 16 GB host. Agreement is the longest common
token prefix of the two tiers' output over the longer of the two outputs: a 4-bit and a
16-bit tier are expected to agree for a while and then diverge, and the useful quantity
is how long they stay together and how often they never diverge at all.

Claim 1. Within one engine the two tiers agree for a few tokens and then separate.

Claim 2. The number does not depend on which prompt set is used. Two sets, frozen
independently before either was measured, give the same answer.

Claim 3. The one prompt in the natural set that is a true continuation prefix rather
than a question is the one prompt where the tiers stay together, which is what the
mechanism in Claim 1 predicts.

Claim 4. This is a property of the tier and not of the engine. `llamacpp` and `ollama`
serve the same Q4_K_M GGUF and the same F16 GGUF, and the two engines' outputs are
byte-identical within each tier on every prompt in both sets, so the agreement tables
below are identical for the two arms to four decimals.

### Prompt set A, the campaign's generated corpus

This is the pre-registered set, the "fixed 16 prompts" of the Stages section above. 16
prompts from the same deterministic generator the S1-S3 corpus uses, at a distinct seed
and distinct case ids, sha256 `c7945f1e…`. The text is content-neutral filler of varying
length, 24 to 640 target tokens.

| arm | tiers | prompts | mean agreement | median | identical |
|---|---|---|---|---|---|
| `llamacpp` | f16 vs q4_k_m | 16 | 0.1135 | 0.0156 | 1/16 |
| `ollama` | f16 vs q4_k_m | 16 | 0.1135 | 0.0156 | 1/16 |

### Prompt set B, the ANEWeave task suite

12 task problems from `AG2-suite-v1.json`, sha256 `660ad1c0…`, frozen as prompt set
`041283c0…`. These are well-posed questions and code tasks, 22 to 493 characters.

| arm | tiers | prompts | mean agreement | median | identical |
|---|---|---|---|---|---|
| `llamacpp` | f16 vs q4_k_m | 12 | 0.1276 | 0.0156 | 0/12 |
| `ollama` | f16 vs q4_k_m | 12 | 0.1276 | 0.0156 | 0/12 |

Set B was run to test a specific worry about set A: that a content-free prompt gives the
model a flat next-token distribution, where a small logit change flips the argmax, so
set A would report a worst case and overstate the divergence. **Set B does not rescue
the number.** Its mean is 0.128 against 0.114 and its median is identical at 0.016, so
the divergence is a property of the tiers and not of the prompt set.

The per-prompt detail says why both sets read the same way. In set B the one prompt that
is a real continuation context, `humaneval_0080`, whose text ends on a closed docstring
and a newline, has both tiers emit the same first 6 tokens and agree for 56 of 64, an
agreement of 0.875. The other 11 prompts state a task rather than continuing one, either
a word problem, an imperative such as `Write a function to check if the given number is
woodball or not.`, or a direct question such as `Simplify $\sqrt{9^3}$.`, and a bare
completion of any of those continues the prompt text rather than answering it. At that
point neither tier's output is gradeable, and the two tiers separate within one to three
tokens. `math_prealgebra_0026` agrees on its first token and diverges on its second. That
is the mechanism behind every low cell in both tables, and it is a statement about the
measurement surface, not about 4-bit arithmetic.

**What this stage does not deliver.** S5 is defined to report token agreement *and*
solved-set differences. The solved-set half needs the model to produce an answer, which
means the chat surface, and every arm in this window ran bare completion by the same
control that S1 through S3 use. No solved set is claimed here, for either tier of either
arm, and the pre-registered S4 gate that consumes solved sets is not applied to this
data. Producing the solved-set half is a different window on the chat surface, and it
would not be comparable with the timing stages above.

**Blocked arm.** `apxinf-native` is not in either S5 window, because it has only one
tier on this host. The native 4-bit weight path is gated on `APXINF_LFM3_4BIT`, read in
`crates/apxinf-model/src/metal_lfm3/q4.rs:19` and
`crates/apxinf-serving/src/native_worker.rs:549`, and it landed with #132
(`e353eda5`, 2026-11-20). The binaries on the host predate it: `strings -a` finds zero
`APXINF_LFM3_*` names at all, for any member of that family, in either
`apxinf-serve` (sha256 `011d65fb…`) or `apxinf-native-worker` (`ab93b0a4…`). Unblocking
is a rebuild of both binaries from `e353eda5` or later with the flag set, which changes
artifact identity, so it is a new window and not an extension of this one. The arm is
reported blocked rather than substituted.

Evidence: `devlocal/engine-comparison-lfm3/results/s5-20261126T142337Z/` for set A
(4 raw arm files, the report, the summary, artifact identity, host state) and
`devlocal/engine-comparison-lfm3/results/s5n-20261126T143050Z/` for set B (the same,
plus the frozen prompt set and the source suite); report exit `0` on both; windows
2026-11-26 22:23:37 to 22:25:58 and 22:30:50 to 22:32:34 CST. Every arm reported
0 failed requests of 12 or 16 issued, every prompt in every arm finished, and each of
the 4 raw files in each window carries its own `prompts_sha256` and `tokenizer_sha256`.

Caveats: batch-1 serial; one host; greedy, so agreement measures one sampled path and
not the distribution; `max_tokens=64`, so a tier that would have re-converged later is
not seen to; two engines per tier but one weight file each, so the tables are one
measurement reported twice; bare completion, as above; artifact identity differs between
the two windows only in the prompt set.

> **Superseded 2026-11-18** by the M4 tier-A result above. This window is void
> (its own A/A gate failed on every arm) and stays visible for history; nothing
> below is a result.

Window: **`m2-quiet` — Apple M2 Air (reference only, VOID)**. Host amendment 2026-11-18: M1 Pro offline, M4 relay ~17 KB/s; the two-arm window ran on the Lead's M2 Air. These are M2 numbers, not M1 numbers, and this window failed the pre-registered comparability gate (see the verdict at the end of this section): nothing here is a result. Every table below is the report's own output.

### Artifact and host identity

| artifact | value |
|---|---|
| `llama_server_sha256` | `e5b1bd78bdf73c6388bd382995586a49958905c4a1aa85104025193983efe43f` |
| `llama_version` | `version: 1.0.0-dev (build 1, commit 5297534)` |
| `ollama_version` | `ollama version is 0.40.16` |
| `gguf_sha256` | `3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4` |

Host before: macOS 27.0.1 (Apple M2, 8 cores); load [16.24, 26.87, 30.85]; swapouts swap used 8628.81M free 1611.19M (pre); thermal unavailable; disk None; busy: host load average 16 -> 52 across window

Host after: macOS 27.0.1 (Apple M2, 8 cores); load [43.67, 52.56, 43.53]; swapouts swap used 9820.75M free 419.25M (post); thermal unavailable; disk None; busy: host load average 16 -> 52 across window

## Arms

| arm | stage | protocol | quantization | segments | requests | failed | generated | evidence | A/A |
|---|---|---|---|---|---|---|---|---|---|
| `llamacpp` | s1 | openai-completions | gguf-q4_k_m | 4 | 12 | 0 | 12 | yes | FAIL (123.9%) |
| `llamacpp` | s2 | openai-completions | gguf-q4_k_m | 4 | 8 | 0 | 8 | yes | FAIL (61.9%) |
| `ollama` | s1 | openai-completions | gguf-q4_k_m | 4 | 12 | 0 | 12 | yes | FAIL (115.9%) |
| `ollama` | s2 | openai-completions | gguf-q4_k_m | 4 | 8 | 0 | 8 | yes | FAIL (143.4%) |

## Corpus and token-count comparability

Prompt tokens are the engines' own reported counts for byte-identical text. Bare-completion arms must agree; a chat-protocol arm carries its template offset (mixed factor 2).

| stage | case | `llamacpp` | `ollama` |
|---|---|---|---|
| s1 | 128 tok prompt | 155 | 155 |
| s1 | 512 tok prompt | 523 | 523 |
| s1 | 2048 tok prompt | 1996 | 1996 |
| s2 | 1024 tok prompt / 128 tok out | 1007 | 1007 |
| s2 | 4096 tok prompt / 128 tok out | 3951 | 3951 |

## S1 prefill / TTFT

Client-observed time to first content delta (the S1 metric), with the engine-reported prefill as a cross-check. Ratio is paired against `llamacpp` per segment (baseline/arm; >1 = this arm faster).

| case | arm | TTFT p50 (ms) | prefill tok/s p50 | engine prefill p50 (ms) | vs baseline | n | seg |
|---|---|---|---|---|---|---|---|
| 128 tok prompt | `llamacpp` | 623.7 | 255.4 | n/a | (baseline / unpaired) | 4 | 4 |
| 128 tok prompt | `ollama` | 662.3 | 237.5 | n/a | 1.09x | 4 | 4 |
| 512 tok prompt | `llamacpp` | 2608.5 | 200.6 | n/a | (baseline / unpaired) | 4 | 4 |
| 512 tok prompt | `ollama` | 1762.3 | 299.3 | n/a | 1.26x | 4 | 4 |
| 2048 tok prompt | `llamacpp` | 12267.4 | 163.2 | n/a | (baseline / unpaired) | 4 | 4 |
| 2048 tok prompt | `ollama` | 11076.3 | 191.8 | n/a | 1.03x | 4 | 4 |

## S2 decode

Client-observed decode window (first to last content delta) per generated token. This file already declares that the decode basis is the client; the engine-reported ms/token is shown next to it.

| case | arm | decode ms/token p50 | decode tok/s p50 | engine ms/token p50 | vs baseline | n | seg |
|---|---|---|---|---|---|---|---|
| 1024 tok prompt / 128 tok out | `llamacpp` | 82.74 | 12.7 | n/a | (baseline / unpaired) | 4 | 4 |
| 1024 tok prompt / 128 tok out | `ollama` | 80.49 | 12.5 | n/a | 1.02x | 4 | 4 |
| 4096 tok prompt / 128 tok out | `llamacpp` | 100.11 | 10.5 | n/a | (baseline / unpaired) | 4 | 4 |
| 4096 tok prompt / 128 tok out | `ollama` | 61.36 | 16.3 | n/a | 1.26x | 4 | 4 |

## S3 serving-layer overhead

No data.

## S4 end-to-end agent suite

No data.

## Mixed factors (quote these with every table above)

1. Quantization is uniform in this data: gguf-q4_k_m. Cross-tier attribution still requires the 16-bit arm (Decision rule).
2. Measurement basis: S1 TTFT and S2 decode are client-observed (first and last SSE content delta). Engine-reported prefill/decode durations are shown separately, and they cover different spans per engine (llama.cpp prompt eval; apxinf state reset + capacity preparation + prefill). S3 subtracts the engine-reported durations from client wall where they exist; where they do not, S3 overhead is n/a and the wall column alone must not be read as serving overhead.
3. Pairing: every delta in these tables is the median over ABBA segments of the within-segment median ratio against `llamacpp`. No absolute number in this report crosses a window, and no single-order number is reported.

## Pre-registered gate

**Gate (comparability):** A/A <= 5% per arm; otherwise the arm's data is void.
- `llamacpp` s1: A/A FAIL (worst group spread 123.9%, 4 segments)
- `llamacpp` s2: A/A FAIL (worst group spread 61.9%, 4 segments)
- `ollama` s1: A/A FAIL (worst group spread 115.9%, 4 segments)
- `ollama` s2: A/A FAIL (worst group spread 143.4%, 4 segments)

**Gate (S2 decode):** paired median; if apxinf < 0.8 x baseline the verdict is written as 'apxinf decode lags' and a follow-up is opened; the word 'parity' is not used.
- S2 decode for `apxinf-native` vs `llamacpp`: not measured (no paired S2 decode data for both arms).

**Gate (S4 end-to-end):** equal solved set and wall delta > 10% => engine-level difference; different solved set => check template/quantization first.
- S4: not measured by this harness (the 12-task agent suite is a separate tool); no end-to-end claim is made here.

**Decision rule:** claim the difference as an engine-implementation difference only when the 16-bit arms reproduce its sign; otherwise attribute it to the quantization tier.
- No 16-bit arm was measured: differences in this report stay attributed to the quantization tier, not to the engine implementation.

## Status

- BLOCKED: `llamacpp` s1: A/A spread 123.9% exceeds 5%; the arm's data is void and must be re-measured
- BLOCKED: `llamacpp` s2: A/A spread 61.9% exceeds 5%; the arm's data is void and must be re-measured
- BLOCKED: `ollama` s1: A/A spread 115.9% exceeds 5%; the arm's data is void and must be re-measured
- BLOCKED: `ollama` s2: A/A spread 143.4% exceeds 5%; the arm's data is void and must be re-measured

No verdict may be claimed from this data set until the blockers above clear (skills/perf-verdict-gate).


## Stage-2 verdict: window void, apxinf not measured (2026-11-18)

Status: **INCONCLUSIVE** (no number from this window may be quoted as a result)

Claim: In the M2-Air two-arm window (2026-11-18 10:51:40-11:00:45) both arms
failed the pre-registered A/A gate on every stage, and the host moved materially
during the window, so the window is void by the campaign's own comparability
rule. Separately, **no `apxinf` arm ran on any reachable host** (M1 Pro offline,
M4 relay ~17 KB/s), so nothing about apxinf — faster, slower or equal — can be
claimed from this campaign, and the pre-registered `apxinf` vs `llamacpp` S2
decode gate has no data.

Read the five points below as the whole verdict:

1. **INCONCLUSIVE, window void**: A/A failed the pre-registered 5% gate on every
   arm — `llamacpp` s1 **123.9%**, s2 **61.9%**; `ollama` s1 **115.9%**, s2
   **143.4%** (worst segment-median spread over 4 ABBA segments).
2. **Why**: the host was contaminated during the window — load average
   16.24/26.87/30.85 -> 43.67/52.56/43.53 on 8 cores, swap used 8.63 GiB ->
   9.82 GiB (free 1.61 GiB -> 0.42 GiB). By `skills/perf-verdict-gate` rule 2 the
   run is reference-only.
3. **What survives the window**: the two engines report *identical* prompt-token
   counts for the same corpus (155/523/1996 prefill, 1007/3951 decode), so corpus
   identity across arms is proven by the engines themselves; and the driver's
   client-side accounting is internally consistent
   (`decode_tok_s = (gen_n - 1) / decode_ms`).
4. **No apxinf arm ran**, on any host, for any stage — the apxinf rows, the
   packed4 lane arm and the native BF16 arm do not exist in this data set.
5. **All numbers here are M2 Air reference-only.** They are not M1 numbers and
   nothing in this document compares across hosts.

Gate:

| gate | result | measured |
|---|---|---|
| comparability: A/A <= 5% per arm | **FAIL** | `llamacpp` s1 123.9%, s2 61.9%; `ollama` s1 115.9%, s2 143.4% (worst group spread of the segment medians) |
| S2 decode: `apxinf` < 0.8 x `llamacpp` | **not evaluable** | no `apxinf` request exists in this window |
| host state: label contaminated if swap/load moved | **contaminated** | load average 16.24/26.87/30.85 -> 43.67/52.56/43.53 on 8 cores; swap used 8.63 GiB -> 9.82 GiB (free 1.61 GiB -> 0.42 GiB) |

What the data does show (reference only, not a verdict): both arms serve the same
`LFM3-1.7B-Q4_K_M.gguf` and the engines report identical prompt-token counts for
the same corpus (155/523/1996 for the prefill cases, 1007/3951 for the decode
cases), so the corpus is identical across arms; the observed decode rate in this
window is about 10-16 tok/s and prefill at 2048 prompt tokens takes 11-12 s — both
far below the M1 smoke numbers and consistent with the measured load/swap move,
which is exactly why the window cannot separate engine effects from host effects.

Evidence: `devlocal/engine-comparison-lfm3/raw-m2-quiet/{raw.jsonl, identity.txt,
run.log, llama-server.log}`; normalized to the harness schema under
`raw-m2-quiet/normalized/`; rendered tables in
`devlocal/engine-comparison-lfm3/report-m2-quiet.md`; machine-readable gate state
in `summary-m2-quiet.json`.

Arms: 2 arms (`llamacpp`, `ollama`) x 4 ABBA segments x 5 cases
(S1 prompt 128/512/2048, S2 context 1024/4096 with 128 generated tokens).
Medians only; no single-order number is reported. Neither arm carries an
engine-reported duration in this window (the driver recorded client timings
only), so the engine-reported columns are n/a and no S3 decomposition exists.

Artifact: llama.cpp `1.0.0-dev` build 5297534 (sha256
`e5b1bd78bdf73c6388bd382995586a49958905c4a1aa85104025193983efe43f`), ollama
0.40.16, GGUF `LFM3-1.7B-Q4_K_M.gguf` sha256
`3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4`
(1,099,627,808 bytes).

Host: Apple M2 Air, 8 cores, macOS 27.0.1. **M2 numbers are not M1 numbers** and
nothing in this document compares across hosts. A verdict needs a quiet window on
the acceptance host (both metal locks, display asleep, no other user process) and
an actual `apxinf` arm; neither existed for this report, so the pool of
`apxinf`-vs-`llamacpp` questions remains open rather than answered.

Open question for a future attempt (not a conclusion): the operator reported both
engines at roughly 10-16 tok/s with GPU utilisation 60-72% and CPU ~90% on an
8-core M2. That pattern is consistent with the LFM3 path inside
llama.cpp/ollama being dispatch/CPU-bound rather than Metal-bound — the same
*class* of problem measured on the MLX side — but this window has no arithmetic
that separates launch cost from bytes, so the mechanism is **unknown** here. A
future attempt should measure per-op/launch attribution on the LFM3 path, not
just tok/s.

Follow-up (needed before any apxinf claim): re-run the minimum window (S2 decode,
`llamacpp` vs `apxinf-mlx-lane`, ABBA, 3+ segments) on a host that is actually
quiet; the one-command window and its acceptance checklist are in
`devlocal/engine-comparison-lfm3/M4-RUNBOOK.md` (same script for M1:
`m1/run_min_window_m1.sh`, with M1 paths).

<!-- END RESULTS -->

<!-- BEGIN TABLES -->
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

### 2026-11-18 (second amendment) - stage-2 host change: M1 Pro offline, measurements move to M4

The M1 Pro acceptance host went offline during stage-2 provisioning: it stopped
advertising on mDNS, stopped answering ssh, and is not resolvable from the
authoring host. Provisioning on it had completed (llama.cpp master with
`LLM_ARCH_LFM3`, ollama 0.40.16 with the Q4_K_M model imported, `apxinf-serve`
and `apxinf-native-worker` built from `feat/lfm3-metal`, the exact-token corpus
verified against llama.cpp `/tokenize`), but **no measurement window ran there**:
the first tier-A window aborted on a runner bug (bash 3.2 has no associative
arrays) before the first arm was driven, and it wrote no raw JSON.

Stage 2 therefore runs on `haiyan-mac-mini` (Apple M4, 16 GB, macOS 27.0.1), or is
reported as blocked/incomplete with that reason. Consequences that must be quoted
with every number:

1. **Host change**: these are M4 numbers. Nothing in this document compares an M4
   number with an M1 number, and the M1 provisioning above is not evidence of any
   measurement.
2. **Arms on M4**: the 4-bit tier is `apxinf-mlx-lane` (packed4; EngineTailor LFM3
   engine + `lfm3_worker.py`, served by `apxinf-serve` from PR #110) against
   `llamacpp` (Q4_K_M) and, if installed, `ollama` (Q4_K_M). The 16-bit tier
   (`apxinf-native` BF16 vs `llamacpp-f16`) is measured only if the BF16 snapshot
   and the F16 GGUF finish downloading on the M4's ~0.4-0.5 MB/s link; otherwise it
   is reported INCOMPLETE with that reason.
3. **Locks**: the M4 has no `flock(1)` and no `lockrun.py`, so the window runs
   under an fcntl implementation of the same contract (two lock paths, same order,
   held for the whole child). This is stated rather than silently assumed.
4. **No verdict without a validated window**: A/A <= 5% per arm, at least three
   ABBA segments per arm, one generation-evidence line per arm in the raw JSON,
   host state before/after, and every server stopped afterwards. A blocked
   download or an unavailable arm is reported as INCOMPLETE, never substituted.

### 2026-11-18 (third amendment) - the two-arm stage-2 window ran on an M2 Air and is VOID

With M1 offline and the M4 relay at ~17 KB/s, the two-arm window (`llamacpp` vs
`ollama`, both the same Q4_K_M GGUF) was run on the Lead's Apple M2 Air. The
rendered tables and the machine-readable gate state are in the results section.

That window is **void by the pre-registered comparability rule**: A/A failed on
every arm (worst segment-median spread 61.9%-143.4% against a 5% gate) and the
host moved materially during it (load average 16.2/26.9/30.9 -> 43.7/52.6/43.5 on
8 cores; swap used 8.63 GiB -> 9.82 GiB). No `apxinf` arm was measured, so the
pre-registered apxinf-vs-llamacpp decode gate has no data. The verdict for this
document is therefore INCONCLUSIVE, not "parity" and not "apxinf lags"; the
observed ~10-16 tok/s and 11-12 s at 2048 prompt tokens are reported as a
reference-only property of this contaminated window, not as an engine result.

Consequences for the reader:

1. Everything under Results is **M2 Air** data, reference-only. It is not
   comparable with any M1 number and must not be quoted as a stage-2 result.
2. The apxinf-vs-llamacpp question is open. The harness, corpus, runbooks and
   one-command window are in place; a quiet acceptance-host window running
   `apxinf-mlx-lane` (packed4) against `llamacpp` (Q4_K_M) is the follow-up.
3. The window did establish two things that survive it: the two engines report
   identical prompt-token counts for the same corpus (155/523/1996 prefill,
   1007/3951 decode), and the driver's client-side accounting is internally
   consistent (decode tok/s = (gen_n - 1) / decode_ms).

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
<!-- END TABLES -->
