# LFM3-1.7B engine comparison: apxinf vs llama.cpp vs ollama

Status: **method + pre-registered gate; stage-2 tier-A window complete on the M4
(2026-11-18) — `llamacpp` vs `ollama` on identical Q4_K_M weights, every gate met.**
The `apxinf-native` arm and the 16-bit matched-precision pair are **not** measured
(see the stage-2 section); no apxinf-vs-llama.cpp claim is made in this document.
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
S2 `max_tokens=128`; S3/S4/S5 not run; two different HTTP surfaces (declared
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
