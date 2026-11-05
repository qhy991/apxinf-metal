# ApxInf Metal W8 lm_head

## Compilation features

The default features retain the existing APIs, tests, and examples.
Consumers that select a smaller scope must disable this crate's default features.

| Features | Compiled scope on macOS |
|---|---|
| None | Empty crate, without native bridges or Metal linkage |
| `w8-head-mlp` | W8 head, complete MLP, their CPU packing helpers, two bridges, and two shaders |
| `experiments` | The head and MLP scope, plus generic matvec, experimental modules, and eight additional bridges |

`experiments` depends on `w8-head-mlp`. Experimental integration tests and
examples require `experiments`. The minimal build excludes their test targets.
It does not read or embed the matvec, GDN, linear-layer, or full-attention shaders.
The head bridge excludes its matvec implementation through a matching compile definition.

```sh
cargo check -p apxinf-metal --no-default-features --features w8-head-mlp --all-targets
cargo check -p apxinf-metal --all-targets
```

These commands check compilation only. They do not establish GPU correctness or inference performance.
The existing build script embeds selected shader text for runtime Metal compilation.
This feature selection does not provide Metal AOT compilation.

## Runtime scope

`apxinf-metal` is an explicitly enabled generation output-head accelerator for
the native Qwen3.5 runtime on Apple Silicon. The transformer body and prompt
body stay on the existing CPU/Accelerate path. For both the first generated
token and subsequent decode tokens, the final normalized F32 hidden row is
projected by a persistent Metal W8 tied embedding, reduced to four candidates
on the GPU, and reranked from the original F32 rows on the CPU.

This path is not enabled by default. Build it and request it explicitly:

```console
cargo build --release --features accelerate,metal-w8
target/release/apxinf generate \
  --model /path/to/Qwen3.5-0.8B \
  --prompt Hello \
  --max-tokens 100 \
  --max-context 4096 \
  --no-eos-stop \
  --device cpu \
  --dtype fp32 \
  --json \
  --metal-w8-lm-head
```

Requests fail closed when the feature was not compiled, the host is not
macOS, the body is not on CPU, the model is not Qwen3.5, the checkpoint does
not use tied embeddings, or the packed shape is invalid. There is no silent
fallback after prompt or decode state has advanced.

## Stable packed-weight contract

- Source layout: Hugging Face `[vocab, hidden]`, contiguous row-major F32.
- Quantization: symmetric signed int8, independently for each 64-value group
  within each row.
- Scale: one F32 value per row/group; `scale = max(abs(group)) / 127`.
- Quantized value: `round(value / scale)`, clamped to `[-127, 127]`.
- Zero group: all quantized values are zero and the stored scale is `1.0`.
- Shape: both dimensions must be nonzero; hidden size must be divisible by 64
  and by 4; dimensions and kernel indexing must fit the checked u32 contract.

For Qwen3.5-0.8B (`vocab=248320`, `hidden=1024`), the hidden transfer is
exactly 4096 bytes per generated token. Packing happens once at model load. W8
weights, scales, the hidden staging buffer, and the result token buffer remain
resident in shared Metal buffers; the first-stage candidate buffer is private.

## Current submission contract

One output-head call uses one command buffer and one host wait:

1. Eight SIMDgroups each evaluate one vocabulary row. Each eight-row
   threadgroup retains its best four `(score, token)` candidates. Full logits
   are never materialized.
2. One 256-thread threadgroup reduces those lists to the global top four,
   breaking exact-score ties in favor of the lowest token ID.
3. The CPU reads four IDs and recomputes only those four scores from the
   original tied F32 embedding, using the same deterministic tie rule.

Only 16 bytes of candidate IDs are read from Metal.

## Verification

The crate's synthetic tests compare packed W8 scores and argmax against a CPU
W8 oracle, reject invalid inputs, and compare the real Metal two-stage argmax
against the same packed CPU oracle. The real-checkpoint diagnostic can be run
with:

```console
cargo run --release -p apxinf-model \
  --example qwen35_metal_w8_gate \
  --features accelerate,metal-w8 -- \
  /path/to/Qwen3.5-0.8B 128 Hello
```

The diagnostic is teacher-forced: each next input is selected by the native
CPU/F32 head, while CPU/F32 and Metal/W8 see the same decode hidden state.
Passing it demonstrates agreement with ApxInf's native F32 implementation; it
does **not** demonstrate exact agreement with Hugging Face or a BF16 oracle.
Separate BF16 probing found one case where the W8 top-1 changed and the BF16
winner was W8 rank 2, so the current implementation remains an explicit
opt-in approximation.

The frozen Qwen3.5-0.8B correctness and performance record is in
[`evidence/qwen35-0.8b-m4-lm-head-20260824.json`](evidence/qwen35-0.8b-m4-lm-head-20260824.json).
It retains the contaminated original block 6 and records why a predeclared
replacement block was used.

## V2 addendum: global top-4 plus exact-F32 rerank

The current source advances the decode head from v1 top-1 to a quality-recovery
pipeline. This addendum overrides the earlier submission description; the v1
evidence and binary hashes remain historical and unchanged.

The only Metal shader source is
[`src/metal_w8.metal`](src/metal_w8.metal). `build.rs` reads that file and
generates the Objective-C++ raw-string include in Cargo's `OUT_DIR`, so the
runtime compiler and KerSor both consume the same kernel text. Editing the
`.metal` file triggers a rebuild.

One decode still uses one command buffer and one wait:

1. Each 8-row threadgroup writes its best four candidates. Keeping only one
   candidate here would be incorrect when several global winners share a group.
2. The final Metal stage deterministically merges all group lists into the
   global top four, with lowest-token tie breaks, and returns only four IDs
   (16 bytes). Full logits remain unmaterialized.
3. The CPU recomputes those four scores from the original tied F32 embedding
   and the same normalized hidden row, then applies the same lowest-token tie
   break. Any Metal or rerank failure after the model state advances is returned
   to the caller; it never falls back and advances state twice.

For Qwen3.5-0.8B the v2 private partial buffer is 993,280 bytes, an increase of
744,960 bytes over v1; including the larger result buffer, persistent storage
increases by 744,972 bytes. The real-checkpoint 128-step native F32 teacher gate
passed both raw W8 top-1 and F32-reranked selection 128/128. Reranking changed
none of those native tokens. Its isolated scalar work averaged 5.86 microseconds
per decode in that diagnostic (about 0.14% of measured top-4-plus-rerank head
time). The production 100-token trajectory also exactly matches native F32 and
the frozen v1 vector.

These are ApxInf native F32 results, not Hugging Face/BF16 parity. The v2
same-binary ABBA/BAAB admission run is intentionally deferred because Logitech
utilities and WindowServer were consuming substantial CPU and the host already
had 5.7 GiB of swap in use. A rejected same-binary candidate pair had zero
per-process swaps and roughly 4.76 GB maximum RSS, but is not used to pass a
performance gate. Full v2 correctness, overhead, candidate timing, background
snapshot, and the pending performance protocol are preserved in
[`evidence/qwen35-0.8b-m4-lm-head-20260824-v2.json`](evidence/qwen35-0.8b-m4-lm-head-20260824-v2.json).

## First-token fast path

The shared generation interface now has an optional direct prefill-token hook.
When this Metal sidecar is enabled, Qwen3.5 runs the prompt body exactly once,
selects only its final hidden row, and uses the same Metal top-4 plus F32
rerank path as decode. Other models inherit the prior logits-returning prefill
behavior. A Metal error is terminal because the prompt cache has already
advanced; the loop never retries prefill through the CPU head.

A real-checkpoint production-loop smoke generated the frozen first ten tokens
exactly and measured 95.90 ms TTFT on the target M4. The host was heavily
contaminated by desktop background work and existing swap, so that timing is
directional evidence only and is not a formal performance admission result.

## Experimental body matvec tracer

The crate also exposes a separate `MetalW8MatVec` primitive for an explicitly
selected Qwen3.5 decode-body experiment. It deliberately uses
[`src/metal_w8_matvec.metal`](src/metal_w8_matvec.metal), leaving the historical
lm-head shader source above unchanged. The model-side diagnostic constructors
`GeneralQwen35::from_weights_with_metal_w8_body_layer` and
`GeneralQwen35::from_weights_with_metal_w8_body_layers` pack each selected
layer's MLP gate and up projections into one `[2 * intermediate, hidden]` W8
matrix. The selectable layer set fails closed on an empty, duplicate, or
out-of-range selection. It is used only for a one-row continuation with
`start_pos > 0`; prompt prefill, unselected layers, SiLU/multiply, the down
projection, attention, and recurrent state remain on CPU/F32. Ordinary
constructors are the kill switch.

For Qwen3.5-0.8B the selected layer adds 7,831,552 bytes of persistent Metal
buffers: 7,340,032 weight bytes, 458,752 scale bytes, a 4,096-byte input, and a
28,672-byte output. Each decode dispatch transfers only the latter input and
output and performs one command-buffer wait. The original CPU/F32 matrices are
retained for rollback, so this is added storage rather than a memory-saving
path. Selecting all 24 layers therefore adds 187,957,248 bytes (179.25 MiB) of
persistent Metal buffers, transfers 786,432 bytes per decode token, and incurs
24 command-buffer waits per decode token.

The synthetic, same-binary alternating screen is available as:

```console
cargo run --release -p apxinf-model \
  --example qwen35_metal_w8_body_screen \
  --features accelerate,metal-w8 -- 20
```

This tracer is not wired into the production CLI and remains disabled by
default. It is a quantized-quality diagnostic lane, not an F32-equivalent
backend. The frozen one-layer strict oracle failed the declared logits,
convolution-state, recurrent-state, and cached-vs-fresh envelopes, although its
top-20 overlap and exact ten-token greedy trajectory passed. The quantization
scope must not expand beyond the tested gate+up projections on the strength of
token-selection agreement alone.

The real-checkpoint teacher gate is intentionally split into two processes so
the 16-GB target never holds two complete F32 runtimes at once. Both modes use
the same binary. First capture CPU teacher inputs and expected outputs, then
force those exact inputs through the body candidate:

```console
target/release/examples/qwen35_metal_w8_body_gate \
  --model-dir /path/to/Qwen3.5-0.8B --mode cpu --steps 128 \
  > /tmp/qwen35-body-cpu-teacher.json

target/release/examples/qwen35_metal_w8_body_gate \
  --model-dir /path/to/Qwen3.5-0.8B --mode body --all-layers --steps 128 \
  --teacher-json /tmp/qwen35-body-cpu-teacher.json
```

The all-24-layer real-checkpoint gate matched the CPU teacher for 128/128
steps, with 128 hits in every selected layer. A separate same-binary direct
free run matched all 128 CPU token IDs; every selected layer had 127 hits
because the first generated token comes from unchanged CPU/F32 prefill. Both
processes reported zero process swaps. The body process added about 367.02 MiB
of observed peak RSS over the CPU process. These results establish the tested
greedy trajectory, not strict F32 logit or state parity.

The lightweight same-binary end-to-end screen intentionally stopped this
topology from performance promotion. Across three CPU and three body samples,
with desktop noise and pre-existing system swap, median generation throughput
was 17.8465 token/s for CPU and 17.2962 token/s for the body lane: a 3.08%
regression rather than the required 5% improvement. The 24 Metal
projection-and-wait calls consumed a median 13.445 ms per decode token. This is
a candidate stop/go screen, not formal ABBA evidence; a formal run was not
warranted for the rejected candidate.

The complete receipts and STOP decision are archived in
[`evidence/body/qwen35-0.8b-m4-all24-summary-20260824.json`](evidence/body/qwen35-0.8b-m4-all24-summary-20260824.json),
with the one-layer quality boundary in
[`evidence/body/qwen35-0.8b-m4-layer0-summary-20260824.json`](evidence/body/qwen35-0.8b-m4-layer0-summary-20260824.json).
The selectable lane and kill switch remain useful for diagnostics, but the next
performance experiment should keep a full layer or the whole decode body GPU
resident and batch work into one command buffer instead of synchronizing the
CPU with Metal 24 times per token.

The frozen Transformers oracle example also accepts
`--metal-w8-body-layer INDEX` when built with `accelerate,metal-w8`; its default
remains the unchanged CPU/F32 path.

## Experimental complete MLP block tracer

The next independent tracer keeps the complete selected MLP on Metal. Its
dedicated [`src/metal_w8_mlp.metal`](src/metal_w8_mlp.metal) source and bridge
do not modify the historical lm-head or gate+up shaders. One selected decode
layer copies a normalized hidden row to a persistent shared buffer and encodes
three dispatches in one command buffer:

1. W8 gate+up matvec into a private `[2 * intermediate]` buffer.
2. SiLU(gate) times up into a private `[intermediate]` buffer.
3. W8 down matvec into a shared `[hidden]` output buffer.

The command buffer is committed and waited exactly once. Only the 4-KiB input
and 4-KiB output cross the CPU/Metal boundary. Residual addition, attention,
GDN/state, prefill, and unselected layers remain CPU/F32. Metal failures are
returned without a CPU retry, and ordinary constructors remain the kill
switch. The explicit constructors are
`GeneralQwen35::from_weights_with_metal_w8_mlp_block_layer` and
`GeneralQwen35::from_weights_with_metal_w8_mlp_block_layers`.

For Qwen3.5-0.8B, persistent Metal storage is 11,749,376 bytes per selected
layer and 281,985,024 bytes (268.922 MiB) for all 24 layers. All-24 transfer is
196,608 bytes per decode token, 75% less than the earlier gate+up sidecars, but
the current CPU-residual topology still requires 24 waits per token. Original
CPU/F32 weights stay resident for immediate rollback.

Build and run the two-process teacher gate with:

```console
cargo build --release -p apxinf-model \
  --features accelerate,metal-w8 \
  --example qwen35_metal_w8_mlp_block_gate

target/release/examples/qwen35_metal_w8_mlp_block_gate \
  --model-dir /path/to/Qwen3.5-0.8B --mode cpu --steps 128 \
  > /tmp/qwen35-mlp-block-cpu-teacher.json

target/release/examples/qwen35_metal_w8_mlp_block_gate \
  --model-dir /path/to/Qwen3.5-0.8B --mode block --all-layers --steps 128 \
  --teacher-json /tmp/qwen35-mlp-block-cpu-teacher.json
```

The single-layer and all-24 teacher gates matched 128/128 CPU selections. The
direct all-24 free run also matched all 128 token IDs, with 127 hits in every
selected layer because the first token is produced by unchanged prefill. All
recorded processes reported zero process swaps. These are quantized
token-selection quality results, not strict F32 parity; the complete block does
not override the previously failed strict F32 envelopes.

A same-binary A-B-B-A-A-B candidate screen produced median throughput of
19.3001 token/s for CPU and 21.9083 token/s for the complete block, a 13.51%
candidate improvement, with all three pairs positive. The 24 block calls
consumed a median 10.408 ms per token. Logitech, XProtect, CloudKit, and other
desktop work were active and the host already had about 4.62 GiB of system
swap, so this is **PROMISING / awaiting quiet-host formal admission**, not a
formal performance claim and not authorization to enable the lane by default.

Exact source, binary, model, trajectory, resource, and raw sample hashes are in
[`evidence/mlp-block/qwen35-0.8b-m4-mlp-block-summary-20260824.json`](evidence/mlp-block/qwen35-0.8b-m4-mlp-block-summary-20260824.json).

## Experimental combined MLP-block + lm-head tracer

The combined tracer is a separate, explicit constructor:
`GeneralQwen35::from_weights_with_metal_w8_mlp_blocks_and_lm_head`. It selects
all layers for the complete MLP block above and reuses the existing v2 Metal W8
top-4 plus exact-F32 rerank head. Ordinary construction, the MLP-only
constructors, and the head-only constructor are unchanged kill switches. No
Metal error falls through to CPU after state advancement.

Body prefill remains CPU/F32. In the direct generation hook, the first token
uses one Metal head call but no Metal MLP call. Each later token uses all 24 MLP
blocks and the head. Separate receipts count MLP calls by layer and head calls
by prefill/decode/teacher phase, so a missing lane or accidental double
projection is visible.

For Qwen3.5-0.8B, the combined Metal buffers total 553,154,576 bytes
(527.529 MiB): 281,985,024 bytes for 24 MLP blocks plus 271,169,552 bytes for
the v2 head. The reusable Rust MLP outputs add 98,304 bytes. Each decode token
crosses 200,720 bytes (196,608 bytes for all MLP blocks, 4,096 head input
bytes, and 16 candidate-ID output bytes) and waits 25 times: once per MLP layer
and once for the head. CPU F32 weights remain resident for immediate rollback.

The same release gate binary matched the CPU teacher on 128/128 outputs. The
direct free run matched all 128 CPU token IDs; every MLP layer recorded 127
decode calls, while the head recorded one prefill and 127 decode calls. All
quality and screen processes stayed below 6 GiB RSS and reported zero process
swaps.

An A-B-B-A-A-B same-binary candidate screen measured median throughput of
18.9539 token/s for CPU and 32.0277 token/s for the combined tracer, a 68.98%
candidate improvement, with all three paired comparisons positive. Median
TPOT fell from 52.760 ms to 31.223 ms. The host was not quiet: Logitech agents
reached roughly 65% CPU, load average was about 3.7--4.1, and the system already
had roughly 6 GiB of swap in use. Therefore this is **PROMISING / awaiting
quiet-host formal admission**, not a formal claim or default-path promotion.

The frozen combined topology, correctness, resource, timing, and hash record is
[`evidence/combined/qwen35-0.8b-m4-combined-summary-20260824.json`](evidence/combined/qwen35-0.8b-m4-combined-summary-20260824.json).

## Independent state-resident GDN tracer primitive

`PackedW8GdnBlock` and `MetalW8GdnBlock` are an independent next-step tracer.
`GeneralQwen35::from_weights_with_metal_w8_gdn_layer` can construct exactly one
explicitly selected diagnostic layer; the production CLI, `AutoModel`, ordinary
constructors, and default path still never construct it. The primitive accepts
one already input-normalized decode row and keeps the GDN
convolution histories plus canonical F32
`[value_heads, key_dim, value_dim]` recurrent matrix resident on Metal. Its
single command buffer contains the stacked W8 `q/k/v/z/a/b` projection,
depthwise convolution, SiLU/L2 preprocessing, recurrent update, output RMSNorm
and z gate, and W8 output projection. It commits and waits exactly once.

The CPU packed oracle consumes the exact same W8 weights and canonical state.
Convolution histories retain K samples, matching the native Qwen3.5 backend's
cache contract. Decode writes only scratch state buffers. Active and scratch
buffers are swapped after successful command completion; a validation, Metal,
or injected post-execution error leaves the committed state unchanged. The
public path requires an explicit state seed, rejects non-finite or wrong-width
inputs, and never retries through CPU.

Run the checkpoint-free tests with:

```console
cargo test -p apxinf-metal -- --test-threads=1
cargo test --release -p apxinf-metal --test gdn_block \
  production_shape_metal_gdn_matches_the_packed_oracle_for_one_decode \
  -- --ignored --exact --test-threads=1
```

The ignored gate uses Qwen3.5-0.8B production dimensions but deterministic
synthetic weights. It neither reads nor loads a checkpoint and is not a model
quality or formal performance result.

The separate official-checkpoint layer-0 gate is archived in
[`evidence/next-hotspot/qwen35-gdn-real-layer0-gate-summary-20260824.json`](evidence/next-hotspot/qwen35-gdn-real-layer0-gate-summary-20260824.json)
(SHA-256
`0862cd73a36aa8e833e7899d92dc476c0ec6bc0347edf2955d6d3ee0f9a5028f`).
Its 128-step teacher-forced and direct free-run token trajectories both matched
the CPU path exactly, with one prefill seed and the expected 128/127 decode
commits. Every child process reported zero swaps and stayed below 6 GiB RSS.
The timings were single-pass observations on a host with about 5.15 GiB of
pre-existing global swap, so the record is candidate-only correctness/resource
evidence: it makes no performance claim and does not promote the path.

## Independent complete linear-attention layer tracer

`PackedW8LinearLayerBlock` and `MetalW8LinearLayerBlock` compose the existing
packed GDN and complete packed MLP contracts into one decode-only layer tracer:
input RMSNorm, state-resident GDN, attention residual, post-attention RMSNorm,
complete W8 MLP, then the MLP residual. The explicit diagnostic constructor
`GeneralQwen35::from_weights_with_metal_w8_linear_layer` can own exactly one
selected linear-attention layer after CPU prefill seeds its recurrent state.
The CLI, `AutoModel`, ordinary constructors, and every default path remain
unchanged kill switches and never construct it.

The combined handle encodes all 13 dispatches through one compute encoder in
one command buffer, commits once, and waits once. It owns exactly 32 persistent
Metal buffers: 24 shared buffers for packed weights/parameters, host I/O, and
active/scratch state; and eight private reusable activation buffers. Decode
transfers one F32 hidden row in and one out, with zero recurrent-state host
traffic. At Qwen3.5-0.8B hidden width 1024, those transfers are exactly 4,096
bytes in and 4,096 bytes out. `LinearLayerBufferLedger` records every persistent
byte and per-decode transfer/synchronization contract; `LinearLayerMetalStats`
records successful and failed submitted work separately.

GDN convolution and recurrent writes target scratch buffers. Active state
pointers swap only after successful command completion; injected
post-execution failure records its command/encoder/commit/wait and input copy,
but records no output copy or committed state version. Reset clears both state
sets, receipts, and the seed flag, so decode remains fail-closed until a fresh
CPU prefill seed. The tiny synthetic Metal tests compare output and every state
component with the packed CPU oracle and exercise terminal fault handling and
reset without loading a checkpoint:

```console
cargo test -p apxinf-metal --test linear_layer_block -- --test-threads=1
```

### Versioned three-layer transaction

`MetalW8LinearLayerStack3` is a separate diagnostic-only v1 ABI for exactly
three consecutive linear-attention layers. It does not replace or alter the
complete-layer v2 symbols. The stack uploads one hidden row, runs three compute
encoders in one command buffer using two ping-pong hidden rows and one shared
private scratch set, commits and waits once, then downloads only the final row.
All three layers continue to own separate active/scratch recurrent state. The
12 state-buffer pointer swaps occur only after command completion, injected-
fault checks, and a final-output finite check all succeed, so the receipt's
state commit is only `000` or `111`.

This version intentionally does **not** reproduce the two intermediate host
finite checks performed when three independent blocks are staged through the
CPU. Its ledger records zero intermediate checks and one final check. Any GPU
or post-submit validation failure leaves all three active states and the host
output unpublished and makes the lane terminal until reset.

For Qwen3.5-0.8B, one stack owns 76 buffers (68 shared, eight private) and
76,351,488 resident MTLBuffer bytes. Six stacks plus the six full-attention MLP
blocks close to 504 buffers and 528,605,184 bytes, with 12 command buffers,
36 encoders, 12 waits, and 49,152 bytes in each host-transfer direction per
decode. These are static resident-buffer/transaction ledgers, not process RSS
or performance claims. The explicit General constructors remain unreachable
from the CLI, `AutoModel`, registry, and defaults. Synthetic coverage requires
no checkpoint:

```console
cargo test -p apxinf-metal --test linear_layer_stack3 -- --test-threads=1
```

The first official Qwen3.5-0.8B layer-0 teacher-forced gate rejected this
candidate: execution receipts were exact, resources stayed bounded, and 127 of
128 tokens matched CPU, but step 14 changed one argmax token. Per the frozen
quality contract, free-run was not attempted and the lane remains
diagnostic-only and unpromoted. The candidate timing is not performance
evidence. The no-replace receipts, resource ledger, hashes, and qualified root
cause hypothesis are archived in
[`evidence/next-hotspot/qwen35-linear-layer-real-layer0-gate-summary-20260824.json`](evidence/next-hotspot/qwen35-linear-layer-real-layer0-gate-summary-20260824.json).

The explicit gate-only constructor
`GeneralQwen35::from_weights_with_packed_w8_linear_layer_reference` reuses the
same packing function and runs `PackedW8LinearLayerBlock::decode_reference` on
CPU. It is a discrimination control for quantization error versus Metal
arithmetic/reduction differences, not a serving backend. It preserves the same
prefill seed, state transaction, terminal-error, and reset contracts and is
also absent from CLI, `AutoModel`, and defaults.
