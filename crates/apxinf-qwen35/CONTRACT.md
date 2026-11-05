# Qwen3.5 dedicated executable contract

Status: implementation contract. Date: 2026-10-10.

This executable serves one local user through a command line.
It implements no HTTP service or worker protocol.
Existing ApxInf CLI and serving protocols retain their behavior.

## Fixed target

The executable name is `apxinf-qwen35-08b`.
Its profile is `qwen35-0.8b-f32-metal-w8-head-mlp-v1`.
The model is `Qwen/Qwen3.5-0.8B` at the existing local revision `2fc06364715b967f1860aea9cf38778875588b17`.
The target device is Apple Silicon with Metal and Accelerate on macOS.

The context limit is 2048 tokens, including the generated output budget.
Each request has batch size one and exclusive use of its model state.
Generation uses the existing greedy algorithm and the tokenizer EOS token.
The fixed tokenizer EOS is `<|im_end|>` (248046), as used by the existing ApxInf CLI.
The model configuration EOS, `<|endoftext|>` (248044), does not control generation stopping.
A missing or different tokenizer EOS is an error.

The CPU body uses F32 with Accelerate.
Decode uses the existing Metal W8 head and all 24 MLP blocks.
The executable does not choose another provider or precision.

The runtime checks each required asset against an embedded SHA-256 digest before model construction.
Required assets are config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja, and the fixed single SafeTensors shard.
The runtime loads that shard directly and selects the existing text tensor names.
The runtime does not require a shard index or load vision tensors.
An asset mismatch is an error, including weights with the same dimensions but different bytes.

Assets remain separate files. The executable does not contain the weights.

Keep asset contents unchanged while the process runs.
The weight file descriptor prevents path replacement from selecting another file after hashing.
It does not prevent another process from modifying that file in place.

## Arguments

`--model PATH` selects the directory containing the fixed assets.
`--prompt TEXT` selects one request.
`--max-tokens N` sets a positive output budget, with default 64 and maximum 2047.
`--json` selects one JSON result for a single request.
`--jsonl` selects repeated independent requests from standard input.
`--help` describes these arguments without loading assets.

Unknown, duplicate, missing, and conflicting arguments are errors.
The prompt and JSONL modes are mutually exclusive.
JSON and JSONL are mutually exclusive.
No provider, model, precision, image, sampling, or context override exists.

## Repeated requests

JSONL means one JSON object per line.
A request contains `prompt` and optional `max_tokens`.
Other fields, empty prompts, invalid types, and invalid budgets are errors.
Each JSONL request contains at most 65536 encoded bytes, excluding its final newline.

The runtime emits a ready record after asset checks, tokenizer loading, and model construction.
It processes one request before reading the next request.
It resets model state before every generation.
Requests do not share conversation history or cached prefixes.

End of input closes the process successfully.
On any request or inference error, the runtime writes a diagnostic to standard error and exits unsuccessfully.
An error never produces a successful result for that request.

## Output and timing

The JSON format is `apxinf-qwen35-specialized-v1`.
Records use a `kind` field with value `ready` or `result`.
A ready record states the profile, asset identity, context, precision, and startup durations.

A result states the profile, prompt token IDs, generated token IDs, text, stop reason, and timing.
A result also contains the existing generation path receipt.
Receipt counters accumulate across requests in one process, although each request resets the model state.
The stop reason is `eos` or `max_tokens`.

Plain output flushes complete text chunks from the existing incremental tokenizer decoder.
The runtime checks the final text against complete decoding and writes any remaining suffix.
The complete plain output equals the decoded answer followed by a newline.

JSONL emits and flushes each complete record.
JSON modes do not emit partial result records.

`asset_check_ms` measures complete asset hashing.
`tokenizer_load_ms` measures tokenizer construction.
`model_load_ms` measures native model construction, including packing and Metal preparation.

`request_ms` measures reset through generation return.
`ttft_ms` measures request start through the first token callback.
`decode_ms` measures the first token callback through generation return.
`total_request_ms` also includes prompt preparation and final text decoding.

JSON measurements exclude result serialization and standard-output writes.
JSON timing excludes incremental text decoding because JSON modes use only final text decoding.

Plain generation includes its incremental text decoding and output costs.
The first token callback does not necessarily produce a visible text chunk.
The runtime measures all durations with the same monotonic clock.

## Build scope and verification

The default dedicated build excludes other model families, registration, VLA, and diagnostic model modules.
It excludes experimental Metal modules, MatVec, GGUF, tokenizer progress bars, and tokenizer C++ training code.
The `general-reference` feature restores those build components for comparison through the same entry point.
Both selections use identical runtime settings and asset checks.
The third-party tokenizer retains upstream Rust code without optional feature controls.
That dependency boundary remains explicit.

Compilation evidence must include selected features, dependency files, and native objects.
Correctness checks must compare the actual executable with the existing native path.
Tests must cover invalid assets, invalid arguments, context rejection, and repeated request reset.
Performance comparisons must retain the same weights, precision, algorithm, and timing boundaries.
Build cost, artifact size, startup, and generation results remain separate measurements.
