# Qwen3.5-0.8B dedicated executable

This executable uses one fixed checkpoint through the existing native ApxInf implementation.
The profile uses CPU/Accelerate F32, the Metal W8 head, and all 24 Metal W8 decode MLP blocks.
Generation uses tokenizer EOS `<|im_end|>` (248046), matching the general ApxInf CLI.
Read [the contract](CONTRACT.md) for inputs, output records, limits, and timing definitions.

## Build

Build this package separately to retain its reduced feature selection.
A workspace build can combine dependency features from other packages.

```sh
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk \
cargo build --locked --release -p apxinf-qwen35 --bin apxinf-qwen35-08b
```

Use an SDK compatible with the installed compiler.
The SDK path above matches the measured local configuration.
Add `--features general-reference` to build the same entry with the general compilation scope.
That feature changes compilation scope, not runtime settings.

## Run

```sh
target/release/apxinf-qwen35-08b \
  --model /Users/haiyan-mini/Agent4Kernel/.apxinf-models/Qwen3.5-0.8B-2fc063647-staged \
  --prompt '法国的首都是哪里？请只用一句话回答。' --max-tokens 64
```

Plain mode streams decoded text.
Add `--json` for one complete result with timing and token IDs.
Use `--jsonl` without `--prompt` to keep the model loaded for independent requests.
Each input line contains a JSON object, such as `{"prompt":"What is 7 times 8?","max_tokens":64}`.
The process emits a ready record before its result records.

## Compilation scope

The default build includes Qwen3.5 text inference, SafeTensors loading, tokenization, and two Metal bridges.
It excludes other model families, the model registry, GGUF, experimental Metal bridges, and tokenizer C++ training code.
It also excludes the general CLI and its command parser.
The upstream tokenizer still compiles Rust code without optional feature controls.
Source exclusions do not establish a globally minimal binary.

This package uses existing dependencies through their public APIs.
The local core, loader, model, and tokenizer crates provide inference and text handling.
`serde_json` provides the request and result format.
The existing `sha2` version provides fixed-asset identity checks.
No new third-party dependency enters the lock file.

## Delivery boundary

The executable needs the five external assets listed in the contract.
Weights are not embedded in the executable.
The runtime needs macOS system libraries, including Metal and Accelerate.
It does not require Python or a separate inference process.
Metal shaders remain embedded source compiled by Metal during model construction.
This build does not provide precompiled GPU machine code.

Asset hashing occurs on each process startup and contributes to startup cost.
The resident JSONL mode amortizes model loading across requests.
Every request resets its model state and does not reuse previous prompts.

## Measurement

The stage-two comparison lives in `benchmarks/specialization/v2`.
It compares general ApxInf, the dedicated entry with general features, and the reduced dedicated build.
Build time, file size, startup, and generation remain separate results.
The comparison preserves weights, precision, generation algorithm, and compiler settings.
It does not qualify a new kernel or claim a stable inference speedup.
