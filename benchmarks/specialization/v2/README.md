# Dedicated native binary comparison

This comparison uses the fixed Qwen3.5-0.8B checkpoint from the first specialization experiment.
It compares three build selections from one source snapshot.

| Selection | Entry | Compiled scope |
|---|---|---|
| full | Existing ApxInf CLI | All model families and legacy Metal features |
| reference | Dedicated Qwen3.5 entry | General model, Metal, loader, and tokenizer features |
| minimal | Identical dedicated entry | Required Qwen3.5, Metal, loader, and tokenizer features |

Only reference and minimal share the dedicated timing boundaries and asset checks.
The full CLI supplies output-token parity and prompt-length checks.
Its existing result format does not expose prompt token IDs.
It does not supply a controlled runtime comparison with the dedicated entry.

## Reproduce

Use new absolute paths for `SNAPSHOT` and `OUTPUT` below.
Do not reuse a previous build directory for a clean-build measurement.
The prepare phase hashes assets and saves a source snapshot.
The build phase uses six independent target directories.
It measures clean, unchanged, and comment-only rebuilds separately.

```sh
python3 benchmarks/specialization/v2/compare_native.py prepare \
  --source "$PWD" --snapshot "$SNAPSHOT" --output "$OUTPUT"

SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk \
CARGO_HOME=/private/tmp/apxinf-specialization-cargo-home \
python3 benchmarks/specialization/v2/compare_native.py build \
  --source "$PWD" --snapshot "$SNAPSHOT" --output "$OUTPUT"

python3 benchmarks/serving/with_metal_lock.py \
  --lock /Users/haiyan-mini/.cache/enginetailor/metal-measure.lock --wait-seconds 60 -- \
  python3 benchmarks/specialization/v2/compare_native.py runtime \
    --source "$PWD" --snapshot "$SNAPSHOT" --output "$OUTPUT"

python3 benchmarks/specialization/v2/compare_native.py summarize \
  --source "$PWD" --snapshot "$SNAPSHOT" --output "$OUTPUT"
```

The configured dependency cache must already contain the locked dependencies.
The local Metal device must be accessible to the measured processes.
The runtime phase needs exclusive use of the measurement lock.

## Evidence

The preregistration is `contract.json`. The fixed corpus is `cases.json`.
`inputs.json` binds assets, source files, configuration, and measurement scripts by SHA-256.
`builds.json` contains build commands, durations, binary identities, dependency records, native objects, and symbols.
`runtime.json` contains process state, raw output identities, parsed records, and request labels.
`summary.json` validates the sample grid, source exclusions, identities, and corpus equivalence.

The two dedicated selections each run twice, in reference/minimal/minimal/reference order.
Each process runs two warmup rounds and three measured rounds over the complete corpus.
The corpus includes short Chinese and English prompts, arithmetic, and a longer English response.
Every request resets model state.
One separate plain-output request checks actual streaming text against complete decoding.

The runtime records swap and thermal state around every process.
Active swapping makes attribution of latency changes inconclusive.
These measurements do not establish formal performance qualification.
The source snapshot excludes optional CUDA vendor files because this experiment does not build CUDA.
It includes declared crate tests because Cargo validates their paths during binary builds.

## Setup records

`model-specialization-v2-20261010` contains preparation only, before review corrections.
`model-specialization-v2-20261010-r1` records a manifest failure before compilation.
That snapshot omitted explicitly declared test files.
Its preserved setup script and build log explain the failure.
`model-specialization-v2-20261010-r2` includes those files and completes its build measurements.
Its dedicated runtime fails before generation because its original EOS check was incorrect.
The existing CLI uses tokenizer EOS 248046. The model configuration contains EOS 248044.
The corrected dedicated entry preserves the CLI choice.
The final experiment uses `model-specialization-v2-20261010-r3`.
Only the final experiment contributes measurements.
