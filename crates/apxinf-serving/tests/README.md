# HTTP integration checks

`http_gateway.rs` and `http_output_budget.rs` start the real `apxinf-serve` binary through the same Python test runner.
The runner uses Python standard-library HTTP clients and loopback TCP sockets.
Each case starts a separate gateway and an original CPU worker fixture.
The fixture sends the serial protocol through the normal process pipes.
Startup checks the fixture manifest against local files and the actual interpreter.

The runner owns fixture layout, gateway lifecycle, process groups, and diagnostics.
Its `--select` and `--exclude` options map cases onto the Rust test targets;
`http_output_budget.rs` selects the output-budget cases and `http_gateway.rs` runs the rest.
Each target passes `--expect-cases`, so a renamed selected case fails instead of silently matching nothing.
Every target passes `--expect-total`, so a case added without a target fails instead of never running under `cargo test`.
The output-budget target also runs `--count-only` over the class, so a new output-budget case must gain its own target instead of silently dropping out of the run.

These checks use the existing [local HTTP profile](../../../doc/serving/local-service-v0.1.md)
and [serial worker profile](../../../doc/serving/serial-profile-v0.1.md).
They do not load MLX, download models, execute tools, or contact external services.
The fixture reports its runtime as `cpu-fixture` and its MLX packages as `absent`.
Its execution labels satisfy the manifest schema only.
They do not establish real inference, model quality, performance, or Claude Code compatibility.

## Coverage

| Boundary | Required observation |
| --- | --- |
| Declared routes | Health, readiness, models, token counting, both generation APIs, and metrics respond through HTTP. |
| Nonstreamed generation | Both APIs preserve output, usage, the model alias, and request identity headers. |
| Anthropic SSE | Event names match payload types, indices remain stable, and one success terminal ends the stream. |
| OpenAI SSE | The stream ends with `[DONE]`. Only the requested final usage chunk contains usage counts. |
| OpenAI content parts | Text part arrays are normalized into the serial text form in every role on the text profile, and an image part stays an explicit request error there. |
| OpenAI message roles | A `developer` message reaches the worker as a `system` message, and an unsupported role or content shape is an actionable 400 `invalid_request` before admission. |
| Greedy sampling no-ops | Anthropic `top_k` and OpenAI `seed` are accepted on a greedy profile only as a positive integer and a non-negative safe integer, an invalid value stays an actionable 400 `invalid_request`, and a sampled or vision profile rejects both. |
| Invalid input | Malformed JSON, duplicate keys, unsupported sampling, conflicting limits, and malformed tool or message shapes never reach the worker. |
| HTTP routing | Unknown paths return 404. Unsupported methods return 405. Oversized bodies return 413. |
| Tools and stop sequences | Checked arguments become tool blocks. A stop match suppresses the matched text and following text. |
| Recoverable failure | Preparation errors and failed attempts retain readiness after resource settlement. |
| Protocol failure | Token counting preserves `protocol_fault`. Streaming emits one error without a success terminal. |
| Worker-declared failure | A failed worker terminal keeps the documented HTTP status for its error code on both generation APIs and leaves the worker ready. |
| Error envelope | `error.type` derives from the contract status (`invalid_request_error`, `not_found_error`, `rate_limit_error`, `overloaded_error`, `api_error`), and a stream error reports it without a success terminal. |
| Queue and disconnect | Overflow returns 429. Waiting and active disconnects release queue capacity and reservations. |
| Deadline | Expiry emits an error, records one expired outcome, and releases resources after worker settlement. |
| Worker exit | Readiness returns 503. Health remains available. New generation requests fail. |
| CLI shutdown | The binary exits successfully and reaps its worker process. |
| Output budget | An omitted limit returns 200 with `completion_tokens`/`output_tokens` equal to the deployment budget and `finish_reason=length` / `stop_reason=max_tokens`. |
| Rejected output limit | An explicit zero returns 400 `invalid_request` before admission on both generation APIs. |
| Exhausted output budget | Readiness and token counting stay available; both text-generation routes return 400 `invalid_request`. |

The coverage above is the text profile's HTTP surface.
The vision profile's OpenAI `image_url` acceptance has unit and prepared-frame evidence only:
the CPU fixture worker declares `image:false`, and a synthetic vision worker would need the full
vision runtime identity digests. The HTTP cases therefore cover the text-profile rejection of image parts.

## Run the checks

Run this command from the repository root:

```sh
cargo test --locked -p apxinf-serving --test http_gateway -- --nocapture
cargo test --locked -p apxinf-serving --test http_output_budget -- --nocapture
```

Python 3 must be available as `python3`.
The test adds no Rust or Python package dependency.
The runner uses process groups and therefore requires Linux or macOS.
Each listener selects an available port through port zero.
The runner sends signals only to its own gateway or process group.

Diagnostics remain in unique directories under `devlocal/serving-http-integration/`.
Set `APXINF_HTTP_TEST_OUTPUT` to select another output directory.
The runner retains gateway diagnostics and worker events when a check fails.
The CI workflow uploads this directory after success or failure.

The CPU workflow runs Rust checks and Python unit tests on Linux and macOS.
It fixes the Rust and Python versions and uses `Cargo.lock`.
It leaves `APXINF_SERVING_URL` empty, so the six external model tests remain skipped.
Run those tests separately against an approved real model deployment.
