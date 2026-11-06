# d3-lite × Mario demo runbook

How to run the **d3-lite** decision backend on the Mario demo end to end: the
resident `apxinf-decide-server`, the `evaluate.py` episode, and the browser
page. This is the P4 companion to the [port report](d3-lite-metal-demo.md).

The demo speaks the same JSONL protocol as the Clef `apxinf-decide-server`, so
no new bridge is needed and the Clef bridge is untouched: set
`APXINF_DECIDE_CMD` to the d3-lite server and select the `clef` policy.

## 0. Paths and build

```sh
# Apple Silicon + MLX
export PATH="/opt/homebrew/bin:$HOME/.cargo/bin:$PATH"
export MLX_ROOT="$HOME/apxinf-runtime/mlx-sdk"
export DYLD_LIBRARY_PATH="$MLX_ROOT/lib"

REPO=$HOME/src/apxinf-metal/<worktree>          # this checkout
DEMO=$HOME/Documents/Infinity/Agent4Kernel/mario-showcase-20261103/mario-demo
DEMO_PY=$HOME/Documents/Infinity/Agent4Kernel/mario-demo-env/bin/python
MODEL=$HOME/Documents/Infinity/Agent4Kernel/models/d3-lite-9c10703

cargo build --release --locked --features mlx --bin apxinf-decide-server
BIN=$REPO/target/release/apxinf-decide-server
```

The server verifies the package's code→token mapping (`d3_format.answer_codes`
recomputed from `tokenizer.json` and compared to `decision_config.json`) before
it loads the model; `ready.codes_verified` must be `true`.

## 1. Standalone JSONL server

```sh
export APXINF_DECIDE_CMD="$BIN --model $MODEL"
printf '%s\n' \
  '{"id":1,"state":"Answer each question using only the picture it refers to.",
    "questions":{"enemy":{"type":"noul","instructions":"Close-up of the area just ahead of Mario in a Super Mario Bros. level:\nIs there an enemy (a brown Goomba or a Koopa) in this picture?"}},
    "images":["<base64 png>"]}' \
  | $APXINF_DECIDE_CMD
```

Response (one line):

```json
{"id":1,"answers":{"enemy":{"type":"noul","noul":0.0231}},
 "latency_ms":184.2,"stages":{"host_prep_ms":3.1,"layout_ms":4.0,
 "vision_ms":61.0,"forward_ms":115.8,"sync_ms":2.1,"questions":1},
 "model":{"name":"metal_d3","format_id":"d3-code-readout-v1","codes":255}}
```

`latency_ms` is the full request clock (base64 + PNG decode → preprocess →
prompt → model → readback). The stages sum to the model's own call; `forward`
is the text layers + FP32 readout.

**Image contract.** One image per request. The demo's separate protocol
(`MARIO_CLEF_PROTOCOL=separate`, one crop + one question) is the supported
shape; `joint` sends three images and is rejected by name.

## 2. Episode (the 11-decision measurement)

Same startup parameters as the Clef reference:

```sh
export APXINF_DECIDE_CMD="$BIN --model $MODEL"
export MARIO_CLEF_PROTOCOL=separate
export APXINF_LOAD_TIMEOUT=600
export APXINF_DECIDE_TIMEOUT=180

cd $DEMO
$DEMO_PY examples/mario_demo/evaluate.py \
  --policy clef --max-decisions 12 --out <new-output-dir>
```

The episode ends on death (11 decisions on this trajectory). Output:
`summary.json` (median/min/max latency, wall, confusion), `steps.json`
(per-decision probabilities and actions), `step-*/` crops and annotations.

## 3. Browser page

```sh
# 8766 may be occupied by a pre-existing Clef demo session; use another port.
$DEMO_PY examples/mario_demo/server.py --port 8767
```

Open <http://127.0.0.1:8767>, choose `clef` in the policy dropdown and click
**重开**, **单步** or **自动**. The header shows `ApxInf Metal · 本地 · 无云端
请求`, `本地请求` counts worker requests, and the latency column shows the
client-measured wall per decision. Cloud usage stays at 0 / 400 and
`cost_usd` at 0.

## 4. Measurement discipline

- **Locks.** Acquire `~/.cache/enginetailor/metal-measure.lock` then
  `/tmp/enginetailor-metal.lock` (`flock`, wait, never kill other campaigns)
  and hold them for the whole window: snapshot → probe → episode → page.
- **Swap.** Snapshot `vm_stat` before/after and sample during the episode;
  `Swapouts` delta > 0 marks the window contaminated. Report the numbers as a
  reference, not a KEEP.
- **One model process.** Probe and episode do not overlap; the page's worker
  is spawned only after the episode's worker has exited.
- **No optimization claims.** Different models (Clef Flash 9B vs d3-lite
  0.8B): report each side's latency only, no speedup ratio.

The reference script used for the reported P4 window is
`devlocal/mario-demo-adapt/scripts/real_d3lite_run.sh <run-name>` (ignored
working copy; the command sequence above is the maintained form).

## 5. Expected environment limits

- The Metal port loads the BF16 checkpoint as stored and compiles one
  whole-request profile per `(length, candidate_count[, image_tokens])`; the
  first request of a new shape pays the trace.
- CPU reference comparisons (correctness) use the checkpoint's own
  `pipeline_d3.py` / `d3_runtime.D3` on a PyTorch install — see the report's
  reference section; the demo itself needs no Python ML stack.
