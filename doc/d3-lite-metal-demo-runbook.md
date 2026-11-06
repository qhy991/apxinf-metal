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

## 2. Episode (the demo measurement)

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

The episode runs until the game ends or the decision cap. On the Clef
reference trajectory it ends in death at 11 decisions; d3-lite runs the full
**12-decision cap** (x=213) because its Noul probabilities stay below the 0.5
jump threshold on this all-negative trajectory. Output: `summary.json`
(median/min/max latency, wall, confusion), `steps.json` (per-decision
probabilities and actions), `step-*/` crops and annotations.

**What `steps.json` `latency_ms` measures.** The worker's per-request clock
starts *after* `_ensure_started()`, so the first decision's recorded latency
**excludes the resident worker's model load** (spawn → ready ≈ 1.3 s
externally). It contains the three cold requests, each paying that shape's
first compiled-profile trace. The player-visible first-step wall on the page
*does* include the load.

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

### 4.1 Clean-window rerun — the reported baseline

The maintained clean-window protocol (used for the independent confirmation
recorded in report §6) comes from
`devlocal/mario-demo-adapt/scripts/real_d3lite_clean_run.sh <run-name>`, which
enforces the gates in order:

1. acquire both locks (wait, never kill);
2. wait for any lock-ignoring foreign harness to finish **on its own**. If it
   does not clear within `FOREIGN_DEADLINE` (default 840 s) or stops making log
   progress for `HUNG_AFTER_S` (default 320 s), **abort (92)** and ask the
   requester to stop it — the script refuses to kill a live campaign;
3. 60 s idle `Swapouts` baseline — abort (93) if non-zero;
4. confirm no foreign model process, then snapshot;
5. `probe_load.py`, then `evaluate.py` under the 200 ms `vm_stat` sampler with
   timestamped output;
6. build a receipt from the episode's 36 crops (`make_episode_receipt.py`) and
   replay them through a **fresh** server (`run_correctness.py`) in the same
   window to collect the server's per-request `stages`;
7. snapshot and write `verdict.txt`; `contaminated=0` is required.

After releasing the locks, run (host-only):

```sh
$DEMO_PY devlocal/mario-demo-adapt/scripts/analyze_clean_run.py \
  --clean devlocal/mario-demo-adapt/runs/<clean-run> \
  --p4 devlocal/mario-demo-adapt/runs/<prior-run> \
  --out devlocal/mario-demo-adapt/runs/<clean-run>/analysis.json
```

for the per-decision table with Swapouts, the crop-hash reproducibility check,
the replay-vs-episode falsifier, stage shares and the first-decision
decomposition.

The script used for the original **contaminated** P4 window is
`devlocal/mario-demo-adapt/scripts/real_d3lite_run.sh <run-name>`; the command
sequence in §2 is the maintained form.

## 5. Expected environment limits

- The Metal port loads the BF16 checkpoint as stored and compiles one
  whole-request profile per `(length, candidate_count[, image_tokens])`; the
  first request of a new shape pays the trace.
- CPU reference comparisons (correctness) use the checkpoint's own
  `pipeline_d3.py` / `d3_runtime.D3` on a PyTorch install — see the report's
  reference section; the demo itself needs no Python ML stack.
