#!/usr/bin/env python3
"""Issue #135 (Lead 11:30) four-step window on this M1 Pro host, in one lock window.

Preregistered by doc/decode-bandwidth-utilization.md (Gate/Decision frozen in
commit before this file; RUNNER §1). Order A1 -> A2 -> A3 -> prefill AB, all in
one process so the same hot/power state covers every step.

  run-bw-util.py window --res ~/lfm3-dig-run/bw-util [--ctxs 512,4096,16384,32768]
      Take both Metal locks (wait, never steal), verify the machine model,
      then run all four steps and write receipts.

  run-bw-util.py a1 --res <dir>        MLX streaming copy/reduce, no lock assumed
  run-bw-util.py a2 --res <dir> ...    lane ctx ladder + reference arm
  run-bw-util.py a3 --res <dir> ...    GPU busy: xctrace, then probe/fallback
  run-bw-util.py prefill --res <dir>   packed4 vs bf16 prefill AB

Not a hypothesis test by itself: raw numbers go to receipts, the doc §5 write-up
computes the five-tier verdicts frozen in the doc.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any

LOCK_PATHS = [
    "/tmp/enginetailor-metal.lock",
    Path.home() / ".cache/enginetailor/metal-measure.lock",
]
LANE_VENV = Path.home() / "ag0-lfm3-0f6bbbfaa348/.venvs/lfm3"
LX_VENV = Path.home() / "lfm3bench/mlxvenv"
ENGINE_ROOT = Path.home() / "ag1-serving-025dfff3bd11/serving/engine"
WORKER_SCRIPT = Path.home() / (
    "ag1-serving-025dfff3bd11/serving/python/apxinf/apxinf/serving/lfm3_worker.py"
)
PACKED4 = Path.home() / "Models/LFM3-1.7B-MLX-4bit"
BF16 = Path.home() / "lfm3bench/models/LFM3-1.7B-bf16"
SMALL = Path.home() / "Models/Qwen3-0.6B/snapshots"
CTXS = (512, 4096, 16384, 32768)
GEN = 128
REPEATS = 3
AFFIX_WORDS = 40  # words per repetition of the fixed filler text

# EngineTailor Engine step/prefill cost excludes host argmax by construction
# (step returns float32 logits, mx.eval barrier inside). Both A2 arms consume the
# argmax themselves; cost is measured at engine boundary.
SEED_PROMPT = (
    "Explain, in one paragraph and without bullet points, why a benchmark result "
    "is only as trustworthy as the window it was measured in."
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _dump(res: Path, name: str, payload: Any) -> Path:
    path = res / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False))
    return path


def _log(res: Path, line: str) -> None:
    res.mkdir(parents=True, exist_ok=True)
    with (res / "window.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{_now()} {line}\n")
    print(line, flush=True)


def _spread(values: list[float]) -> float:
    """A/A spread: (max-min)/median of per-repeat values."""
    if len(values) < 2 or statistics.median(values) <= 0:
        return 0.0
    return (max(values) - min(values)) / statistics.median(values)


def _run(argv: list[str], *, env: dict | None = None, cwd: Path | None = None,
         timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pressure() -> dict:
    """Free memory + swap around a window (16 GB shared host)."""
    out = {}
    for cmd, key in (
        (["vm_stat"], "vm_stat"),
        (["sysctl", "-n", "vm.swapusage"], "swap"),
        (["sysctl", "-n", "hw.memsize"], "memsize"),
    ):
        try:
            out[key] = _run(cmd).stdout.strip()
        except Exception as error:  # noqa: BLE001
            out[key] = f"unavailable:{error!r}"
    return out


def _machine() -> dict:
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.strip()
    hw = _run(["system_profiler", "SPHardwareDataType"]).stdout
    return {
        "brand_string": brand,
        "hardware_profile": hw,
        "model_name": _run(["sysctl", "-n", "hw.model"]).stdout.strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "loadavg": list(os.getloadavg()),
        "memsize_bytes": int(_run(["sysctl", "-n", "hw.memsize"]).stdout.strip() or 0),
    }


def _mlx_version(venv: Path) -> str:
    try:
        return md.version("mlx")
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# A1: real streaming bandwidth
# --------------------------------------------------------------------------- #

A1_SRC = r"""
import json, statistics, sys, time
import mlx.core as mx

N = 1 << 30           # 1 GiB
WARMUP, ROUNDS = 2, 5
a = mx.random.uniform(shape=(N // 4,), dtype=mx.float32)
b = mx.random.uniform(shape=(N // 4,), dtype=mx.float32)
mx.eval(a, b)
out = {"bytes": N, "warmup": WARMUP, "rounds": ROUNDS, "device": str(mx.default_device()),
       "mlx_version": __import__("importlib").metadata.version("mlx"), "copy": [], "reduce": []}

def timed(fn, rounds, bucket):
    for _ in range(WARMUP):
        fn(); mx.eval()
    for _ in range(rounds):
        mx.clear_cache()
        t0 = time.perf_counter()
        fn(); mx.eval()
        out[bucket].append({"ms": (time.perf_counter() - t0) * 1e3})

timed(lambda: mx.add(a, b, stream=mx.gpu), ROUNDS, "copy")   # read 2x1GiB, write 1GiB
timed(lambda: mx.sum(a), ROUNDS, "reduce")
def red_stats():
    for row in out["reduce"]:
        row["gbps"] = (N / 1e9) / (row["ms"] / 1e3)
for row in out["copy"]:
    row["gbps"] = (3 * N / 1e9) / (row["ms"] / 1e3)   # read 2 + write 1
red_stats()
out["copy_median_gbps"] = statistics.median(r["gbps"] for r in out["copy"])
out["copy_spread"] = (max(r["gbps"] for r in out["copy"]) - min(r["gbps"] for r in out["copy"]))
out["copy_spread_frac"] = out["copy_spread"] / out["copy_median_gbps"]
out["reduce_median_gbps"] = statistics.median(r["gbps"] for r in out["reduce"])
out["reduce_spread"] = (max(r["gbps"] for r in out["reduce"]) - min(r["gbps"] for r in out["reduce"]))
out["reduce_spread_frac"] = out["reduce_spread"] / out["reduce_median_gbps"]
print(json.dumps(out))
"""


def run_a1(res: Path) -> dict:
    proc = _run([str(LANE_VENV / "bin/python"), "-I", "-c", A1_SRC], timeout=600)
    if proc.returncode != 0:
        _dump(res, "a1-error.json", {"stderr": proc.stderr[-4000:], "rc": proc.returncode})
        raise RuntimeError(f"A1 failed: {proc.stderr[-500:]!r}")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload["valid"] = payload["copy_spread_frac"] <= 0.05 and payload["reduce_spread_frac"] <= 0.05
    _dump(res, "a1.json", payload)
    _log(res, f"A1 copy_median_gbps={payload['copy_median_gbps']:.2f} "
              f"spread_frac={payload['copy_spread_frac']:.4f} "
              f"reduce_median_gbps={payload['reduce_median_gbps']:.2f} "
              f"spread_frac={payload['reduce_spread_frac']:.4f} valid={payload['valid']}")
    return payload


# --------------------------------------------------------------------------- #
# A2: lane ctx ladder + reference arm
# --------------------------------------------------------------------------- #

A2_LX_SRC = r"""
import argparse, json, statistics, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--token-ids", required=True)
p.add_argument("--gen", type=int, default=128)
p.add_argument("--steps", type=int, default=16)   # steady-state engine step count
a = p.parse_args()

import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer
import importlib.metadata as md

tok = load_tokenizer(Path(a.model))
# txids = [int(x) for x in Path(a.token_ids).read_text().split(",")]
# print(json.dumps({"backend":"mlx-lm load_model","model":a.model,"tx":len(txids)}), file=sys.stderr)
# model = load_model(Path(a.model), lazy=True)
# logits = model(mx.array([txids]))
# mx.eval(logits)
# next_id = int(mx.argmax(logits[0, -1]).item())
# steps = []
# for _ in range(a.steps):
#     t0 = time.perf_counter()
#     logits = model(mx.array([[next_id]]))
#     mx.eval(logits)
#     steps.append((time.perf_counter() - t0) * 1e3)
#     next_id = int(mx.argmax(logits[0, -1]).item())
# print(json.dumps({"step_ms": steps, "median": statistics.median(steps)}))
# (chunk kept in history for audit; active path below)
model = load_model(Path(a.model), lazy=True)
txids = [int(x) for x in Path(a.token_ids).read_text().split(",")]
logits = model(mx.array([txids]))
mx.eval(logits)
next_id = int(mx.argmax(logits[0, -1]).item())
steps = []
for _ in range(a.steps):
    t0 = time.perf_counter()
    logits = model(mx.array([[next_id]]))
    mx.eval(logits)
    steps.append((time.perf_counter() - t0) * 1e3)
    next_id = int(mx.argmax(logits[0, -1]).item())
print(json.dumps({"step_ms": steps, "median": statistics.median(steps),
                  "mlx": md.version("mlx"), "backend": "mlx_lm.load_model"}))
"""


A2_LX_LONG_SRC = r"""
import argparse, json, statistics, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--gen", type=int, default=128)
p.add_argument("--steps", type=int, default=16)
p.add_argument("--chunk", type=int, default=512)
a = p.parse_args()
import mlx.core as mx
from mlx_lm.utils import load_model, load_tokenizer
import importlib.metadata as md
tok = load_tokenizer(Path(a.model))
model = load_model(Path(a.model), lazy=True)
txids = [int(x) for x in sys.stdin.read().split(",") if x.strip()]
if not txids:
    print(json.dumps({"error": "no tokens"})); sys.exit(3)
logits = None
# chunked prefill: feed the full prompt in fixed-size windows; each call's argmax
# is the token the model would emit after that window; final argmax is the
# steady-state seed. No cross-prompt cache state.
for start in range(0, len(txids), a.chunk):
    chunk = txids[start:start + a.chunk]
    logits = model(mx.array([chunk]))
    mx.eval(logits)
next_id = int(mx.argmax(logits[0, -1]).item())
steps = []
for _ in range(a.steps):
    t0 = time.perf_counter()
    logits = model(mx.array([[next_id]]))
    mx.eval(logits)
    steps.append((time.perf_counter() - t0) * 1e3)
    next_id = int(mx.argmax(logits[0, -1]).item())
print(json.dumps({"step_ms": steps, "median": statistics.median(steps),
                  "mlx": md.version("mlx"), "backend": "mlx_lm.load_model"}))
"""


def _lx_step(receipt: Path, venv: Path, model: Path, tokens: Path, *,
             long: bool = False, gen: int = 128) -> dict | None:
    """Run the mlx-lm reference arm; returns parsed JSON or None with stderr kept."""
    src = A2_LX_LONG_SRC if long else A2_LX_SRC
    if long:
        cmd = [str(venv / "bin/python"), "-I", "-c", src,
               "--model", str(model), "--gen", str(gen)]
        proc = subprocess.run(cmd, input=tokens.read_text(), capture_output=True, text=True,
                              timeout=600)
    else:
        cmd = [str(venv / "bin/python"), "-I", "-c", src,
               "--model", str(model), "--token-ids", str(tokens), "--gen", str(gen)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    (receipt / "lx-stderr.log").open("a").write(f"\n--- {tokens.name} rc={proc.returncode}\n{proc.stderr[-2000:]}\n")
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as error:  # noqa: BLE001
        (receipt / "lx-stderr.log").open("a").write(f"parse-failed {error!r} {proc.stdout[-400:]}\n")
        return None


class LaneWorker:
    """apxinf-worker/2.0 process with prepare_input + submit (ledger basis)."""

    def __init__(self, res: Path, ledger: Path, *, max_context: int = 32768,
                 prefix_cache: str = "off"):
        self.res = res
        self.proc = subprocess.Popen(
            [str(LANE_VENV / "bin/python"), "-u", str(WORKER_SCRIPT),
             "--worker-epoch", str(uuid.uuid4()), "--model", str(PACKED4),
             "--max-context", str(max_context), "--max-output-tokens", str(GEN),
             "--prefill-step-size", "256", "--output-batch-tokens", "1",
             "--memory-limit-bytes", str(10 * 1024**3), "--prefix-cache", prefix_cache,
             "--ledger", str(ledger), "--engine-root", str(ENGINE_ROOT),
             "--snapshot-interval", "256"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=(res / "lane-stderr.log").open("a"),
            text=True, bufsize=1,
            env={k: v for k, v in os.environ.items() if not k.startswith(("ENGINETAILOR", "APXINF"))},
        )
        self.ready = self._read()
        if self.ready.get("kind") != "ready":
            raise RuntimeError(f"lane worker did not become ready: {self.ready}")

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("lane worker closed stdout")
        return json.loads(line)

    def _send(self, frame: dict) -> None:
        self.proc.stdin.write(json.dumps(frame, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def prepare(self, messages: list[dict]) -> list[int]:
        cid = str(uuid.uuid4())
        self._send({"protocol": "apxinf-worker/2.0", "kind": "prepare_input",
                    "worker_epoch": self.ready["worker_epoch"], "command_id": cid,
                    "model_revision": self.ready["model_revision"], "messages": messages,
                    "tools": [], "template_options": {"enable_thinking": False}})
        while True:
            frame = self._read()
            if frame.get("command_id") == cid:
                if frame["kind"] != "prepared_input":
                    raise RuntimeError(f"prepare failed: {frame}")
                return [int(t) for t in frame["token_ids"]]

    def submit(self, tokens: list[int], max_tokens: int = GEN) -> dict:
        request_id = str(uuid.uuid4())
        self._send({"protocol": "apxinf-worker/2.0", "kind": "submit",
                    "worker_epoch": self.ready["worker_epoch"], "command_id": str(uuid.uuid4()),
                    "request_id": request_id, "attempt": 1,
                    "model_revision": self.ready["model_revision"],
                    "capability_revision": self.ready["capability_revision"],
                    "token_ids": tokens, "max_tokens": max_tokens,
                    "remaining_timeout_ms": 3_600_000,
                    "eos_token_ids": self.ready["eos_token_ids"], "capacity_lease_ids": []})
        emitted: list[int] = []
        terminal = None
        while True:
            frame = self._read()
            if frame.get("kind") == "tokens":
                emitted.extend(int(t) for t in frame["token_ids"])
            elif frame.get("kind") == "terminal":
                terminal = frame
                break
        while True:
            frame = self._read()
            if frame.get("kind") == "resources_released":
                break
        terminal["request_id"] = request_id
        terminal["emitted_tokens"] = emitted
        return terminal

    def shutdown(self) -> None:
        try:
            self._send({"protocol": "apxinf-worker/2.0", "kind": "shutdown",
                        "worker_epoch": self.ready["worker_epoch"],
                        "command_id": str(uuid.uuid4())})
            self.proc.stdin.close()
            self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()


def _filler_tokens(tx_path: Path, count: int) -> list[int]:
    """Deterministic token list from the lane tokenizer, no model needed."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(PACKED4 / "tokenizer.json"))
    text = ("The sky is blue. " * (count // 3 + 1)).strip()
    ids = tok.encode(text).ids
    # Avoid EOS in the middle of the prompt (would trip decode-time stop logic
    # only in the reference arm; lane worker uses eos_token_ids on output).
    ids = [int(t) for t in ids if int(t) != 124900]
    return ids[:count]


def _txfile(res: Path, name: str, ids: list[int]) -> Path:
    path = res / f"tx-{name}.csv"
    path.write_text(",".join(str(int(i)) for i in ids))
    return path


def _ledger_rows(path: Path, request_ids: set[str]) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("request_id") in request_ids:
            rows.append(row)
    return rows


def run_a2(res: Path, ctxs: tuple[int, ...]) -> dict:
    """Two-arm ladder: lane worker ledger step_ms vs mlx-lm load_model step_ms."""
    ledger = res / "a2-lane-ledger.jsonl"
    ledger.unlink(missing_ok=True)
    lane = LaneWorker(res, ledger)
    out: dict[str, Any] = {"ctxs": list(ctxs), "gen": GEN, "repeats": REPEATS,
                           "lane": {}, "reference": {}, "reference_status": {}}
    lx_tokens_src = BF16
    lx_model = SMALL  # dense HF-architecture sanity arm; see §5 lane note
    if not lx_model.is_dir():
        lx_model = None
    for ctx in ctxs:
        n_fill = max(16, ctx - 64)  # reserve room for the template wrapper
        for rep in range(REPEATS):
            ids = _filler_tokens(res, n_fill + 8)
            tx = _txfile(res, f"ctx{ctx}-r{rep}", ids)
            # -- lane arm (prepare -> submit; ledger is the engine basis) --
            msgs = [{"role": "system", "content": "You are a terse assistant."},
                    {"role": "user",
                     "content": "Reply with a short answer. " + "word " * n_fill}]
            try:
                lane_ids = lane.prepare(msgs)
                term = lane.submit(lane_ids[: ctx], GEN)
                out["lane"].setdefault(str(ctx), []).append(
                    {"rep": rep, "terminal": term, "tx_path": str(tx),
                     "token_count": len(lane_ids), "request_id": term["request_id"]})
            except Exception as error:  # noqa: BLE001
                out["lane"].setdefault(str(ctx), []).append(
                    {"rep": rep, "error": repr(error)})
                break
            # -- reference arm: same ctx token list (engine-boundary argmax) --
            if lx_model is not None:
                row = _lx_step(res, LX_VENV, lx_model, tx)
                out["reference"].setdefault(str(ctx), []).append(row)
                if row is None:
                    out["reference_status"][str(ctx)] = "failed"
    lane.shutdown()

    all_ids = {r["request_id"] for v in out["lane"].values() for r in v
               if r.get("request_id")}
    rows = _ledger_rows(ledger, all_ids)
    for ctx_key, runs in out["lane"].items():
        req_ids = {r["request_id"] for r in runs if r.get("request_id")}
        matching = [r for r in rows if r.get("request_id") in req_ids]
        vals = []
        for m in matching:
            if m.get("output_tokens", 0) and m.get("decode_ms", 0.0) > 0:
                vals.append(m["decode_ms"] / m["output_tokens"])
        out["lane"][ctx_key] = {
            "step_ms_per_token": vals,
            "median": statistics.median(vals) if vals else None,
            "spread_frac": _spread(vals) if vals else None,
            "rows": matching,
        }
        # reference arm summary
        ref_vals = [r["median"] for r in out["reference"].get(ctx_key, []) if r]
        out["reference"][ctx_key] = {
            "step_ms_median": statistics.median(ref_vals) if ref_vals else None,
            "repeats": ref_vals,
            "spread_frac": _spread(ref_vals) if ref_vals else None,
        }
    # Gate: every lane cell needs >=1 ledger value; missing median => INVALID,
    # never a silent pass (RUNNER §4 — no median-fixing on empty cells).
    lane_cells_ok = all(
        v["median"] is not None and v["spread_frac"] is not None
        for v in out["lane"].values())
    lane_aa_ok = all(v["spread_frac"] <= 0.05 for v in out["lane"].values()
                     if v["spread_frac"] is not None)
    ref_vals_present = [v for v in out["reference"].values()
                        if v["step_ms_median"] is not None]
    ref_ok = (not ref_vals_present) or all(v["spread_frac"] <= 0.05
                                           for v in ref_vals_present)
    out["valid"] = lane_cells_ok and lane_aa_ok and ref_ok
    _dump(res, "a2.json", out)
    _log(res, f"A2 lane medians={[out['lane'][str(c)]['median'] for c in ctxs]} "
              f"ref={[out['reference'][str(c)]['step_ms_median'] for c in ctxs]} "
              f"lane_cells_ok={lane_cells_ok} valid={out['valid']}")
    return out


# --------------------------------------------------------------------------- #
# A3: GPU busy vs wall clock
# --------------------------------------------------------------------------- #

def _xctrace_available() -> tuple[bool, str]:
    """Probe template + tracing permission without touching the lock."""
    templates = _run(["/usr/bin/xctrace", "list", "templates"])
    if templates.returncode != 0 or "Metal System Trace" not in templates.stdout:
        return False, f"template unavailable: rc={templates.returncode} {templates.stderr[-200:]}"
    return True, "Metal System Trace template present"


def run_a3(res: Path) -> dict:
    out: dict[str, Any] = {"trace_ok": False, "mode": None}
    trace_ok, note = _xctrace_available()
    out["trace_probe"] = note
    trace_path = res / "a3"
    if trace_ok:
        # Capture a short resident decode through the lane worker under xctrace.
        ledger = res / "a3-lane-ledger.jsonl"
        ledger.unlink(missing_ok=True)
        lane = LaneWorker(res, ledger, max_context=8192)
        ids = _filler_tokens(res, 512)
        msgs = [{"role": "system", "content": "You are a terse assistant."},
                {"role": "user", "content": "Reply with one sentence. " + "word " * 440}]
        try:
            lane_ids = lane.prepare(msgs)
            trace_dir = res / "a3-trace"
            trace_dir.mkdir(exist_ok=True)
            cmd = ["/usr/bin/xctrace", "record", "--template", "Metal System Trace",
                   "--output", str(trace_path), "--time-limit", "60s",
                   "--attach", str(lane.proc.pid), "--non-interactive"]
            # xctrace attach needs the target alive and (usually) a host that
            # allows task_for_pid; if it fails we fall back below.
            proc = subprocess.Popen(cmd, stdout=(res / "a3-xctrace.log").open("w"),
                                     stderr=subprocess.STDOUT, text=True)
            time.sleep(3)
            term = lane.submit(lane_ids, GEN)
            proc.wait(timeout=90)
            out["lane_terminal"] = term
            out["trace_rc"] = proc.returncode
            # join before judging: the trace file is written on exit.
            out["trace_ok"] = proc.returncode == 0 and trace_path.exists()
        except Exception as error:  # noqa: BLE001
            out["trace_error"] = repr(error)
        finally:
            lane.shutdown()
    # Fallback path (Issue #135 wording): per-step decomposition + bytes/BW.
    # Probe attribution: engine begin_probe() records per-component synchronized
    # splits; we use it here only for the *share* / mechanism, not headline latency.
    fallback = _a3_fallback(res)
    out["fallback"] = fallback
    out["mode"] = "trace+fallback" if out["trace_ok"] else "fallback"
    _dump(res, "a3.json", out)
    _log(res, f"A3 mode={out['mode']} trace_ok={out['trace_ok']} "
              f"fallback_gpu_ms={fallback.get('gpu_ms_per_token')}")
    return out


A3_PROBE_SRC = r"""
import argparse, json, statistics, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--root", required=True); p.add_argument("--steps", type=int, default=32)
a = p.parse_args()
sys.path.insert(0, a.root)
import mlx.core as mx
from enginetailor.models.lfm3.runtime import Engine
from enginetailor.models.lfm3 import contract
from tokenizers import Tokenizer

ids = [int(x) for x in Path(__import__('os').environ["A3_TOKENS"]).read_text().split(",") if x]
eng = Engine(root=__import__('os').environ["A3_MODEL"], dtype="bfloat16", quant="packed4")
t0 = time.perf_counter(); logits = eng.prefill(ids, last_only=True); prefill_ms = (time.perf_counter()-t0)*1e3
tok = int(mx.argmax(mx.array(logits[0])).item())
eng.begin_probe()
wall = []
for _ in range(a.steps):
    t0 = time.perf_counter(); logits = eng.step(tok); wall.append((time.perf_counter()-t0)*1e3)
    tok = int(mx.argmax(mx.array(logits[0])).item())
probe = eng.end_probe()
print(json.dumps({"prefill_ms": prefill_ms, "wall_step_ms": wall,
                  "wall_median": statistics.median(wall), "probe": probe,
                  "probe_total_ms": sum(v["ms"] for v in probe.values())}))
"""


def _a3_fallback(res: Path) -> dict:
    """Per-step decomposition via engine probe + bytes / measured BW."""
    a1 = json.loads((res / "a1.json").read_text())
    bw = a1.get("reduce_median_gbps") or a1.get("copy_median_gbps")
    ids = _filler_tokens(res, 512)
    tx = _txfile(res, "a3-probe", ids)
    env = dict(os.environ, A3_TOKENS=str(tx), A3_MODEL=str(PACKED4))
    proc = _run([str(LANE_VENV / "bin/python"), "-I", "-c", A3_PROBE_SRC,
                 "--root", str(ENGINE_ROOT)], env=env, timeout=900)
    (res / "a3-probe-stderr.log").open("a").write(proc.stderr[-3000:])
    payload = None
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            payload = None
    out: dict[str, Any] = {"probe_returncode": proc.returncode, "probe": payload,
                           "bw_gbps": bw}
    if payload and bw:
        step_ms = payload["wall_median"]
        # Weight bytes per token: packed4 checkpoint ~0.973 GB (all matmuls packed).
        weight_bytes = 973_083_386
        gpu_ms = (weight_bytes / 1e9) / bw * 1e3
        out["weight_bytes"] = weight_bytes
        out["gpu_ms_per_token"] = gpu_ms
        out["wall_ms_per_token"] = step_ms
        out["host_or_kernel_gap_ms"] = step_ms - gpu_ms
        probe_total = payload.get("probe_total_ms")
        if probe_total and step_ms:
            out["probe_share_of_wall"] = probe_total / step_ms
    _dump(res, "a3-fallback.json", out)
    return out


# --------------------------------------------------------------------------- #
# prefill AB: packed4 vs bf16
# --------------------------------------------------------------------------- #

PREFILL_AB_SRC = r"""
import argparse, json, statistics, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--quant", required=True)
p.add_argument("--tokens", required=True); p.add_argument("--repeats", type=int, default=3)
a = p.parse_args()
import mlx.core as mx
from enginetailor.models.lfm3.runtime import Engine
ids = [int(x) for x in Path(a.tokens).read_text().split(",") if x]
eng = Engine(root=a.model, dtype="bfloat16", quant=(None if a.quant == "none" else a.quant))
ms = []
for _ in range(a.repeats):
    t0 = time.perf_counter()
    logits = eng.prefill(ids, last_only=True)
    mx.eval(logits)
    ms.append((time.perf_counter() - t0) * 1e3)
print(json.dumps({"prefill_ms": ms, "median": statistics.median(ms),
                  "spread_frac": (max(ms)-min(ms))/statistics.median(ms),
                  "tokens": len(ids), "quant": a.quant,
                  "packed_bytes": (eng.last_stats.get("packed_bytes") if a.quant != "none" else None)}))
"""


def run_prefill_ab(res: Path, tokens_n: int = 2048) -> dict:
    ids = _filler_tokens(res, tokens_n)
    tx = _txfile(res, "prefill", ids)
    env = dict(os.environ)
    env["ENGINETAILOR_LFM3_QUANT_ROOT"] = str(PACKED4)
    rows = {}
    for tag, model, quant in (("packed4", PACKED4, "packed4"), ("bf16", BF16, "none")):
        proc = _run([str(LANE_VENV / "bin/python"), "-I", "-c", PREFILL_AB_SRC,
                     "--model", str(model), "--quant", quant, "--tokens", str(tx),
                     "--repeats", str(REPEATS)], env=env, timeout=1200)
        (res / f"prefill-{tag}-stderr.log").open("w").write(proc.stderr[-3000:])
        payload = None
        if proc.returncode == 0:
            try:
                payload = json.loads(proc.stdout.strip().splitlines()[-1])
            except Exception:
                payload = None
        rows[tag] = payload
    valid = all(rows[t] and rows[t]["spread_frac"] <= 0.05 for t in rows)
    out = {"tokens": tokens_n, "repeats": REPEATS, "arms": rows, "valid": valid}
    if all(rows.values()):
        out["packed4_over_bf16"] = rows["packed4"]["median"] / rows["bf16"]["median"]
    _dump(res, "prefill-ab.json", out)
    _log(res, f"prefillAB packed4={rows['packed4'] and rows['packed4']['median']} "
              f"bf16={rows['bf16'] and rows['bf16']['median']} valid={valid}")
    return out


# --------------------------------------------------------------------------- #
# Window driver
# --------------------------------------------------------------------------- #

class Locks:
    """Both fcntl locks, in fixed order, waited on (never stolen)."""

    def __init__(self, paths: list[Path] = LOCK_PATHS):
        self.paths = [Path(p) for p in paths]
        self.fds: list[int] = []

    def acquire(self, deadline_s: float = 21600.0) -> None:
        end = time.time() + deadline_s
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() > end:
                        os.close(fd)
                        raise TimeoutError(f"lock wait deadline ({deadline_s}s): {path}")
                    time.sleep(5)
            self.fds.append(fd)

    def release(self) -> None:
        for fd in reversed(self.fds):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        self.fds.clear()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


def run_window(res: Path, ctxs: tuple[int, ...]) -> int:
    res.mkdir(parents=True, exist_ok=True)
    machine = _machine()
    (res / "hw.txt").write_text(machine["hardware_profile"])
    _dump(res, "machine.json", machine)
    if "M1 Pro" not in machine["brand_string"]:
        _log(res, f"WINDOW_ABORT machine mismatch: {machine['brand_string']}")
        return 101
    _log(res, f"window start machine={machine['brand_string']} ctxs={ctxs}")
    locks = Locks()
    try:
        locks.acquire()
    except TimeoutError as error:
        _log(res, f"lock timeout: {error}")
        return 3
    (res / "locks.log").open("a").write(f"LOCKS_ACQUIRED {_now()} both\n")
    _log(res, "LOCKS_ACQUIRED both")
    caff = subprocess.Popen(["caffeinate", "-si"])
    try:
        _run(["pmset", "displaysleepnow"])  # window discipline
        t0 = time.time()
        results = {}
        try:
            results["a1"] = run_a1(res)
        except Exception as error:  # noqa: BLE001
            results["a1"] = {"error": repr(error)}
            _log(res, f"A1 FAILED: {error!r}")
        try:
            results["a2"] = run_a2(res, ctxs)
        except Exception as error:  # noqa: BLE001
            results["a2"] = {"error": repr(error)}
            _log(res, f"A2 FAILED: {error!r}")
        try:
            results["a3"] = run_a3(res)
        except Exception as error:  # noqa: BLE001
            results["a3"] = {"error": repr(error)}
            _log(res, f"A3 FAILED: {error!r}")
        try:
            results["prefill_ab"] = run_prefill_ab(res)
        except Exception as error:  # noqa: BLE001
            results["prefill_ab"] = {"error": repr(error)}
            _log(res, f"prefill FAILED: {error!r}")
        _dump(res, "window.json", {
            "started": _now(), "machine": machine, "ctxs": list(ctxs),
            "duration_s": time.time() - t0, "results": results,
        })
        _log(res, f"window done duration_s={time.time() - t0:.1f}")
        return 0
    finally:
        caff.terminate()
        locks.release()
        (res / "locks.log").open("a").write(f"LOCKS_RELEASED {_now()} both\n")
        _log(res, "LOCKS_RELEASED both")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_res(p):
        p.add_argument("--res", type=Path, default=Path.home() / "lfm3-dig-run/bw-util")

    win = sub.add_parser("window", help="all four steps under both locks")
    add_res(win)
    win.add_argument("--ctxs", default=",".join(str(c) for c in CTXS))
    for name in ("a1", "a3", "prefill"):
        p = sub.add_parser(name); add_res(p)
    a2 = sub.add_parser("a2"); add_res(a2); a2.add_argument("--ctxs", default=",".join(str(c) for c in CTXS))
    args = ap.parse_args(argv)
    res: Path = args.res.expanduser().resolve()
    if args.cmd == "window":
        ctxs = tuple(int(x) for x in args.ctxs.split(",") if x.strip())
        return run_window(res, ctxs)
    if args.cmd == "a1":
        return 0 if run_a1(res).get("valid") else 1
    if args.cmd == "a3":
        return 0 if run_a3(res).get("mode") else 1
    if args.cmd == "prefill":
        return 0 if run_prefill_ab(res).get("valid") else 1
    if args.cmd == "a2":
        ctxs = tuple(int(x) for x in args.ctxs.split(",") if x.strip())
        return 0 if run_a2(res, ctxs).get("valid") else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
