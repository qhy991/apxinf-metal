#!/usr/bin/env python3
"""G1' evaluator: model-emitted code -> solution.py -> pristine tests.

Verbatim copy of the audited action template
`~/ag-run/agents/code_loop.py` (PV-11) with ONE behavioural change and
its justification (G1' needs 解出耗时, which the original discards):

  * each attempt also records engine-reported `prefill_ms` /
    `decode_ms_per_token` by reading the gateway's `--request-log`
    (one row per request, joined by response id / ordered fallback —
    the exact schema scripts/bench_engines_request_log.py joins, which
    is how the G4-speed window got its engine-basis numbers).

Everything else is byte-identical in behaviour: prompt text,
code-fence extraction, feedback format, retry loop, pass criterion
(`returncode==0 and "OK" in stdout`). The frozen task payloads and
pristine tests come from suite.json; no gold answer is ever passed to
the model.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.S)
BODY_HINT = re.compile(r"(def\s+\w+\s*\(.*)", re.S)


def ask(gateway: str, model: str, prompt: str, max_tokens: int, timeout: int) -> str:
    """One plain completion - no tools, so the gateway cannot reject a tool shape."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    url = gateway.rstrip("/") + "/v1/chat/completions"
    proc = subprocess.run(
        ["curl", "-s", "-m", str(timeout), "-X", "POST", url,
         "-H", "content-type: application/json", "-H", "x-api-key: sk-local",
         "-d", json.dumps(payload)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("gateway call failed: %s" % proc.stderr[-300:])
    doc = json.loads(proc.stdout)
    if "choices" not in doc:
        # Gateway errors (e.g. max_tokens over the deployment limit) are
        # {"type":"error","error":{...}}; surface the body instead of a
        # KeyError with no context (round-57 seg-0 failure diagnosis).
        raise RuntimeError("gateway returned an error: %s" % proc.stdout[:400])
    return doc["choices"][0]["message"].get("content") or ""


def extract_code(reply: str) -> str:
    """Take the last fenced block, else from the first `def`; never invent code."""
    blocks = CODE_FENCE.findall(reply)
    if blocks:
        return blocks[-1].strip() + "\n"
    m = BODY_HINT.search(reply)
    return (m.group(1).strip() + "\n") if m else ""


def prompt_for(task: dict, feedback) -> str:
    head = ("Write ONLY the Python implementation, no explanation and no shell commands.\n"
            "The file must define the requested function and be runnable as-is.\n\n"
            f"{task['problem']}\n\n")
    if task.get("prompt_stub"):
        head += "Start from this signature:\n" + task["prompt_stub"] + "\n\n"
    if feedback:
        head += ("Your previous attempt failed with:\n" + feedback +
                 "\nReturn the FULL corrected implementation in one python code block.\n")
    return head


def tests_for(task: dict) -> str:
    parts = []
    if task.get("test_imports"):
        parts.append(task["test_imports"])
    parts.append("from solution import *\n")
    if task["benchmark"] == "humaneval":
        parts.append(task["test_code"])
        parts.append(f"\ncheck({task['entry_point']})\nprint('OK')\n")
    else:
        parts.append(task["test_code"] + "\n")
        parts.append("print('OK')\n")
    return "\n".join(parts)


def ledger_rows(path: str) -> list[dict]:
    """Engine-reported durations, one row per request, in file order
    (schema is the gateway --request-log that bench_engines joins)."""
    rows = []
    if not path:
        return rows
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("request_id"):
                rows.append(row)
    except OSError:
        return rows
    return rows


def join_timing(rows: list[dict], claimed: set[str]) -> dict:
    """Next unclaimed ledger row in file order (one evaluator request ->
    one ledger row; ordering is guaranteed because the gateway is
    exclusively owned by this run and the requests are serial)."""
    for row in rows:
        rid = row["request_id"]
        if rid in claimed:
            continue
        claimed.add(rid)
        return {"engine_prefill_ms": row.get("prefill_ms"),
                "engine_decode_ms_per_token": row.get("decode_ms_per_token"),
                "engine_decode_ms": row.get("decode_ms"),
                "engine_completion_tokens": row.get("completion_tokens"),
                "engine_source": "apxinf.ledger",
                "engine_request_id": rid}
    return {}


def run_task(task: dict, args, ledger: list[dict], claimed: set[str]) -> dict:
    work = Path(tempfile.mkdtemp(prefix="codeloop-"))
    (work / "test_solution.py").write_text(tests_for(task))
    feedback = None
    attempts = []
    t0 = time.time()
    for attempt in range(1, args.retries + 1):
        t_ask = time.time()
        reply = ask(args.gateway, args.model, prompt_for(task, feedback),
                    args.max_tokens, args.timeout)
        wall = time.time() - t_ask
        timing = join_timing(ledger, claimed) if args.request_log else {}
        code = extract_code(reply)
        if not code:
            feedback = "no code block found in the reply"
            attempts.append({"attempt": attempt, "written": False,
                             "wall_s": round(wall, 3), "timing": timing})
            continue
        (work / "solution.py").write_text(code)
        proc = subprocess.run([args.python, "test_solution.py"], cwd=work,
                              capture_output=True, text=True, timeout=args.timeout)
        ok = proc.returncode == 0 and "OK" in proc.stdout
        attempts.append({"attempt": attempt, "written": True, "ok": ok,
                          "wall_s": round(wall, 3), "timing": timing,
                          "tail": (proc.stdout + proc.stderr)[-300:]})
        if ok:
            return {"task_id": task["id"], "solved": True, "attempts": attempts,
                    "attempt_count": attempt,
                    "solve_wall_s": round(time.time() - t0, 3),
                    "work": str(work)}
        feedback = (proc.stdout + proc.stderr)[-600:]
    return {"task_id": task["id"], "solved": False, "attempts": attempts,
            "attempt_count": len(attempts),
            "solve_wall_s": round(time.time() - t0, 3),
            "work": str(work)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--gateway", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--request-log", default="",
                    help="gateway --request-log; engine prefill/decode per attempt (G1')")
    args = ap.parse_args()

    suite = json.loads(Path(args.suite).read_text())
    wanted = set(x.strip() for x in args.tasks.split(",") if x.strip())
    tasks = [t for t in suite["tasks"]
             if t.get("kind") == "code" and (not wanted or t["id"] in wanted)]
    if not tasks:
        print("no code tasks selected", file=sys.stderr)
        return 2
    ledger = ledger_rows(args.request_log)
    claimed: set[str] = set()
    results = []
    for task in tasks:
        r = run_task(task, args, ledger, claimed)
        results.append(r)
        print(f"[{task['id']}] solved={r['solved']} attempts={r['attempt_count']} "
              f"wall_s={r['solve_wall_s']}")
    doc = {"suite": str(args.suite), "model": args.model,
           "gateway": args.gateway, "retries": args.retries,
           "max_tokens": args.max_tokens,
           "request_log": args.request_log or None,
           "n": len(results), "solved": sum(1 for r in results if r["solved"]),
           "results": results}
    Path(args.out).write_text(json.dumps(doc, indent=2))
    print(f"solved {doc['solved']}/{doc['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
