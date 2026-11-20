"""Standalone analysis for Issue #134 ANE+GPU heterogeneous measurements.

Reads the probe JSON receipts written by scripts/run-ane-measurements.sh and
the T1 request ledgers, computes per-cell medians and A/A spread, and prints a
verdict plus writes ane-verdict.json. No heredoc, no crates/ changes.

Exit codes (the driver treats them as the window verdict):
  0    every required probe arm ok and every cell A/A <= 5%
  102  one or more required probe arms missing or A/A > 5%
  103  window blocked (probe binary not built / not present)
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path


def median(values):
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def spread(values):
    """(max-min)/median, matching run-bw-util.py::_spread."""
    med = median(values)
    if len(values) < 2 or med <= 0:
        return 0.0
    return (max(values) - min(values)) / med


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_probe_arm(res, arm):
    """Load the probe receipt for one arm. Returns (payload, error)."""
    raw = res / "raw" / f"{arm}.json"
    if not raw.is_file():
        return None, f"missing receipt {raw}"
    try:
        return read_json(raw), None
    except Exception as exc:  # noqa: BLE001 - receipt parse failure
        return None, f"bad receipt {raw}: {exc}"


def probe_ok(payload, arm):
    """One probe arm is ok when it has no error and a positive measurement.

    Field names are the schema emitted by crates/apxinf-model/examples/
    ane_probe.rs (copied verbatim from the receipt writers):
      gpu         copy_gbps / reduce_gbps
      ane-bw      gbps
      both        ane_gbps + gpu_arm.copy_gbps
      event       single_roundtrip_ms_median / rounds_ms_median
      matmul      matmul_ms_median
      placement   placement (facts only, no timing)
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("error"):
        return False
    if "copy_gbps" in payload or "reduce_gbps" in payload:
        return bool(payload.get("copy_gbps", 0) > 0 and payload.get("reduce_gbps", 0) > 0)
    if "gbps" in payload and payload.get("probe") == "ane_conv_weight_read":
        return bool(payload.get("gbps", 0) > 0)
    if "gpu_arm" in payload:  # T2 both: nested gpu_arm + ane_gbps
        gpu_arm = payload.get("gpu_arm") or {}
        return bool(
            payload.get("ane_gbps", 0) > 0
            and gpu_arm.get("copy_gbps", 0) > 0
        )
    if "single_roundtrip_ms_median" in payload or "rounds_ms_median" in payload:
        return bool(
            payload.get("single_roundtrip_ms_median", 0) > 0
            and payload.get("rounds_ms_median", 0) > 0
        )
    if payload.get("probe", "").startswith("matmul"):
        return bool(payload.get("matmul_ms_median", 0) > 0)
    if payload.get("placement") is not None:
        return True  # placement facts are an effect proof, no timing
    # Generic: any numeric measurement field is enough.
    return any(
        isinstance(v, (int, float)) and v > 0
        for k, v in payload.items()
        if k.endswith("_median") or k.endswith("_gbps") or k.endswith("_ms")
    )


def cell_values(payload, field):
    """Extract per-repetition timing values for one cell if present."""
    if not isinstance(payload, dict):
        return []
    values = payload.get("reps_ms") or payload.get("values") or payload.get("timings_ms")
    if isinstance(values, list):
        out = [float(v) for v in values if isinstance(v, (int, float))]
        if out:
            return out
    if field in payload and isinstance(payload[field], (int, float)):
        return [float(payload[field])]
    return []


def load_llama_t1(res):
    """Parse T1 llama receipts into batch -> engine-basis ms/token values.

    The request ledger (ane-w1-llama-requests.jsonl) is empty: bench_engines.sh
    does not populate it for this protocol. The real data is in the raw bench
    receipts raw/t1-B{b}-s2-*.json, per-request engine_decode_ms /
    engine_decode_tokens (engine basis, from llama-server --timings parsing).
    Warmup cases are skipped (their timing includes cold start).
    """
    out = {}
    for path in sorted((res / "raw").glob("t1-B*-s2-*.json")):
        m = re.search(r"t1-B(\d+)-", path.name)
        if not m:
            continue
        batch = int(m.group(1))
        try:
            payload = read_json(path)
        except Exception:
            continue
        requests = payload.get("requests") or []
        for row in requests:
            if not isinstance(row, dict):
                continue
            if row.get("case") == "__warmup__":
                continue
            if row.get("status") not in (None, "ok"):
                continue
            toks = row.get("engine_decode_tokens") or 0
            ms = row.get("engine_decode_ms") or 0
            if isinstance(toks, (int, float)) and isinstance(ms, (int, float)) and toks > 0:
                out.setdefault(batch, []).append(float(ms) / float(toks))
    return out


def eta_from_probe(both_payload, gpu_payload, ane_payload):
    """Compute eta = (GPU+ANE overlapped BW) / (GPU-only BW + ANE-only BW).

    The both arm reports nested `gpu_arm.copy_gbps` for the concurrent GPU
    stream and `ane_gbps` for the concurrent ANE conv read; the standalone
    arms report `copy_gbps` (gpu) and `gbps` (ane). Same byte budget in all
    three arms (ARRAY_BYTES / conv weight bytes), so the ratio is dimensionless.
    """
    if not isinstance(both_payload, dict):
        return None
    gpu_arm = both_payload.get("gpu_arm") or {}
    both_total = float(both_payload.get("ane_gbps", 0)) + float(gpu_arm.get("copy_gbps", 0))
    if gpu_payload is not None:
        gpu_only = float(gpu_payload.get("copy_gbps", 0))
    else:
        gpu_only = float(gpu_arm.get("copy_gbps", 0))
    if ane_payload is not None:
        ane_only = float(ane_payload.get("gbps", 0))
    else:
        ane_only = float(both_payload.get("ane_gbps", 0))
    denom = gpu_only + ane_only
    if not all(v > 0 for v in (both_total, denom)):
        return None
    return both_total / denom


def analyze(res, window):
    report = {"window": window, "arms": {}, "cells": {}, "errors": []}
    status = 0

    if window == "w1":
        required = ["A0-stream", "T2-gpu", "T2-ane", "T2-both"]
        for arm in required:
            payload, err = load_probe_arm(res, arm)
            ok = payload is not None and err is None and probe_ok(payload, arm)
            report["arms"][arm] = {"ok": ok, "error": err}
            if not ok:
                report["errors"].append(err or f"probe arm {arm} not ok")
                status = 102

        t1 = load_llama_t1(res)
        for batch in (1, 4, 8, 16):
            vals = t1.get(batch, [])
            if not vals:
                report["errors"].append(f"T1 missing batch B={batch}")
                status = 102
                continue
            med = median(vals)
            sp = spread(vals)
            report["cells"][f"T1-B{batch}"] = {
                "median": med,
                "spread": sp,
                "n": len(vals),
            }
            print(f"T1 B={batch}: median={med:.4f} ms/tok spread={sp*100:.2f}% n={len(vals)}")
            if sp > 0.05:
                report["errors"].append(f"T1 B={batch} A/A {sp*100:.2f}% > 5%")
                status = 102

        eta = None
        payloads = {}
        for arm in required:
            payload, _ = load_probe_arm(res, arm)
            payloads[arm] = payload
        if payloads.get("T2-both"):
            eta = eta_from_probe(
                payloads["T2-both"], payloads.get("T2-gpu"), payloads.get("T2-ane")
            )
        report["eta"] = eta
        if eta is not None:
            print(f"T2 eta = {eta:.4f}")
            if eta >= 0.85:
                report["verdict"] = "WIN: eta>=0.85 -> ANE adds no bandwidth; scenarios 1/2 excluded"
            elif eta < 0.70:
                report["verdict"] = "PARTIAL: eta<0.70 -> kernel work still open"
            else:
                report["verdict"] = "INCONCLUSIVE: eta in grey zone"
        else:
            report["verdict"] = "INCONCLUSIVE: eta not computable from probe receipts"
            if status == 0:
                status = 102

    elif window == "w2":
        required = ["T3-event"]
        for arm in required:
            payload, err = load_probe_arm(res, arm)
            ok = payload is not None and err is None and probe_ok(payload, arm)
            report["arms"][arm] = {"ok": ok, "error": err}
            if not ok:
                report["errors"].append(err or f"probe arm {arm} not ok")
                status = 102

        # T4 cells: every matmul cell must be present and not error. Placement
        # facts are an effect proof: missing placement is not fatal, but a
        # placement that says GPU for a T4 cell marks that cell as BLOCKED.
        cells = [
            "1x2048", "4x2048", "8x2048", "16x2048", "32x2048", "64x2048",
            "1x4096", "4x4096", "8x4096", "16x4096", "32x4096", "64x4096",
        ]
        for cell in cells:
            payload, err = load_probe_arm(res, f"T4-{cell}")
            if payload is None or err is not None:
                report["errors"].append(err or f"missing T4 cell {cell}")
                status = 102
                continue
            if not probe_ok(payload, "matmul"):
                report["errors"].append(f"T4 {cell} not ok: {payload.get('error')}")
                status = 102
                continue
            placement, perr = load_probe_arm(res, f"T4-{cell}-placement")
            anticipated = None
            if placement is not None and perr is None:
                usage = placement.get("placement") or {}
                preferred = usage.get("preferred")
                if isinstance(preferred, str):
                    anticipated = preferred
            report["cells"][f"T4-{cell}"] = {
                "ok": True,
                "placement": anticipated,
                "placement_error": perr,
            }
            if anticipated and anticipated.upper() not in ("ANE", "NEURALENGINE"):
                print(f"T4 {cell}: placement={anticipated} -> BLOCKED (matmul not ANE-anticipated)")
            else:
                print(f"T4 {cell}: ok placement={anticipated or 'n/a'}")

        report["verdict"] = (
            "PARTIAL: T4/T3 receipts present; cross-check placement facts "
            "against the six-scenario mapping before declaring a scenario closed"
        )
        if status != 0:
            report["verdict"] = "INVALID: one or more T4/T3 arms failed"

    else:
        print(f"unknown window {window}", file=sys.stderr)
        return 101

    report["status"] = "ok" if status == 0 else "invalid"
    out_path = res / "ane-verdict.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"verdict: {report.get('verdict')}")
    print(f"ANE_VERDICT={out_path}")
    if report["errors"]:
        for e in report["errors"]:
            print(f"ERROR: {e}")
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("res", help="receipt directory ($RUN/ane)")
    ap.add_argument("--window", required=True, choices=["w1", "w2"])
    args = ap.parse_args()
    res = Path(os.path.expanduser(args.res))
    if not res.is_dir():
        print(f"receipt directory missing: {res}", file=sys.stderr)
        return 102
    return analyze(res, args.window)


if __name__ == "__main__":
    sys.exit(main())
