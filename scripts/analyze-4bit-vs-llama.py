#!/usr/bin/env python3
"""Analyze a same-window 3-arm 4bit-vs-llama-Q4 speed window.

Frozen by doc/lfm3-4bit-vs-llama-q4.md (Lead 11:05):
  A = native-4bit   (native-4bit-A, segs 0/3/6)
  B = native-bf16   (native-bf16-B, segs 1/4/7)  — in-window sanity only
  C = llamacpp-Q4   (llamacpp-C,    segs 2/5/8)
  ratio = A / C per S2 cell; A/A = segment spread per arm per cell <= 5%.

No heredoc (round-55 lesson), no device access. Reads only raw bench JSONs.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

CELLS = ((1024, 128), (4096, 128))
ARMS = {"A": "native-4bit-A", "B": "native-bf16-B", "C": "llamacpp-C"}
SEGS = {"A": [0, 3, 6], "B": [1, 4, 7], "C": [2, 5, 8]}
CASE_RE = re.compile(r"s2-c(\d+)-g(\d+)-r\d+")


def main(raw_dir: Path) -> int:
    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        print(f"ERROR no raw json in {raw_dir}")
        return 2

    per_arm: dict[str, dict[tuple[int, int], list[float]]] = {a: {} for a in ARMS}
    per_arm_err: dict[str, dict[tuple[int, int], int]] = {a: {} for a in ARMS}
    provenance: list[str] = []
    errors: list[str] = []

    for path in raw_files:
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"unreadable {path.name}: {exc}")
            continue
        arm_label = doc.get("arm")
        seg = (doc.get("params") or {}).get("segment")
        role = next((r for r, name in ARMS.items() if name == arm_label), None)
        if role is None or seg is None:
            continue
        engine_source = sorted({r.get("engine_source") for r in doc.get("requests", [])
                                if r.get("status") == "ok" and not r.get("warmup")})
        arm_sha = (doc.get("artifact") or {}).get("binary_sha256", "")[:8]
        provenance.append(
            f"file={path.name} role={role} arm={arm_label} seg={seg} "
            f"binary_sha={arm_sha} engine_source={engine_source} failures={doc.get('failures')}"
        )
        if doc.get("failures"):
            errors.append(f"failures in {path.name}: {doc['failures']}")
        if role == "C" and engine_source != ["llama.cpp.timings"]:
            errors.append(
                f"INVALID llama arm {path.name}: engine_source={engine_source} "
                "expected ['llama.cpp.timings']"
            )

        for req in doc.get("requests", []):
            if req.get("warmup") or req.get("status") != "ok":
                continue
            match = CASE_RE.match(req.get("case") or "")
            if not match:
                continue
            if seg not in SEGS[role]:
                errors.append(
                    f"unexpected segment {seg} in {path.name} for role {role} "
                    f"(expected {SEGS[role]})"
                )
            cell = (int(match.group(1)), int(match.group(2)))
            if cell not in CELLS:
                continue
            ms = req.get("decode_ms_per_token")
            if ms is None:
                per_arm_err[role].setdefault(cell, 0)
                per_arm_err[role][cell] += 1
                continue
            per_arm[role].setdefault(cell, []).append(float(ms))

    print("=== PROVENANCE (per raw file) ===")
    for line in sorted(provenance):
        print(line)
    if errors:
        print("=== ERRORS ===")
        for line in errors:
            print(line)

    print("=== PER-ARM PER-CELL (segment medians, then cell median) ===")
    cell_medians: dict[str, dict[tuple[int, int], float]] = {a: {} for a in ARMS}
    aa_worst: dict[str, dict[tuple[int, int], float]] = {a: {} for a in ARMS}
    incomplete = False
    for role, name in ARMS.items():
        for cell in CELLS:
            vals = per_arm[role].get(cell, [])
            missing = len(SEGS[role]) - len(vals)
            err = per_arm_err[role].get(cell, 0)
            if missing > 0 or err:
                print(f"  {role} {name} cell={cell}: INCOMPLETE "
                      f"segs={len(vals)}/{len(SEGS[role])} missing={missing} no_ms={err} "
                      f"vals={[round(v, 4) for v in vals]}")
                incomplete = True
                continue
            med = statistics.median(vals)
            cell_medians[role][cell] = med
            aa = ((max(vals) - min(vals)) / med) * 100.0 if len(vals) > 1 else 0.0
            aa_worst[role][cell] = aa
            flag = " FAIL_A_A" if aa > 5.0 else ""
            print(f"  {role} {name} cell={cell}: n={len(vals)} "
                  f"vals={[round(v, 4) for v in vals]} cell_median={med:.4f} "
                  f"A/A={aa:.2f}%{flag}")

    print("=== RATIO A/C (frozen: <=0.95 WIN-confirmed / 0.95-1.00 slightly faster / >1.00 reversed) ===")
    verdicts: list[str] = []
    for cell in CELLS:
        a = cell_medians["A"].get(cell)
        c = cell_medians["C"].get(cell)
        if a is None or c is None:
            print(f"  cell={cell}: ratio unavailable (A={a} C={c})")
            incomplete = True
            verdicts.append("UNAVAILABLE")
            continue
        ratio = a / c
        if ratio <= 0.95:
            verdict = "4-bit same-window CONFIRMED faster than llama Q4 (>=5%)"
        elif ratio <= 1.00:
            verdict = "slightly faster (direction holds, magnitude in noise)"
        else:
            verdict = "prior cross-window conclusion REVERSED"
        verdicts.append(verdict)
        print(f"  cell={cell}: A={a:.4f} C={c:.4f} A/C={ratio:.4f} => {verdict}")

    print("=== IN-WINDOW SANITY: B/A speedup (expect ~3.3x, not in ticket verdict) ===")
    for cell in CELLS:
        b = cell_medians["B"].get(cell)
        a = cell_medians["A"].get(cell)
        if b is None or a is None:
            print(f"  cell={cell}: unavailable (B={b} A={a})")
            continue
        print(f"  cell={cell}: B/A={b / a:.2f}x (B={b:.4f} A={a:.4f})")

    if incomplete:
        print("=== OUTCOME: INVALID/INCOMPLETE (missing segments or no decode basis) ===")
        return 2
    if errors:
        print("=== OUTCOME: INVALID (provenance errors) ===")
        return 2
    aa_fail = any(v > 5.0 for by_cell in aa_worst.values() for v in by_cell.values())
    if aa_fail:
        print("=== OUTCOME: INVALID (A/A > 5%) ===")
        return 2
    overall = verdicts[0] if verdicts and verdicts[0] == verdicts[1] else " / ".join(verdicts)
    print(f"=== OUTCOME: VALID — {overall} ===")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <raw_dir>")
        raise SystemExit(64)
    raise SystemExit(main(Path(sys.argv[1])))
