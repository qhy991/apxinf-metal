#!/usr/bin/env python3
"""Render the LFM3-1.7B engine comparison from the harness's raw JSON.

The harness (scripts/bench_engines.sh) records one file per arm, stage and ABBA
segment. This report does only what a report may do:

  * prove the arms are comparable (corpus identity, token counts, A/A spread);
  * summarise paired, repeated medians (never a single-order number);
  * print the declared mixed factors next to the tables they qualify;
  * apply the pre-registered gate wording, and refuse to print a verdict when
    the data does not meet the gate (skills/perf-verdict-gate).

It deliberately does not choose thresholds after seeing the data: the gate text
is quoted from the campaign design and the script only compares numbers to it.

Usage:
  scripts/bench_engines_report.py                      # all stages in the raw dir
  scripts/bench_engines_report.py --stage s2 --baseline llamacpp
  scripts/bench_engines_report.py --json-out devlocal/engine-comparison-lfm3/summary.json

Exit status: 0 the data meets every pre-declared gate; 2 an arm's A/A spread
exceeds the tolerance; 3 an arm has no evidence that it generated a token;
4 an arm has fewer than --min-segments segments (a verdict cannot be claimed).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

SCHEMA = "apxinf-engine-bench-arm-v1"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = "devlocal/engine-comparison-lfm3/raw"

STAGE_TITLES = {
    "s1": "S1 prefill / TTFT",
    "s2": "S2 decode",
    "s3": "S3 serving-layer overhead",
    "s4": "S4 end-to-end agent suite",
}
PRIMARY_METRIC = {"s1": "ttft_ms", "s2": "decode_ms_per_token", "s3": "wall_ms", "s4": "wall_ms"}
PRIMARY_UNIT = {"s1": "ms", "s2": "ms/token", "s3": "ms", "s4": "ms"}


# --------------------------------------------------------------------- load ---
def load_runs(raw_dir: Path) -> tuple[list[dict], list[str]]:
    runs, problems = [], []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"{path.name}: unreadable ({error})")
            continue
        if document.get("schema") != SCHEMA:
            problems.append(f"{path.name}: not {SCHEMA}")
            continue
        document["_path"] = path.name
        runs.append(document)
    return runs, problems


def requests_of(run: dict) -> list[dict]:
    return [request for request in run.get("requests", []) if not request.get("warmup")]


def segment_of(run: dict) -> int:
    try:
        return int((run.get("params") or {}).get("segment", 0))
    except (TypeError, ValueError):
        return 0


# ------------------------------------------------------------------ metrics ---
def request_metrics(request: dict) -> dict:
    """Client-first derived metrics for one request; None where unavailable."""
    prompt_tokens = request.get("prompt_tokens")
    completion_tokens = request.get("completion_tokens")
    wall_ms = request.get("client_wall_ms")
    engine_prefill = request.get("engine_prefill_ms")
    engine_decode = request.get("engine_decode_ms")
    metrics: dict = {}

    client_ttft = request.get("client_ttft_ms")
    if client_ttft is not None:
        metrics["ttft_ms"] = client_ttft
        metrics["ttft_basis"] = "client"
    elif engine_prefill is not None:
        metrics["ttft_ms"] = engine_prefill
        metrics["ttft_basis"] = "engine"
    else:
        metrics["ttft_ms"] = None
        metrics["ttft_basis"] = None

    client_decode = request.get("client_decode_ms")
    if client_decode is not None and completion_tokens:
        metrics["decode_ms"] = client_decode
        metrics["decode_tokens"] = max(completion_tokens - 1, 1)
        metrics["decode_basis"] = "client"
    elif engine_decode is not None:
        metrics["decode_ms"] = engine_decode
        metrics["decode_tokens"] = request.get("engine_decode_tokens") or completion_tokens or 1
        metrics["decode_basis"] = "engine"
    else:
        metrics["decode_ms"] = None
        metrics["decode_tokens"] = None
        metrics["decode_basis"] = None

    metrics["prompt_tokens"] = prompt_tokens
    metrics["completion_tokens"] = completion_tokens
    metrics["wall_ms"] = wall_ms
    metrics["engine_prefill_ms"] = engine_prefill
    metrics["engine_decode_ms"] = engine_decode
    metrics["engine_decode_tokens"] = request.get("engine_decode_tokens")
    metrics["prefill_tok_s"] = (
        prompt_tokens / (metrics["ttft_ms"] / 1000.0)
        if prompt_tokens and metrics["ttft_ms"] else None
    )
    metrics["decode_tok_s"] = (
        metrics["decode_tokens"] / (metrics["decode_ms"] / 1000.0)
        if metrics["decode_tokens"] and metrics["decode_ms"] else None
    )
    metrics["decode_ms_per_token"] = (
        metrics["decode_ms"] / metrics["decode_tokens"]
        if metrics["decode_ms"] and metrics["decode_tokens"] else None
    )
    metrics["engine_decode_ms_per_token"] = (
        engine_decode / metrics["engine_decode_tokens"]
        if engine_decode and metrics["engine_decode_tokens"] else None
    )
    # S3: what the serving layer adds beyond the engine's own prefill+decode.
    metrics["overhead_ms"] = (
        wall_ms - (engine_prefill + engine_decode)
        if wall_ms is not None and engine_prefill is not None and engine_decode is not None
        else None
    )
    return metrics


def group_key(stage: str, request: dict) -> tuple:
    """One row per measurement condition: (kind, prompt tokens, generated tokens)."""
    target = request.get("target_tokens")
    max_tokens = request.get("max_tokens")
    kind = "single request" if stage in ("s3", "s4") else "case"
    return (kind, target, max_tokens)


def group_label(key: tuple) -> str:
    kind, target, max_tokens = key
    if kind == "single request":
        return f"{target} tok in / {max_tokens} tok out (x N)"
    if max_tokens == 8:
        return f"{target} tok prompt"
    return f"{target} tok prompt / {max_tokens} tok out"


def median_or_none(values):
    values = [value for value in values if value is not None]
    return median(values) if values else None


def fmt(value, digits=1, suffix=""):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def ratio_label(ratio):
    """Pairing status for one cell: measured ratio, baseline, or no pair."""
    if ratio is None:
        return "(baseline / unpaired)"
    return f"{ratio:.2f}x"


# ------------------------------------------------------------------- report ---
class Report:
    def __init__(self, args):
        self.args = args
        self.raw_dir = Path(args.raw_dir)
        if not self.raw_dir.is_absolute():
            self.raw_dir = Path(args.root) / self.raw_dir
        self.runs, self.problems = load_runs(self.raw_dir)
        self.stages = [stage for stage in ("s1", "s2", "s3", "s4")
                       if stage in self.stage_filter(args.stage)]
        self.blockers: list[str] = []
        self.warnings: list[str] = []
        self.json_summary: dict = {"raw_dir": str(self.raw_dir), "stages": {}}

    @staticmethod
    def stage_filter(value: str) -> set[str]:
        if value == "all":
            return {"s1", "s2", "s3", "s4"}
        return {part.strip() for part in value.split(",") if part.strip()}

    # -- inventory ---------------------------------------------------------
    def inventory(self) -> dict:
        """arm -> stage -> summary of the runs that carry it."""
        inventory: dict = defaultdict(lambda: defaultdict(lambda: {
            "files": [], "segments": set(), "requests": 0, "ok": 0, "failed": 0,
            "generated": 0, "engaged": False, "protocol": set(), "quantization": set(),
            "prompt_protocol": set(), "binary_sha256": set(), "model_sha256": set(),
            "corpus": set(),
        }))
        for run in self.runs:
            arm, stage = run.get("arm"), run.get("stage")
            if stage not in self.stages:
                continue
            entry = inventory[arm][stage]
            entry["files"].append(run["_path"])
            entry["segments"].add(segment_of(run))
            entry["protocol"].add(run.get("protocol"))
            artifact = run.get("artifact") or {}
            entry["quantization"].add(artifact.get("quantization"))
            entry["prompt_protocol"].add(run.get("prompt_protocol") or artifact.get("prompt_protocol"))
            entry["binary_sha256"].add(artifact.get("binary_sha256"))
            entry["model_sha256"].add(artifact.get("model_sha256"))
            entry["corpus"].add((run.get("corpus") or {}).get("sha256"))
            evidence = run.get("evidence") or {}
            entry["engaged"] = entry["engaged"] or bool(evidence.get("engaged"))
            for request in requests_of(run):
                entry["requests"] += 1
                if request.get("status") == "ok":
                    entry["ok"] += 1
                    if (request.get("completion_tokens") or 0) > 0:
                        entry["generated"] += 1
                else:
                    entry["failed"] += 1
        return inventory

    # -- A/A ---------------------------------------------------------------
    def aa_check(self, inventory) -> dict:
        """Same-arm repeatability: per group, spread of the segment medians."""
        metric_by_stage = PRIMARY_METRIC
        results: dict = {}
        for arm, stages in inventory.items():
            for stage, entry in stages.items():
                if stage not in metric_by_stage:
                    continue
                metric = metric_by_stage[stage]
                # (group, segment) -> list of metric values
                buckets: dict = defaultdict(lambda: defaultdict(list))
                for run in self.runs:
                    if run.get("arm") != arm or run.get("stage") != stage:
                        continue
                    segment = segment_of(run)
                    for request in requests_of(run):
                        derived = request_metrics(request)
                        if derived.get(metric) is None:
                            continue
                        buckets[group_key(stage, request)][segment].append(derived[metric])
                spreads = {}
                for group, per_segment in buckets.items():
                    if len(per_segment) < 2:
                        continue
                    medians = [median(values) for values in per_segment.values()]
                    reference = median(medians)
                    if reference:
                        spreads[group] = (max(medians) - min(medians)) / reference
                worst = max(spreads.values()) if spreads else None
                segments = len(entry["segments"])
                if worst is None:
                    status = "UNVERIFIED"
                elif worst <= self.args.aa_tolerance:
                    status = "PASS"
                else:
                    status = "FAIL"
                results[(arm, stage)] = {
                    "status": status, "worst_spread": worst, "segments": segments,
                    "metric": metric, "spreads": spreads,
                }
        return results

    def duplicate_arm_check(self) -> dict:
        """A/A in its strongest form: the same configuration run under <arm>-aa."""
        results = {}
        arms = {run.get("arm") for run in self.runs}
        for arm in sorted(arms):
            base = arm[:-3] if arm.endswith("-aa") else None
            if not base or base not in arms:
                continue
            for stage in self.stages:
                metric = PRIMARY_METRIC.get(stage)
                if not metric:
                    continue
                buckets = {base: defaultdict(list), arm: defaultdict(list)}
                for run in self.runs:
                    name = run.get("arm")
                    if name not in buckets or run.get("stage") != stage:
                        continue
                    for request in requests_of(run):
                        derived = request_metrics(request)
                        if derived.get(metric) is None:
                            continue
                        buckets[name][(group_key(stage, request), segment_of(run))].append(derived[metric])
                deviations = []
                for key, base_values in buckets[base].items():
                    if key not in buckets[arm]:
                        continue
                    base_median = median(base_values)
                    arm_median = median(buckets[arm][key])
                    if base_median:
                        deviations.append(abs(arm_median / base_median - 1.0))
                if deviations:
                    worst = max(deviations)
                    results[(arm, stage)] = {
                        "base": base, "worst_deviation": worst,
                        "status": "PASS" if worst <= self.args.aa_tolerance else "FAIL",
                    }
        return results

    # -- tables ------------------------------------------------------------
    def stage_rows(self, stage: str):
        """(group, arm) -> pooled medians + samples, client-first basis."""
        metric = PRIMARY_METRIC[stage]
        buckets: dict = defaultdict(lambda: {"samples": [], "segments": set(), "bases": set(),
                                             "requests": 0})
        for run in self.runs:
            if run.get("stage") != stage:
                continue
            arm = run.get("arm")
            segment = segment_of(run)
            for request in requests_of(run):
                derived = request_metrics(request)
                key = (group_key(stage, request), arm)
                buckets[key]["requests"] += 1
                buckets[key]["segments"].add(segment)
                buckets[key]["bases"].add(derived.get("ttft_basis") if stage == "s1"
                                         else derived.get("decode_basis"))
                if derived.get(metric) is not None:
                    buckets[key]["samples"].append((segment, derived))
        rows = {}
        for (group, arm), bucket in buckets.items():
            samples = bucket["samples"]
            values = [derived[metric] for _, derived in samples]
            row = {
                "group": group, "arm": arm, "n": bucket["requests"],
                "segments": len(bucket["segments"]),
                "primary": median_or_none(values),
                "primary_n": len(values),
                "bases": sorted(base for base in bucket["bases"] if base),
                "segments_by_id": {},
            }
            if stage == "s1":
                row["rate"] = median_or_none([derived["prefill_tok_s"] for _, derived in samples])
                row["rate_label"] = "prefill tok/s"
                row["engine_primary"] = median_or_none([derived["engine_prefill_ms"] for _, derived in samples])
            elif stage == "s2":
                row["rate"] = median_or_none([derived["decode_tok_s"] for _, derived in samples])
                row["rate_label"] = "decode tok/s"
                row["engine_primary"] = median_or_none(
                    [derived["engine_decode_ms_per_token"] for _, derived in samples])
            else:
                row["ttft"] = median_or_none([derived.get("ttft_ms") for _, derived in samples])
                row["overhead"] = median_or_none([derived.get("overhead_ms") for _, derived in samples])
                row["engine_primary"] = row["overhead"]
            per_segment = defaultdict(list)
            for segment, derived in samples:
                per_segment[segment].append(derived[metric])
            row["segments_by_id"] = {segment: median(vals) for segment, vals in per_segment.items()}
            rows[(group, arm)] = row
        return rows

    def paired_ratios(self, stage: str, rows: dict) -> dict:
        """(group, arm) -> median over segments of baseline/arm median ratios.

        Latency metrics are lower-is-better, so the ratio is baseline/arm: >1
        means the arm is faster than the baseline.
        """
        baseline = self.args.baseline
        ratios = {}
        for (group, arm), row in rows.items():
            base_row = rows.get((group, baseline))
            if base_row is None or arm == baseline:
                continue
            per_segment = []
            for segment, arm_median in row["segments_by_id"].items():
                base_median = base_row["segments_by_id"].get(segment)
                if base_median and arm_median:
                    per_segment.append(base_median / arm_median)
            if per_segment:
                ratios[(group, arm)] = median(per_segment)
        return ratios

    # -- corpus identity ---------------------------------------------------
    def corpus_table(self) -> dict:
        table: dict = defaultdict(dict)
        for run in self.runs:
            stage = run.get("stage")
            if stage not in self.stages:
                continue
            arm = run.get("arm")
            for request in requests_of(run):
                if request.get("prompt_tokens") is None:
                    continue
                key = (stage, group_key(stage, request))
                table[key].setdefault(arm, []).append(request["prompt_tokens"])
        return {key: {arm: median(values) for arm, values in arms.items()}
                for key, arms in table.items()}

    # -- mixed factors -----------------------------------------------------
    def mixed_factors(self, inventory) -> list[str]:
        lines = []
        quantization: dict = defaultdict(set)
        protocols: dict = defaultdict(set)
        prompt_protocols: dict = defaultdict(set)
        for arm, stages in inventory.items():
            for stage, entry in stages.items():
                quantization[arm].update(value for value in entry["quantization"] if value)
                protocols[arm].update(value for value in entry["protocol"] if value)
                prompt_protocols[arm].update(value for value in entry["prompt_protocol"] if value)
        distinct_quant = {value for values in quantization.values() for value in values}
        if len(distinct_quant) > 1:
            lines.append(
                "Quantization differs between arms and is not a controlled variable: "
                + "; ".join(f"`{arm}` = {', '.join(sorted(values)) or 'undeclared'}"
                            for arm, values in sorted(quantization.items()))
                + ". The packed4 affine-g64 tier and the GGUF Q4_K_M tier are both '4-bit' "
                  "but not the same format; a difference measured here cannot be attributed to "
                  "the engine implementation unless the 16-bit arms reproduce its sign "
                  "(pre-registered Decision rule)."
            )
        elif distinct_quant:
            lines.append(
                "Quantization is uniform in this data: "
                + ", ".join(sorted(distinct_quant))
                + ". Cross-tier attribution still requires the 16-bit arm (Decision rule)."
            )
        else:
            lines.append(
                "Quantization is not declared in the raw files: use --quantization when running "
                "the harness, otherwise the tables cannot state the mixed factor they carry."
            )
        if len(distinct_quant) > 1 and any(
                "gguf" in value and "q4" in value.lower() for value in distinct_quant):
            lines.append(
                "Concretely: `apxinf-*` arms run MLX packed4 (affine group 64); `llamacpp` and "
                "`ollama` run LFM3-1.7B-Q4_K_M.gguf (llama.cpp K-quant, 4.5-5 bpw). EP weight "
                "bytes per token therefore differ, so an absolute tok/s gap is expected even for "
                "two perfect engine implementations."
            )
        if any("chat" in value for values in protocols.values() for value in values) and \
                any("completions" in value for values in protocols.values() for value in values):
            lines.append(
                "Prompt protocol is not uniform: arms measured with `openai-completions` sent the "
                "bare completion prompt (no chat template), while `openai-chat` arms go through "
                "the family's lane ChatML prefix. The template tokens are included in their "
                "reported `prompt_tokens`, and the corpus table above shows the offset; S1 TTFT "
                "for those arms includes rendering/evaluating that prefix."
            )
        lines.append(
            "Measurement basis: S1 TTFT and S2 decode are client-observed (first and last SSE "
            "content delta). Engine-reported prefill/decode durations are shown separately, and "
            "they cover different spans per engine (llama.cpp prompt eval; apxinf state reset + "
            "capacity preparation + prefill). S3 subtracts the engine-reported durations from "
            "client wall where they exist; where they do not, S3 overhead is n/a and the wall "
            "column alone must not be read as serving overhead."
        )
        lines.append(
            "Pairing: every delta in these tables is the median over ABBA segments of the "
            "within-segment median ratio against "
            f"`{self.args.baseline}`. No absolute number in this report crosses a window, and no "
            "single-order number is reported."
        )
        return lines

    # -- gate --------------------------------------------------------------
    def gate(self, inventory, aa, stage_rows) -> list[str]:
        lines = []
        lines.append("**Gate (comparability):** A/A <= "
                     f"{self.args.aa_tolerance:.0%} per arm; otherwise the arm's data is void.")
        for (arm, stage), result in sorted(aa.items()):
            spread = result["worst_spread"]
            if result["status"] == "UNVERIFIED":
                lines.append(f"- `{arm}` {stage}: A/A UNVERIFIED "
                             f"({result['segments']} segment(s); need >= 2 repeats)")
            else:
                lines.append(f"- `{arm}` {stage}: A/A {result['status']} "
                             f"(worst group spread {spread:.1%}, {result['segments']} segments)")

        lines.append("")
        lines.append("**Gate (S2 decode):** paired median; if apxinf < 0.8 x baseline the verdict "
                     "is written as 'apxinf decode lags' and a follow-up is opened; the word "
                     "'parity' is not used.")
        s2_rows = stage_rows.get("s2", {})
        s2_paired = self.paired_ratios("s2", s2_rows) if s2_rows else {}
        for apxinf_arm in self.args.apxinf_arms:
            group_ratios = []
            for (group, arm), ratio in sorted(s2_paired.items()):
                if arm != apxinf_arm:
                    continue
                group_ratios.append((group, ratio))
            if not group_ratios:
                lines.append(f"- S2 decode for `{apxinf_arm}` vs `{self.args.baseline}`: "
                             "not measured (no paired S2 decode data for both arms).")
                continue
            ratio = median([value for _, value in group_ratios])
            detail = ", ".join(f"{group_label(group)}={value:.2f}x"
                               for group, value in group_ratios)
            lines.append(f"- S2 decode paired ratios for `{apxinf_arm}` / "
                         f"`{self.args.baseline}`: {detail} (median {ratio:.2f}x).")
            if ratio < 0.8:
                statement = (f"apxinf decode lags {self.args.baseline} "
                             f"({ratio:.2f}x the baseline decode rate, below the 0.8 gate): "
                             "write the lag verdict and open the follow-up.")
            else:
                statement = (f"ratio {ratio:.2f}x is at or above the 0.8 gate: the pre-registered "
                             "lag wording does not apply, and neither does 'parity' -- report the "
                             "paired ratio as measured.")
            lines.append(f"- Verdict wording for the median paired ratio {ratio:.2f}x "
                         f"(`{apxinf_arm}` / `{self.args.baseline}`): {statement}")

        lines.append("")
        lines.append("**Gate (S4 end-to-end):** equal solved set and wall delta > 10% => engine-level "
                     "difference; different solved set => check template/quantization first.")
        if not stage_rows.get("s4"):
            lines.append("- S4: not measured by this harness (the 12-task agent suite is a separate "
                         "tool); no end-to-end claim is made here.")
        else:
            lines.append("- S4: data present; the solved-set comparison is in the S4 table.")

        lines.append("")
        lines.append("**Decision rule:** claim the difference as an engine-implementation difference "
                     "only when the 16-bit arms reproduce its sign; otherwise attribute it to the "
                     "quantization tier.")
        sixteen_bit = [arm for arm in inventory
                       if any(value and ("f16" in value.lower() or "bf16" in value.lower())
                              for stage in inventory[arm].values()
                              for value in stage["quantization"])]
        if sixteen_bit:
            lines.append(f"- 16-bit arms present: {', '.join(sorted(sixteen_bit))}; attribute only "
                         "after their direction agrees.")
        else:
            lines.append("- No 16-bit arm was measured: differences in this report stay attributed "
                         "to the quantization tier, not to the engine implementation.")
        return lines

    # -- main --------------------------------------------------------------
    def run(self):
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print("# LFM3-1.7B engine comparison - paired report")
        print()
        print(f"- raw directory: `{self.raw_dir}`")
        print(f"- runs read: {len(self.runs)}")
        print(f"- stages: {', '.join(self.stages) or '(none)'}")
        print(f"- baseline: `{self.args.baseline}`; apxinf arms: "
              f"{', '.join(f'`{arm}`' for arm in self.args.apxinf_arms)}")
        print(f"- generated: {now}")
        print()
        if self.problems:
            print("## Unreadable inputs")
            for problem in self.problems:
                print(f"- {problem}")
            print()

        inventory = self.inventory()
        if not inventory:
            print("No arm runs found. Run scripts/bench_engines.sh first.")
            return 0

        aa = self.aa_check(inventory)
        duplicates = self.duplicate_arm_check()

        # Inventory table.
        print("## Arms")
        print()
        print("| arm | stage | protocol | quantization | segments | requests | failed | "
              "generated | evidence | A/A |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for arm in sorted(inventory):
            for stage in sorted(inventory[arm]):
                entry = inventory[arm][stage]
                result = aa.get((arm, stage), {"status": "n/a", "worst_spread": None})
                aa_text = result["status"]
                if result.get("worst_spread") is not None:
                    aa_text += f" ({result['worst_spread']:.1%})"
                print(f"| `{arm}` | {stage} | {', '.join(sorted(v for v in entry['protocol'] if v))} "
                      f"| {', '.join(sorted(v for v in entry['quantization'] if v)) or 'undeclared'} "
                      f"| {len(entry['segments'])} | {entry['requests']} | {entry['failed']} "
                      f"| {entry['generated']} | {'yes' if entry['engaged'] else 'NO'} | {aa_text} |")
        print()
        if duplicates:
            print("### A/A replicate arms")
            print()
            for (arm, stage), result in sorted(duplicates.items()):
                print(f"- `{arm}` vs `{result['base']}` {stage}: worst deviation "
                      f"{result['worst_deviation']:.1%} -> {result['status']}")
            print()

        # Corpus identity.
        corpus = self.corpus_table()
        if corpus:
            arms = sorted({arm for values in corpus.values() for arm in values})
            print("## Corpus and token-count comparability")
            print()
            print("Prompt tokens are the engines' own reported counts for byte-identical text. "
                  "Bare-completion arms must agree; a chat-protocol arm carries its template "
                  "offset (mixed factor 2).")
            print()
            print("| stage | case | " + " | ".join(f"`{arm}`" for arm in arms) + " |")
            print("|---|---|" + "---|" * len(arms))
            for (stage, group) in sorted(corpus):
                values = corpus[(stage, group)]
                cells = [fmt(values.get(arm), 0) for arm in arms]
                print(f"| {stage} | {group_label(group)} | " + " | ".join(cells) + " |")
            print()
            bare_arms = [arm for arm in arms
                         if all(value in (None, "bare")
                                for stage in inventory.get(arm, {}).values()
                                for value in stage["prompt_protocol"])]
            for (stage, group), values in corpus.items():
                counts = [values[arm] for arm in bare_arms if values.get(arm) is not None]
                if len(counts) > 1 and max(counts) - min(counts) > 2:
                    self.warnings.append(
                        f"{stage} {group_label(group)}: bare-completion prompt token counts differ "
                        f"across arms ({', '.join(f'{arm}={values[arm]:.0f}' for arm in bare_arms if values.get(arm) is not None)}); "
                        "the corpus is not token-identical across engines.")

        # Stage tables.
        for stage in self.stages:
            rows = self.stage_rows(stage)
            if not rows:
                print(f"## {STAGE_TITLES[stage]}")
                print()
                print("No data.")
                print()
                continue
            ratios = self.paired_ratios(stage, rows)
            groups = sorted({group for group, _ in rows})
            arms = sorted({arm for _, arm in rows})
            print(f"## {STAGE_TITLES[stage]}")
            print()
            if stage == "s1":
                print("Client-observed time to first content delta (the S1 metric), with the "
                      "engine-reported prefill as a cross-check. Ratio is paired against "
                      f"`{self.args.baseline}` per segment (baseline/arm; >1 = this arm faster).")
                print()
                print("| case | arm | TTFT p50 (ms) | prefill tok/s p50 | engine prefill p50 (ms) "
                      "| vs baseline | n | seg |")
                print("|---|---|---|---|---|---|---|---|")
                for group in groups:
                    for arm in arms:
                        row = rows.get((group, arm))
                        if row is None:
                            continue
                        ratio = ratios.get((group, arm))
                        print(f"| {group_label(group)} | `{arm}` | {fmt(row['primary'])} "
                              f"| {fmt(row['rate'])} | {fmt(row.get('engine_primary'))} "
                              f"| {ratio_label(ratio)} "
                              f"| {row['n']} | {row['segments']} |")
            elif stage == "s2":
                print("Client-observed decode window (first to last content delta) per generated "
                      "token. This file already declares that the decode basis is the client; the "
                      "engine-reported ms/token is shown next to it.")
                print()
                print("| case | arm | decode ms/token p50 | decode tok/s p50 | engine ms/token p50 "
                      "| vs baseline | n | seg |")
                print("|---|---|---|---|---|---|---|---|")
                for group in groups:
                    for arm in arms:
                        row = rows.get((group, arm))
                        if row is None:
                            continue
                        ratio = ratios.get((group, arm))
                        print(f"| {group_label(group)} | `{arm}` | {fmt(row['primary'], 2)} "
                              f"| {fmt(row['rate'])} | {fmt(row.get('engine_primary'), 2)} "
                              f"| {ratio_label(ratio)} "
                              f"| {row['n']} | {row['segments']} |")
            elif stage == "s3":
                print("Same request (max_tokens=8) issued --count times per segment. Wall is the "
                      "client total; overhead is wall minus the engine-reported prefill+decode "
                      "(n/a where the engine does not report both).")
                print()
                print("| case | arm | wall p50 (ms) | TTFT p50 (ms) | overhead p50 (ms) "
                      "| vs baseline | n | seg |")
                print("|---|---|---|---|---|---|---|---|")
                for group in groups:
                    for arm in arms:
                        row = rows.get((group, arm))
                        if row is None:
                            continue
                        ratio = ratios.get((group, arm))
                        print(f"| {group_label(group)} | `{arm}` | {fmt(row['primary'])} "
                              f"| {fmt(row.get('ttft'))} | {fmt(row.get('overhead'))} "
                              f"| {ratio_label(ratio)} "
                              f"| {row['n']} | {row['segments']} |")
            else:
                print("| case | arm | wall p50 (ms) | vs baseline | n | seg |")
                print("|---|---|---|---|---|---|")
                for group in groups:
                    for arm in arms:
                        row = rows.get((group, arm))
                        if row is None:
                            continue
                        ratio = ratios.get((group, arm))
                        print(f"| {group_label(group)} | `{arm}` | {fmt(row['primary'])} "
                              f"| {ratio_label(ratio)} "
                              f"| {row['n']} | {row['segments']} |")
            print()
            bases = sorted({base for row in rows.values() for base in row["bases"]})
            if len(bases) > 1:
                self.warnings.append(
                    f"{stage}: the table mixes measurement bases across arms ({', '.join(bases)}); "
                    "a cross-arm comparison in this stage is not like-for-like.")
            self.json_summary["stages"][stage] = {
                "rows": [
                    {"group": group_label(group), "arm": arm, "primary": row["primary"],
                     "primary_metric": PRIMARY_METRIC[stage], "rate": row.get("rate"),
                     "engine_primary": row.get("engine_primary"), "n": row["n"],
                     "segments": row["segments"],
                     "vs_baseline": ratios.get((group, arm))}
                    for (group, arm), row in sorted(rows.items())
                ],
            }

        # Mixed factors are printed with the tables, not in an appendix.
        print("## Mixed factors (quote these with every table above)")
        print()
        for index, line in enumerate(self.mixed_factors(inventory), start=1):
            print(f"{index}. {line}")
        print()

        print("## Pre-registered gate")
        print()
        for line in self.gate(inventory, aa, {
                stage: self.stage_rows(stage) for stage in self.stages
                if stage in ("s2", "s4")}):
            print(line)
        print()

        # Verdict blockers.
        for arm, stages in inventory.items():
            for stage, entry in stages.items():
                if not entry["engaged"] or entry["generated"] == 0:
                    self.blockers.append(f"`{arm}` {stage}: no evidence of generated tokens "
                                         "(no-op arm, not a result)")
                if len(entry["segments"]) < self.args.min_segments and not self.args.allow_sparse:
                    self.blockers.append(
                        f"`{arm}` {stage}: {len(entry['segments'])} segment(s), fewer than the "
                        f"pre-registered {self.args.min_segments}; a verdict cannot be claimed")
                if entry["failed"]:
                    self.warnings.append(f"`{arm}` {stage}: {entry['failed']} request(s) failed")
        for (arm, stage), result in aa.items():
            if result["status"] == "FAIL":
                self.blockers.append(
                    f"`{arm}` {stage}: A/A spread {result['worst_spread']:.1%} exceeds "
                    f"{self.args.aa_tolerance:.0%}; the arm's data is void and must be re-measured")
        for key, result in duplicates.items():
            if result["status"] == "FAIL":
                arm, stage = key
                self.blockers.append(
                    f"`{arm}` {stage}: A/A replicate deviates {result['worst_deviation']:.1%} from "
                    f"`{result['base']}`; the arm's data is void and must be re-measured")

        if self.warnings:
            print("## Warnings")
            print()
            for warning in self.warnings:
                print(f"- {warning}")
            print()
        print("## Status")
        print()
        if self.blockers:
            for blocker in self.blockers:
                print(f"- BLOCKED: {blocker}")
            print()
            print("No verdict may be claimed from this data set until the blockers above clear "
                  "(skills/perf-verdict-gate).")
        else:
            print("- Every arm met the pre-declared comparability gate (A/A, segments, evidence).")
            print("- The measured ratios above are the reportable result; the verdict wording is "
                  "fixed by the pre-registered gate text quoted above.")
        print()

        if self.args.json_out:
            self.json_summary["aa"] = {
                f"{arm}/{stage}": {k: v for k, v in result.items() if k != "spreads"}
                for (arm, stage), result in aa.items()
            }
            self.json_summary["blockers"] = self.blockers
            self.json_summary["warnings"] = self.warnings
            path = Path(self.args.json_out)
            if not path.is_absolute():
                path = Path(self.args.root) / path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.json_summary, indent=2, default=str) + "\n")
            print(f"summary written to {path}")

        no_evidence = any("no evidence" in blocker for blocker in self.blockers)
        aa_failed = any(("A/A spread" in blocker) or ("A/A replicate" in blocker)
                        for blocker in self.blockers)
        sparse = any("fewer than" in blocker for blocker in self.blockers)
        if no_evidence:
            return 3
        if aa_failed:
            return 2
        if sparse:
            return 4
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW,
                        help=f"raw JSON directory (default: {DEFAULT_RAW})")
    parser.add_argument("--stage", default="all", help="s1,s2,s3,s4 or all")
    parser.add_argument("--baseline", default="llamacpp", help="arm every ratio is paired against")
    parser.add_argument("--apxinf-arms", default="apxinf-native",
                        help="comma-separated apxinf arms for the S2 gate wording")
    parser.add_argument("--aa-tolerance", type=float, default=0.05,
                        help="maximum A/A spread (default 5%%, pre-registered)")
    parser.add_argument("--min-segments", type=int, default=3,
                        help="pre-registered minimum ABBA segments per arm (default 3)")
    parser.add_argument("--allow-sparse", action="store_true",
                        help="inspect partial data without failing the segment gate")
    parser.add_argument("--json-out", help="write a machine-readable summary here")
    args = parser.parse_args()
    args.apxinf_arms = [part.strip() for part in args.apxinf_arms.split(",") if part.strip()]
    # The tolerance is fractional (0.05 = 5%). Refuse to reinterpret a number like
    # 5 as 500%: a gate that silently changes magnitude is how a 5% rule becomes an
    # unenforceable one.
    if not 0.0 < args.aa_tolerance <= 1.0:
        parser.error("--aa-tolerance is fractional: 0.05 means 5%")
    return Report(args).run()


if __name__ == "__main__":
    sys.exit(main())
