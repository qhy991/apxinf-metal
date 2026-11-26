#!/usr/bin/env python3
"""Report S5 token agreement between the two quantization tiers of each arm.

The metric is the longest common token prefix of the two tiers' greedy output for the
same prompt, over the longer of the two lengths. It is deliberately not bit-exactness
across quantizations: a 4-bit and a 16-bit tier are expected to agree for a while and
then diverge, and the useful quantity is how long they stay together and how often
they never diverge at all.
"""

import argparse
import glob
import json
import pathlib
import statistics
from collections import defaultdict


def common_prefix(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="raw directory holding s5-<arm>-<tier>-*.json")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    runs = defaultdict(dict)
    for path in sorted(glob.glob(str(pathlib.Path(args.dir) / "*.json"))):
        document = json.loads(pathlib.Path(path).read_text())
        runs[document["arm"]][document["tier"]] = document

    print("# S5 quality sanity - 4-bit tier against 16-bit tier, per arm")
    print()
    print(f"- raw directory: `{args.dir}`")
    print(f"- arms: {', '.join(sorted(runs))}")
    print()

    if not runs:
        print("No raw arm file on disk, so nothing was measured. A window that reaches")
        print("this point produced no data and is not a result.")
        print()
        if args.json_out:
            pathlib.Path(args.json_out).write_text("{}\n")
        return 4

    summary = {}
    for arm in sorted(runs):
        tiers = runs[arm]
        if len(tiers) < 2:
            print(f"## `{arm}`")
            print()
            print(f"not comparable: only one tier on disk ({', '.join(tiers) or 'none'}).")
            print()
            summary[arm] = {"comparable": False, "tiers": sorted(tiers)}
            continue

        names = sorted(tiers)
        left, right = tiers[names[0]], tiers[names[1]]
        left_by_id = {record["id"]: record for record in left["records"]}
        right_by_id = {record["id"]: record for record in right["records"]}
        shared = sorted(set(left_by_id) & set(right_by_id))

        # Agreement between two tiers is only meaningful if both tiers were asked the
        # same question. The digest is over the prompt text, so a mismatch means the
        # table below would be comparing different inputs.
        mismatched = [case for case in shared
                      if left_by_id[case]["prompt_sha256"] != right_by_id[case]["prompt_sha256"]]
        if mismatched:
            raise SystemExit(f"{arm}: prompts differ between tiers on {mismatched}")

        sample = left_by_id[shared[0]] if shared else {}
        if "prompt_chars" in sample:
            size_key, size_label = "prompt_chars", "prompt chars"
        elif "target_tokens" in sample:
            size_key, size_label = "target_tokens", "target tok"
        else:
            raise SystemExit(f"{arm}: records carry no prompt size field")

        print(f"## `{arm}` - `{names[0]}` against `{names[1]}`")
        print()
        print(f"Same {len(shared)} prompts, greedy, max_tokens {left['max_tokens']}, "
              f"bare completion on both tiers.")
        print()
        print(f"| prompt | {size_label} | `{names[0]}` tok | `{names[1]}` tok | common prefix | agreement |")
        print("|---|---|---|---|---|---|")

        agreements, identical = [], 0
        for case in shared:
            a, b = left_by_id[case], right_by_id[case]
            prefix = common_prefix(a["token_ids"], b["token_ids"])
            span = max(len(a["token_ids"]), len(b["token_ids"])) or 1
            agreements.append(prefix / span)
            identical += int(a["token_ids"] == b["token_ids"])
            print(f"| {case} | {a[size_key]} | {len(a['token_ids'])} "
                  f"| {len(b['token_ids'])} | {prefix} | {prefix / span:.3f} |")

        print()
        print(f"Mean agreement **{statistics.mean(agreements):.3f}**, median "
              f"**{statistics.median(agreements):.3f}**, identical output on "
              f"**{identical}/{len(shared)}** prompts.")
        print()
        summary[arm] = {"comparable": True, "tiers": names, "prompts": len(shared),
                        "mean_agreement": round(statistics.mean(agreements), 4),
                        "median_agreement": round(statistics.median(agreements), 4),
                        "identical": identical}

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(summary, indent=2))
        print(f"summary written to {args.json_out}")

    incomplete = [arm for arm, record in summary.items() if not record["comparable"]]
    return 4 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
