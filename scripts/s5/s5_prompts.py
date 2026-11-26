#!/usr/bin/env python3
"""Freeze an S5 prompt set.

Two sets, both written before any measurement and both frozen with a sha256.

`filler` is 16 prompts from the campaign's own deterministic generator, extracted
verbatim from `apxinf-metal scripts/bench_engines.sh`, at a seed and case ids distinct
from the S1-S3 corpus. The text is content-neutral word salad of varying length.
That neutrality is also its weakness: a meaningless prompt gives the model a flat
next-token distribution, where a small logit change flips the argmax, so filler is
close to a worst case for agreement between two quantization tiers.

`natural` is the 12 task problems from the ANEWeave suite `AG2-suite-v1.json`, whose
sha256 is checked against the value recorded beside that suite. These are well-posed
prompts, so the two tiers have sharper distributions to agree on.

Reporting both is the point: one number from filler alone would be a property of the
prompt set as much as of the quantization.
"""

import argparse
import hashlib
import json
import pathlib

import generator

SEED = 20261126
LENGTHS = [24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 640]
AG2_SHA256 = "660ad1c07c4f111ed1879c12eec9260d45547068951718d2c889a11f1bb75722"


def build_filler():
    generator.CORPUS_SEED = SEED
    prompts = []
    for index, target in enumerate(LENGTHS):
        case = {"id": f"s5-{index:03d}", "target_tokens": target}
        prompts.append({"id": case["id"], "target_tokens": target,
                        "prompt": generator.generate_prompt(case)})
    return {"suite_version": "lfm3-s5-prompts-v1",
            "seed": SEED,
            "generator": "extracted verbatim from apxinf-metal scripts/bench_engines.sh",
            "chars_per_token": 3.85,
            "prompts": prompts}


def build_natural(suite_path):
    blob = pathlib.Path(suite_path).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != AG2_SHA256:
        raise SystemExit(f"{suite_path} has sha256 {digest}, expected {AG2_SHA256}")
    suite = json.loads(blob)
    prompts = [{"id": task["id"], "benchmark": task["benchmark"], "kind": task["kind"],
                "prompt": task["problem"]}
               for task in suite["tasks"]]
    return {"suite_version": "lfm3-s5-natural-prompts-v1",
            "source": pathlib.Path(suite_path).name,
            "source_sha256": digest,
            "prompts": prompts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", default="filler", choices=["filler", "natural"])
    parser.add_argument("--suite", help="the AG2 suite, required for --set natural")
    parser.add_argument("--check", action="store_true",
                        help="compare the file on disk against the generator instead of writing")
    args = parser.parse_args()

    if args.set == "natural":
        if not args.suite:
            raise SystemExit("--set natural needs --suite")
        document = build_natural(args.suite)
    else:
        document = build_filler()

    blob = json.dumps(document, indent=2) + "\n"
    digest = hashlib.sha256(blob.encode()).hexdigest()
    path = pathlib.Path(args.out)

    if args.check:
        on_disk = path.read_text()
        print(f"on-disk sha256 {hashlib.sha256(on_disk.encode()).hexdigest()}")
        print(f"expected  sha256 {digest}")
        return 0 if on_disk == blob else 1

    if path.exists() and path.read_text() != blob:
        raise SystemExit(f"{path} exists and does not match the generator; refusing to overwrite")
    path.write_text(blob)
    print(f"wrote {path}")
    print(f"set {args.set} sha256 {digest} prompts {len(document['prompts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
