#!/usr/bin/env python3
"""Compare the general CLI and two compilation scopes of one native entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compare as common

BUILD_ORDER = ["full", "reference", "minimal", "minimal", "reference", "full"]
RUN_ORDER = ["reference", "minimal", "minimal", "reference"]
PROFILE = "qwen35-0.8b-f32-metal-w8-head-mlp-v1"
FORMAT = "apxinf-qwen35-specialized-v1"
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def checked_stdout(sample):
    if sample["returncode"] != 0:
        raise ValueError("A measured process failed")
    common.verify_file(sample["stdout"])
    common.verify_file(sample["stderr"])
    return Path(sample["stdout"]["path"]).read_text()


def validate_result(result, budget, eos):
    tokens = result["generated_token_ids"]
    if not tokens or len(tokens) > budget or result["max_tokens"] != budget:
        raise ValueError("Result token count or budget is invalid")
    reason = "eos" if tokens[-1] == eos else "max_tokens"
    if result["stop_reason"] != reason or (reason == "max_tokens" and len(tokens) != budget):
        raise ValueError("Generation stopped without EOS or its budget")
    timing = result["timing"]
    for key in ("request_ms", "ttft_ms", "decode_ms", "total_request_ms"):
        if not isinstance(timing[key], (int, float)) or not math.isfinite(timing[key]) or timing[key] < 0:
            raise ValueError("Timing must be finite and nonnegative")
    if (not math.isclose(timing["request_ms"], timing["ttft_ms"] + timing["decode_ms"], abs_tol=0.0001)
            or timing["total_request_ms"] < timing["request_ms"]):
        raise ValueError("Timing boundaries are inconsistent")
    if len(tokens) == 1:
        if timing["decode_tps"] is not None:
            raise ValueError("One token has no steady decode measurement")
    elif (timing["decode_ms"] <= 0 or timing["decode_tps"] is None
          or not math.isclose(timing["decode_tps"], (len(tokens) - 1) * 1000 / timing["decode_ms"], rel_tol=1e-9)):
        raise ValueError("Decode throughput differs from token count and duration")


def verify(args, model=False):
    record = read(args.output / "inputs.json")
    for key in ("harness", "helper", "contract", "cases", "asset_reference", "snapshot_manifest"):
        common.verify_file(record[key])
    if common.verify_snapshot(args.snapshot)["snapshot_sha256"] != record["snapshot_sha256"]:
        raise ValueError("Source snapshot changed")
    if model and common.model_assets(Path(record["model"])) != record["assets"]:
        raise ValueError("Model assets changed")
    return record


def prepare(args):
    if args.output.exists():
        raise ValueError("Use a new experiment output directory")
    contract = read(HERE / "contract.json")
    if contract["build_order"] != BUILD_ORDER or contract["runtime_order"] != RUN_ORDER:
        raise ValueError("Driver order differs from preregistration")
    cases = read(HERE / "cases.json")
    if (not cases or len({case["id"] for case in cases}) != len(cases)
            or any(not case["id"] or not case["prompt"] or not 1 <= case["max_tokens"] < 2048 for case in cases)):
        raise ValueError("Cases need unique IDs, nonempty prompts and valid budgets")
    reference_path = args.source / contract["asset_reference"]
    reference = read(reference_path)
    if common.model_assets(Path(reference["model"])) != reference["assets"]:
        raise ValueError("Assets differ from the previous pinned experiment")
    common.snapshot(args.source, args.snapshot)
    # Cargo build validates explicit test targets even for a binary-only build.
    for crate in (args.source / "crates").iterdir():
        if (crate / "Cargo.toml").is_file() and (crate / "tests").is_dir():
            shutil.copytree(crate / "tests", args.snapshot / "crates" / crate.name / "tests")
    manifest = read(args.snapshot / "snapshot.json")
    files = {str(path.relative_to(args.snapshot)): common.digest(path)
             for path in sorted(args.snapshot.rglob("*")) if path.is_file() and path != args.snapshot / "snapshot.json"}
    manifest["source_files"] = files
    manifest["snapshot_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest["scope"] = "Native build inputs and declared crate tests. Optional CUDA vendor assets are excluded."
    common.save(args.snapshot / "snapshot.json", manifest)
    source = common.verify_snapshot(args.snapshot)
    args.output.mkdir(parents=True)
    record = {"schema": "apxinf-specialization-inputs-v2", "model": reference["model"],
              "assets": reference["assets"], "snapshot_sha256": source["snapshot_sha256"],
              "snapshot_manifest": common.identity(args.snapshot / "snapshot.json"),
              "harness": common.identity(Path(__file__)),
              "helper": common.identity(Path(common.__file__)),
              "contract": common.identity(HERE / "contract.json"),
              "cases": common.identity(HERE / "cases.json"),
              "asset_reference": common.identity(reference_path)}
    tokenizer_config = read(Path(reference["model"]) / "tokenizer_config.json")
    tokenizer = read(Path(reference["model"]) / "tokenizer.json")
    eos_ids = [token["id"] for token in tokenizer["added_tokens"]
               if token["content"] == tokenizer_config["eos_token"]]
    if eos_ids != [248046]:
        raise ValueError("Pinned chat tokenizer EOS differs from the existing CLI")
    record["eos_token_id"] = eos_ids[0]
    common.save(args.output / "inputs.json", record)
    print(json.dumps(record), flush=True)


def features(variant):
    if variant == "full":
        return ["--features", "accelerate,metal-w8"]
    return ["--features", "general-reference"] if variant == "reference" else []


def scope(target, variant, args, identifier):
    minimal = variant == "minimal"
    evidence = {}
    texts = {}
    for crate in ("model", "metal", "loader", "tokenizer"):
        paths = list((target / "release/deps").glob(f"apxinf_{crate}-*.d"))
        if len(paths) != 1:
            raise ValueError(f"Expected one dependency record for {crate}")
        texts[crate] = paths[0].read_text()
        saved = args.output / f"{identifier}-{crate}.d"
        saved.write_text(texts[crate])
        evidence[crate + "_dependencies"] = common.identity(saved)
    modules = {name: f"src/{name}/" in texts["model"]
               for name in ("llama", "qwen35", "qwen3vl", "pi05")}
    if modules != {name: not minimal or name == "qwen35" for name in modules}:
        raise ValueError(f"Unexpected model families: {modules}")
    registry = {name: ("src/vla/mod.rs" if name == "vla" else f"src/{name}.rs") in texts["model"]
                for name in ("auto", "builtin", "registry", "vla", "debug")}
    if any(present == minimal for present in registry.values()):
        raise ValueError(f"Unexpected registry closure: {registry}")
    native = {suffix: sorted(str(p.relative_to(target)) for p in
                             target.glob(f"release/build/apxinf-metal-*/out/*{suffix}"))
              for suffix in (".a", ".o", ".inc")}
    expected_counts = [2, 2, 2] if minimal else [10, 10, 7]
    if [len(native[suffix]) for suffix in (".a", ".o", ".inc")] != expected_counts:
        raise ValueError(f"Unexpected native compilation scope: {native}")
    head = next(p for p in native[".a"] if p.endswith("libapxinf_metal_w8_bridge.a"))
    symbols = common.command(["nm", "-g", str(target / head)], cwd=args.snapshot)["stdout"]
    symbol_path = args.output / f"{identifier}-head-symbols.txt"
    symbol_path.write_text(symbols)
    if ("apxinf_metal_w8_matvec_create" in symbols) == minimal:
        raise ValueError("Unexpected MatVec native symbols")
    gguf = "src/gguf.rs" in texts["loader"]
    if gguf == minimal:
        raise ValueError("Unexpected GGUF compilation scope")
    fingerprints = list(target.glob("release/.fingerprint/tokenizers-*/lib-tokenizers.json"))
    if len(fingerprints) != 1:
        raise ValueError("Expected one tokenizer feature record")
    tokenizer_features = json.loads(read(fingerprints[0])["features"])
    if ("progressbar" in tokenizer_features) == minimal or ("esaxx_fast" in tokenizer_features) == minimal:
        raise ValueError("Unexpected tokenizer training features")
    if "onig" not in tokenizer_features:
        raise ValueError("Required tokenizer regular expression backend is absent")
    model_records = list(target.glob("release/.fingerprint/apxinf-model-*/lib-apxinf_model.json"))
    if len(model_records) != 1:
        raise ValueError("Expected one model feature record")
    model_features = json.loads(read(model_records[0])["features"])
    if ("metal-experiments" in model_features) == minimal or ("model-registry" in model_features) == minimal:
        raise ValueError("Unexpected model feature selection")
    experimental_modules = [name for name in ("gdn", "linear_layer", "tail_mlp_head_v1",
                            "gdn_core_fused_profile_v1", "gdn_recurrent_profile_v1", "full_attention_decode_v1")
                            if f"src/{name}.rs" in texts["metal"]]
    if len(experimental_modules) != (0 if minimal else 6):
        raise ValueError("Unexpected experimental Metal Rust modules")
    package = "apxinf" if variant == "full" else "apxinf-qwen35"
    tree = common.command(["cargo", "tree", "--locked", "--offline", "-p", package,
                           *features(variant), "-e", "normal,build", "--prefix", "none"],
                          cwd=args.snapshot)["stdout"]
    tree_path = args.output / f"{identifier}-dependency-tree.txt"
    tree_path.write_text(tree)
    excluded = {name: any(line.startswith(name + " v") for line in tree.splitlines())
                for name in ("clap", "byteorder", "indicatif")}
    if minimal and any(excluded.values()):
        raise ValueError(f"Unexpected minimal dependencies: {excluded}")
    evidence.update(model_families=modules, model_features=model_features, registry_modules=registry, native=native,
                    tokenizer_features=tokenizer_features, gguf=gguf,
                    experimental_metal_modules=experimental_modules, dependency_presence=excluded,
                    dependency_tree=common.identity(tree_path), head_symbols=common.identity(symbol_path))
    return evidence


def build(args):
    inputs = verify(args)
    report = {"status": "running", "inputs": common.identity(args.output / "inputs.json"),
              "builds": [], "environment_before": common.system_state()}
    output = args.output / "builds.json"
    if output.exists():
        raise ValueError("Refuse to replace build evidence")
    env = dict(os.environ)
    for name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"):
        env.pop(name, None)
    report["compiler_environment"] = {k: v for k, v in env.items() if k in
        ("CC", "CXX", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "CARGO_HOME") or k.startswith("CARGO_PROFILE_")}
    report["rustc"] = common.command(["rustc", "-vV"], cwd=args.snapshot, env=env)["stdout"]
    report["cargo"] = common.command(["cargo", "--version"], cwd=args.snapshot, env=env)["stdout"]
    for index, variant in enumerate(BUILD_ORDER, 1):
        identifier = f"{index}-{variant}"
        target = args.output / "build-targets" / identifier
        if target.exists():
            raise ValueError("Clean target already exists")
        package = "apxinf" if variant == "full" else "apxinf-qwen35"
        binary_name = "apxinf" if variant == "full" else "apxinf-qwen35-08b"
        argv = ["cargo", "build", "--offline", "--locked", "--release", "--jobs", "4",
                "--target-dir", str(target), "-p", package, *features(variant), "--bin", binary_name]
        row = {"id": identifier, "variant": variant, "target": str(target)}
        report["builds"].append(row)
        common.save(output, report)
        print(f"Clean build {identifier}", flush=True)
        row["clean"] = common.command(argv, cwd=args.snapshot, env=env, log=args.output / f"{identifier}-clean.log")
        artifact = args.output / "artifacts" / identifier / binary_name
        artifact.parent.mkdir(parents=True)
        shutil.copy2(target / "release" / binary_name, artifact)
        row["binary"] = common.identity(artifact)
        row["runtime_libraries"] = common.command(["otool", "-L", str(artifact)], cwd=args.snapshot)["stdout"]
        row["scope"] = scope(target, variant, args, identifier)
        row["noop"] = common.command(argv, cwd=args.snapshot, env=env, log=args.output / f"{identifier}-noop.log")
        source = args.snapshot / ("src/main.rs" if variant == "full" else "crates/apxinf-qwen35/src/main.rs")
        original = source.read_bytes()
        try:
            source.write_bytes(original + b"\n// Incremental rebuild measurement.\n")
            row["incremental"] = common.command(argv, cwd=args.snapshot, env=env,
                                                log=args.output / f"{identifier}-incremental.log")
        finally:
            source.write_bytes(original)
        common.save(output, report)
        print(json.dumps({"id": identifier, "seconds": row["clean"]["wall_seconds"],
                          "bytes": row["binary"]["bytes"]}), flush=True)
    if verify(args) != inputs:
        raise ValueError("Build inputs changed")
    report.update(status="complete", environment_after=common.system_state())
    common.save(output, report)


def process(args, argv, label, input_text=None):
    before = common.system_state()
    started = time.perf_counter()
    result = subprocess.run(argv, input=input_text, text=True, capture_output=True, timeout=1200, cwd=args.output)
    wall = time.perf_counter() - started
    after = common.system_state()
    stdout = args.output / f"{label}.stdout"
    stderr = args.output / f"{label}.stderr"
    stdout.write_text(result.stdout)
    stderr.write_text(result.stderr)
    record = {"command": argv, "wall_seconds": wall, "returncode": result.returncode,
              "stdout": common.identity(stdout), "stderr": common.identity(stderr),
              "environment_before": before, "environment_after": after}
    common.save(args.output / f"{label}.process.json", record)
    if result.returncode:
        raise RuntimeError(f"Process failed; see {stderr}")
    return record, result.stdout


def runtime(args):
    inputs = verify(args, model=True)
    builds = read(args.output / "builds.json")
    if builds["status"] != "complete":
        raise ValueError("Build phase is incomplete")
    if builds["inputs"] != common.identity(args.output / "inputs.json"):
        raise ValueError("Build report belongs to another experiment")
    variants = {variant: next(row["binary"] for row in builds["builds"] if row["variant"] == variant)
                for variant in ("full", "reference", "minimal")}
    for binary in variants.values():
        common.verify_file(binary)
    output = args.output / "runtime.json"
    if output.exists():
        raise ValueError("Refuse to replace runtime evidence")
    cases = read(HERE / "cases.json")
    report = {"status": "running", "inputs": common.identity(args.output / "inputs.json"),
              "build_report": common.identity(args.output / "builds.json"), "cli": [], "resident": []}
    common.save(output, report)
    for case in cases:
        print(f"General CLI parity {case['id']}", flush=True)
        argv = [variants["full"]["path"], "generate", "--model", inputs["model"], "--prompt", case["prompt"],
                "--max-tokens", str(case["max_tokens"]), "--max-context", "2048", "--provider", "native",
                "--device", "cpu", "--dtype", "fp32", "--metal-w8-lm-head", "--metal-w8-mlp-block", "--json"]
        run, stdout = process(args, argv, "cli-" + case["id"])
        report["cli"].append({"case": case["id"], **run, "receipt": json.loads(stdout)})
        common.save(output, report)
    requests = [{"round": round_index, "case": case["id"], "warmup": round_index < 2,
                 "request": {"prompt": case["prompt"], "max_tokens": case["max_tokens"]}}
                for round_index in range(5) for case in cases]
    request_text = "".join(json.dumps(item["request"], ensure_ascii=False) + "\n" for item in requests)
    for index, variant in enumerate(RUN_ORDER, 1):
        print(f"Resident process {index} {variant}", flush=True)
        run, stdout = process(args, [variants[variant]["path"], "--model", inputs["model"], "--jsonl"],
                              f"resident-{index}-{variant}", request_text)
        records = [json.loads(line) for line in stdout.splitlines()]
        if len(records) != len(requests) + 1:
            raise ValueError("Resident process returned incomplete records")
        report["resident"].append({"variant": variant, **run, "ready": records[0],
                                   "runs": [{**setting, "result": value} for setting, value in zip(requests, records[1:])]})
        common.save(output, report)
    case = cases[0]
    argv = [variants["minimal"]["path"], "--model", inputs["model"], "--prompt", case["prompt"],
            "--max-tokens", str(case["max_tokens"])]
    run, stdout = process(args, argv + ["--json"], "minimal-single-json")
    report["single_json"] = {**run, "case": case["id"], "result": json.loads(stdout)}
    run, stdout = process(args, argv, "minimal-plain-stream")
    report["plain"] = {**run, "case": case["id"], "text": stdout}
    for binary in variants.values():
        common.verify_file(binary)
    if verify(args, model=True) != inputs:
        raise ValueError("Runtime inputs changed")
    common.verify_file(report["build_report"])
    report["status"] = "complete"
    common.save(output, report)


def summarize(args):
    inputs = verify(args)
    builds = read(args.output / "builds.json")
    runtime_report = read(args.output / "runtime.json")
    summary = {"accepted": False, "formal_performance_accepted": False,
               "acceptance_scope": "Input identity, source exclusions, full sample grid and corpus equivalence only",
               "build": {}, "runtime": {}}
    output = args.output / "summary.json"
    common.save(output, summary)
    if builds["status"] != "complete" or runtime_report["status"] != "complete":
        raise ValueError("Incomplete phase")
    common.verify_file(runtime_report["build_report"])
    common.verify_file(builds["inputs"])
    common.verify_file(runtime_report["inputs"])
    if (builds["inputs"] != runtime_report["inputs"]
            or builds["inputs"] != common.identity(args.output / "inputs.json")
            or runtime_report["build_report"] != common.identity(args.output / "builds.json")):
        raise ValueError("Phase reports belong to different experiments")
    if [row["variant"] for row in builds["builds"]] != BUILD_ORDER:
        raise ValueError("Incomplete build grid")
    for row in builds["builds"]:
        common.verify_file(row["binary"])
        for key, value in row["scope"].items():
            if isinstance(value, dict) and "sha256" in value:
                common.verify_file(value)
    cases = read(HERE / "cases.json")
    case_map = {case["id"]: {"prompt": case["prompt"], "max_tokens": case["max_tokens"]} for case in cases}
    expected_grid = [(round_index, case["id"], round_index < 2) for round_index in range(5) for case in cases]
    references = {}
    measured = {variant: [] for variant in RUN_ORDER}
    if [row["variant"] for row in runtime_report["resident"]] != RUN_ORDER:
        raise ValueError("Incomplete resident process grid")
    for sample in runtime_report["resident"]:
        raw = [json.loads(line) for line in checked_stdout(sample).splitlines()]
        if raw != [sample["ready"]] + [run["result"] for run in sample["runs"]]:
            raise ValueError("Parsed resident records differ from raw output")
        ready = sample["ready"]
        expected_assets = {k: v for k, v in inputs["assets"].items() if k != "model.safetensors.index.json"}
        if (ready["format"] != FORMAT or ready["kind"] != "ready" or ready["profile"] != PROFILE
                or ready["max_context"] != 2048 or ready["asset_identity"]["assets"] != expected_assets
                or ready["model_revision"] != read(HERE / "contract.json")["revision"]):
            raise ValueError("Unexpected ready identity or settings")
        if [(row["round"], row["case"], row["warmup"]) for row in sample["runs"]] != expected_grid:
            raise ValueError("Incomplete request grid")
        prefills = decodes = 0
        for run in sample["runs"]:
            result = run["result"]
            if run["request"] != case_map[run["case"]]:
                raise ValueError("Request differs from the fixed corpus")
            validate_result(result, run["request"]["max_tokens"], inputs["eos_token_id"])
            tokens = result["generated_token_ids"]
            if (result["format"] != FORMAT or result["kind"] != "result" or result["profile"] != PROFILE
                    or result["max_tokens"] != run["request"]["max_tokens"] or not tokens
                    or len(tokens) > result["max_tokens"] or not result["prompt_token_ids"]):
                raise ValueError("Invalid result settings or token counts")
            prefills += 1
            decodes += len(tokens) - 1
            common.check_path_calls(result["generation_path_receipt"], prefills, decodes)
            value = {key: result[key] for key in ("prompt_token_ids", "generated_token_ids", "text", "stop_reason")}
            value["path"] = common.path_signature(result["generation_path_receipt"])
            if references.setdefault(run["case"], value) != value:
                raise ValueError("Prompt, output, or execution path changed")
            if not run["warmup"]:
                measured[sample["variant"]].append(run)
    if [row["case"] for row in runtime_report["cli"]] != [case["id"] for case in cases]:
        raise ValueError("Incomplete general CLI parity grid")
    for sample in runtime_report["cli"]:
        if json.loads(checked_stdout(sample)) != sample["receipt"]:
            raise ValueError("Parsed general CLI record differs from raw output")
        reference = references[sample["case"]]
        receipt = sample["receipt"]
        if (receipt["format"] != "apxinf-generation-v1" or receipt["device"] != "cpu"
                or receipt["dtype"] != "fp32" or receipt["build"]["matmul_feature"] != "accelerate"
                or receipt["prompt_token_count"] != len(reference["prompt_token_ids"])
                or receipt["generated_token_ids"] != reference["generated_token_ids"]
                or common.path_signature(receipt["generation_path"]) != reference["path"]):
            raise ValueError("General CLI does not match dedicated results")
        common.check_path_calls(receipt["generation_path"], 1, len(reference["generated_token_ids"]) - 1)
    single = runtime_report["single_json"]
    if (single["case"] != cases[0]["id"] or runtime_report["plain"]["case"] != single["case"]
            or json.loads(checked_stdout(single)) != single["result"]
            or checked_stdout(runtime_report["plain"]) != runtime_report["plain"]["text"]):
        raise ValueError("Single-request output identity differs")
    validate_result(single["result"], cases[0]["max_tokens"], inputs["eos_token_id"])
    reference = references[single["case"]]
    for key in ("prompt_token_ids", "generated_token_ids", "text", "stop_reason"):
        if single["result"][key] != reference[key]:
            raise ValueError("Single JSON result differs")
    if runtime_report["plain"]["text"] != reference["text"] + "\n":
        raise ValueError("Streaming plain output differs from complete decoding")
    for variant in ("full", "reference", "minimal"):
        rows = [row for row in builds["builds"] if row["variant"] == variant]
        summary["build"][variant] = {name + "_seconds": common.stats([row[name]["wall_seconds"] for row in rows])
                                      for name in ("clean", "noop", "incremental")}
        summary["build"][variant]["binary_bytes"] = common.stats([row["binary"]["bytes"] for row in rows])
    for variant, runs in measured.items():
        samples = [sample for sample in runtime_report["resident"] if sample["variant"] == variant]
        summary["runtime"][variant] = {
            "startup": {key: common.stats([sample["ready"]["startup"][key] for sample in samples])
                        for key in ("asset_check_ms", "tokenizer_load_ms", "model_load_ms")},
            "cases": {case["id"]: {key: nullable_stats([run["result"]["timing"][key] for run in runs
                                                        if run["case"] == case["id"]])
                                     for key in ("request_ms", "ttft_ms", "decode_tps", "total_request_ms")}
                      for case in cases}}
    summary.update(accepted=True, resident_requests=sum(len(s["runs"]) for s in runtime_report["resident"]),
                   measured_requests=sum(len(rows) for rows in measured.values()), references=references)
    common.save(output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "references"}, ensure_ascii=False), flush=True)


def nullable_stats(values):
    available = [value for value in values if value is not None]
    return {**(common.stats(available) if available else {"count": 0, "median": None, "min": None, "max": None}),
            "unavailable": len(values) - len(available)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "build", "runtime", "summarize"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("source", "snapshot", "output"):
        setattr(args, name, getattr(args, name).resolve())
    globals()[args.phase](args)


if __name__ == "__main__":
    main()
