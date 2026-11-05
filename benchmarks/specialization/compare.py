#!/usr/bin/env python3
"""Measure model source pruning with isolated builds and fixed native calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time


ORDER = ("baseline", "candidate", "candidate", "baseline")
FEATURES = {"baseline": "accelerate,metal-w8",
            "candidate": "model-qwen35,accelerate,metal-w8"}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(path):
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def verify_file(record):
    if identity(Path(record["path"])) != record:
        raise ValueError(f"File identity changed: {record['path']}")


def verify_snapshot(path):
    record = json.loads((path / "snapshot.json").read_text())
    actual = {relative: digest(path / relative) for relative in record["source_files"]}
    checksum = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    if actual != record["source_files"] or checksum != record["snapshot_sha256"]:
        raise ValueError("Snapshot inputs changed")
    return record


def model_assets(path):
    return {str(item.relative_to(path)): {"bytes": item.stat().st_size, "sha256": digest(item)}
            for item in sorted(path.rglob("*"))
            if item.is_file() and item.suffix in (".json", ".safetensors", ".jinja")}


def prepare(args):
    output = args.output / "inputs.json"
    if output.exists():
        raise ValueError("Use a new output directory for preparation")
    source = verify_snapshot(args.snapshot)
    contract = json.loads(args.contract.read_text())
    workload = contract["workload"]
    if (contract["model"] != "Qwen/Qwen3.5-0.8B" or workload["max_context"] != 2048
            or workload["batch"] != 1 or workload["concurrency"] != 1
            or workload["stop_on_eos"] is not True
            or contract["build_measurement"]["order"] != list(ORDER)
            or contract["generation_measurement"]["order"] != list(ORDER)):
        raise ValueError("The contract does not match this measurement driver")
    if digest(args.cases) != digest(Path(source["source"]) / workload["cases"]):
        raise ValueError("Cases differ from the contract's declared corpus")
    cases = json.loads(args.cases.read_text())
    if (not cases or any(not isinstance(case.get("id"), str) or not case["id"]
                         or not isinstance(case.get("prompt"), str) or not case["prompt"]
                         for case in cases)
            or len({case["id"] for case in cases}) != len(cases)):
        raise ValueError("Cases require unique nonempty IDs and prompts")
    assets = model_assets(args.model)
    if assets.get("config.json", {}).get("sha256") != contract["expected_config_sha256"]:
        raise ValueError("The model config does not match the fixed 0.8B target")
    if "tokenizer.json" not in assets or not any(name.endswith(".safetensors") for name in assets):
        raise ValueError("Model weights or tokenizer are missing")
    evidence = Path(source["source"]) / contract["revision_evidence"]
    provenance = json.loads(evidence.read_text())["model"]
    if provenance["repo_id"] != contract["model"] or provenance["revision"] != contract["revision"]:
        raise ValueError("Local revision evidence does not match the contract")
    record = {"schema": "apxinf-specialization-inputs-v1", "model": str(args.model),
              "assets": assets, "snapshot_sha256": source["snapshot_sha256"],
              "snapshot_manifest": identity(args.snapshot / "snapshot.json"),
              "contract": identity(args.contract), "cases": identity(args.cases),
              "harness": identity(Path(__file__)), "revision_evidence": identity(evidence)}
    save(output, record)
    print(json.dumps({"prepared": str(output), "sha256": digest(output)}), flush=True)


def verify_inputs(args, *, model=False):
    record = json.loads((args.output / "inputs.json").read_text())
    for name in ("snapshot_manifest", "contract", "cases", "harness", "revision_evidence"):
        verify_file(record[name])
    if (Path(record["snapshot_manifest"]["path"]).parent != args.snapshot
            or verify_snapshot(args.snapshot)["snapshot_sha256"] != record["snapshot_sha256"]
            or Path(record["harness"]["path"]) != Path(__file__).resolve()):
        raise ValueError("Prepared source or harness identity differs")
    if model and (str(args.model) != record["model"]
                  or str(args.cases) != record["cases"]["path"]
                  or model_assets(args.model) != record["assets"]):
        raise ValueError("Model assets or cases changed after preparation")
    return record


def command(argv, *, cwd, env=None, timeout=1200, log=None):
    start = time.perf_counter()
    completed = subprocess.run(argv, cwd=cwd, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=timeout)
    record = {"command": [str(v) for v in argv], "returncode": completed.returncode,
              "wall_seconds": time.perf_counter() - start}
    if log is not None:
        log.write_text(completed.stdout + "\n" + completed.stderr)
        record["log"] = str(log)
    else:
        record.update(stdout=completed.stdout, stderr=completed.stderr)
    if completed.returncode:
        raise RuntimeError(json.dumps(record, ensure_ascii=False))
    return record


def system_state():
    result = {}
    for name, argv in (("swap", ["sysctl", "vm.swapusage"]),
                       ("vm", ["vm_stat"]), ("thermal", ["pmset", "-g", "therm"])):
        run = subprocess.run(argv, text=True, capture_output=True)
        result[name] = {"returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    return result


def snapshot(source, destination):
    if destination.exists():
        raise ValueError("Use a new snapshot directory")
    destination.mkdir(parents=True)
    for name in ("Cargo.toml", "Cargo.lock", "AGENTS.md", "LICENSE", "NOTICE"):
        origin = source / name
        if origin.is_file():
            shutil.copy2(origin, destination / name)
    shutil.copytree(source / "src", destination / "src")
    for crate in sorted((source / "crates").iterdir()):
        if not (crate / "Cargo.toml").is_file():
            continue
        target = destination / "crates" / crate.name
        target.mkdir(parents=True)
        for name in ("Cargo.toml", "build.rs", "README.md"):
            if (crate / name).is_file():
                shutil.copy2(crate / name, target / name)
        for name in ("src", "examples"):
            if (crate / name).is_dir():
                shutil.copytree(crate / name, target / name)
    inputs = {str(path.relative_to(destination)): digest(path)
              for path in sorted(destination.rglob("*")) if path.is_file()}
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    record = {"source": str(source), "snapshot": str(destination), "source_files": inputs,
              "snapshot_sha256": identity,
              "git_head": command(["git", "rev-parse", "HEAD"], cwd=source)["stdout"].strip(),
              "scope": "Native build inputs only. Optional CUDA vendor assets and tests are excluded from both snapshots."}
    save(destination / "snapshot.json", record)
    print(json.dumps({"snapshot": str(destination), "files": len(inputs), "sha256": identity}), flush=True)


def build_argv(snapshot_path, target, variant, *, example=False):
    argv = ["cargo", "build", "--locked", "--offline", "--release", "--jobs", "4",
            "--target-dir", str(target), "-p", "apxinf-model" if example else "apxinf"]
    if variant == "candidate":
        argv.append("--no-default-features")
    argv += ["--features", FEATURES[variant]]
    argv += ["--example", "specialization_bench"] if example else ["--bin", "apxinf"]
    return argv


def build(args):
    prepared = verify_inputs(args)
    source_record = verify_snapshot(args.snapshot)
    report = {"schema": "apxinf-specialization-build-v1", "status": "running",
              "snapshot": str(args.snapshot), "snapshot_sha256": source_record["snapshot_sha256"],
              "inputs": identity(args.output / "inputs.json"),
              "order": ORDER, "builds": [],
              "environment_before": system_state(), "platform": platform.platform()}
    env = dict(os.environ)
    for name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"):
        env.pop(name, None)
    report["toolchain"] = command(["rustc", "-vV"], cwd=args.snapshot, env=env)["stdout"]
    report["cargo"] = command(["cargo", "--version"], cwd=args.snapshot, env=env)["stdout"]
    report["compiler_environment"] = {k: v for k, v in env.items()
                                      if k in ("CC", "CXX", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "CARGO_HOME")
                                      or k.startswith("CARGO_PROFILE_")}
    output = args.output / "builds.json"
    source_file = args.snapshot / "src/main.rs"
    original_source = source_file.read_bytes()
    for index, variant in enumerate(ORDER):
        identifier = f"{index + 1}-{variant}"
        target = args.output / "build-targets" / identifier
        if target.exists():
            raise ValueError(f"Refuse cached clean build: {target}")
        argv = build_argv(args.snapshot, target, variant)
        entry = {"id": identifier, "variant": variant, "target": str(target)}
        report["builds"].append(entry)
        save(output, report)
        print(f"Clean build {identifier}", flush=True)
        entry["clean"] = command(argv, cwd=args.snapshot, env=env,
                                 log=args.output / f"{identifier}-clean.log")
        binary = target / "release/apxinf"
        saved = args.output / "artifacts" / identifier / "apxinf"
        saved.parent.mkdir(parents=True)
        shutil.copy2(binary, saved)
        entry["binary"] = identity(saved)
        entry["dependencies"] = command(["otool", "-L", str(saved)], cwd=args.snapshot)["stdout"]
        dependencies = sorted((target / "release/deps").glob("apxinf_model-*.d"))
        if len(dependencies) != 1:
            raise ValueError(f"Expected one model dependency file: {dependencies}")
        dependency_text = dependencies[0].read_text()
        inventory = args.output / f"{identifier}-model-dependencies.d"
        inventory.write_text(dependency_text)
        entry["model_modules"] = {family: f"src/{family}/" in dependency_text
                                  for family in ("llama", "pi05", "qwen3vl", "qwen35")}
        entry["native_archives"] = sorted(str(path.relative_to(target))
                                          for path in target.glob("release/build/apxinf-metal-*/out/*.a"))
        expected_modules = {name: variant == "baseline" or name == "qwen35"
                            for name in ("llama", "pi05", "qwen3vl", "qwen35")}
        if entry["model_modules"] != expected_modules:
            raise ValueError(f"Unexpected model compilation scope: {entry['model_modules']}")
        entry["noop"] = command(argv, cwd=args.snapshot, env=env,
                                log=args.output / f"{identifier}-noop.log")
        try:
            source_file.write_bytes(original_source + b"\n// Incremental rebuild measurement.\n")
            entry["incremental"] = command(argv, cwd=args.snapshot, env=env,
                                           log=args.output / f"{identifier}-incremental.log")
        finally:
            source_file.write_bytes(original_source)
        save(output, report)
        print(json.dumps({"id": identifier, "clean_seconds": entry["clean"]["wall_seconds"],
                          "binary_bytes": entry["binary"]["bytes"], "modules": entry["model_modules"]}), flush=True)
    for variant in ("baseline", "candidate"):
        entry = next(item for item in report["builds"] if item["variant"] == variant)
        target = Path(entry["target"])
        argv = build_argv(args.snapshot, target, variant, example=True)
        entry["measurement_driver_build"] = command(argv, cwd=args.snapshot, env=env,
                                                    log=args.output / f"{variant}-driver.log")
        source = target / "release/examples/specialization_bench"
        destination = args.output / "artifacts" / variant / "specialization_bench"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entry["measurement_driver"] = identity(destination)
    if verify_inputs(args) != prepared:
        raise ValueError("Prepared inputs changed during the build")
    verify_file(report["inputs"])
    report["environment_after"] = system_state()
    report["status"] = "complete"
    save(output, report)


def runtime(args):
    prepared = verify_inputs(args, model=True)
    contract = json.loads(Path(prepared["contract"]["path"]).read_text())
    cases = json.loads(args.cases.read_text())
    workload = contract["workload"]
    generation = contract["generation_measurement"]
    builds = json.loads((args.output / "builds.json").read_text())
    if builds["status"] != "complete":
        raise ValueError("Complete the build comparison first")
    verify_file(builds["inputs"])
    if builds["snapshot_sha256"] != prepared["snapshot_sha256"]:
        raise ValueError("The builds use a different prepared snapshot")
    output = args.output / "runtime.json"
    if output.exists():
        raise ValueError("Refuse to replace an existing runtime campaign")
    report = {"schema": "apxinf-specialization-runtime-v1", "status": "running",
              "model": str(args.model), "environment_before": system_state(),
              "inputs": builds["inputs"], "build_report": identity(args.output / "builds.json"),
              "startup": [], "cli": [], "resident": [], "assets": prepared["assets"]}
    save(output, report)
    variants = {v: next(b for b in builds["builds"] if b["variant"] == v)
                for v in ("baseline", "candidate")}
    for _ in range(3):
        for variant in ORDER:
            artifact = variants[variant]["binary"]
            verify_file(artifact)
            binary = artifact["path"]
            sample = command([binary, "--help"], cwd=args.output)
            verify_file(artifact)
            report["startup"].append({"variant": variant, "wall_seconds": sample["wall_seconds"]})
    for case in cases:
        for index, variant in enumerate(ORDER):
            artifact = variants[variant]["binary"]
            verify_file(artifact)
            argv = [artifact["path"], "generate", "--model", str(args.model), "--prompt", case["prompt"],
                    "--max-tokens", str(workload["max_new_tokens"]), "--max-context", str(workload["max_context"]),
                    "--provider", "native", "--device", "cpu", "--dtype", "fp32",
                    "--metal-w8-lm-head", "--metal-w8-mlp-block", "--json"]
            print(f"CLI request {case['id']} {index + 1} {variant}", flush=True)
            sample = command(argv, cwd=args.output, timeout=300)
            verify_file(artifact)
            receipt = json.loads(sample.pop("stdout"))
            report["cli"].append({"case": case["id"], "variant": variant, **sample, "receipt": receipt})
            save(output, report)
    for index, variant in enumerate(ORDER):
        artifact = variants[variant]["measurement_driver"]
        verify_file(artifact)
        driver = artifact["path"]
        destination = args.output / f"resident-{index + 1}-{variant}.json"
        if destination.exists():
            raise ValueError(f"Refuse to replace a resident receipt: {destination}")
        argv = [driver, "--model", str(args.model), "--cases", str(args.cases),
                "--output", str(destination), "--warmups", str(generation["warmups_per_case_per_process"]),
                "--repeats", str(generation["repeats_per_case_per_process"]),
                "--max-tokens", str(workload["max_new_tokens"])]
        print(f"Resident process {index + 1} {variant}", flush=True)
        sample = command(argv, cwd=args.output, timeout=600,
                         log=args.output / f"resident-{index + 1}-{variant}.log")
        verify_file(artifact)
        report["resident"].append({"variant": variant, **sample, "receipt_path": str(destination),
                                   "receipt_file": identity(destination)})
        save(output, report)
    if verify_inputs(args, model=True) != prepared:
        raise ValueError("Prepared inputs changed during inference")
    verify_file(report["build_report"])
    report["environment_after"] = system_state()
    report["status"] = "complete"
    save(output, report)


def stats(values):
    return {"count": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def path_signature(receipt):
    if (receipt["format"] != "apxinf-qwen35-generation-path-v1"
            or receipt["metal_w8_mlp_block"] is not True or receipt["metal_w8_lm_head"] is not True
            or [row["layer_index"] for row in receipt["mlp_block_layers"]] != list(range(24))
            or not isinstance(receipt["lm_head"], dict)):
        raise ValueError("Generation did not use the declared Metal head and 24 MLP blocks")
    return {key: value for key, value in receipt.items() if key not in ("mlp_block_layers", "lm_head")} | {
        "mlp_layers": [row["layer_index"] for row in receipt["mlp_block_layers"]]}


def check_path_calls(receipt, prefill_calls, decode_calls):
    path_signature(receipt)
    head = receipt["lm_head"]
    if (head["prefill_calls"] != prefill_calls or head["decode_calls"] != decode_calls
            or head["teacher_calls"] != 0
            or any(row["decode_calls"] != decode_calls for row in receipt["mlp_block_layers"])):
        raise ValueError("Generation path call counts differ from the returned token counts")


def validate_comparison(args, builds, runtime_report):
    prepared = verify_inputs(args)
    verify_file(builds["inputs"])
    verify_file(runtime_report["inputs"])
    verify_file(runtime_report["build_report"])
    if (builds["status"] != "complete" or runtime_report["status"] != "complete"
            or runtime_report["inputs"] != builds["inputs"]
            or builds["snapshot_sha256"] != prepared["snapshot_sha256"]
            or runtime_report["assets"] != prepared["assets"]
            or runtime_report["model"] != prepared["model"]):
        raise ValueError("Reports are incomplete or use different prepared inputs")
    contract = json.loads(Path(prepared["contract"]["path"]).read_text())
    case_ids = [case["id"] for case in json.loads(Path(prepared["cases"]["path"]).read_text())]
    generation = contract["generation_measurement"]
    warmups = generation["warmups_per_case_per_process"]
    repeats = generation["repeats_per_case_per_process"]
    if ([row["variant"] for row in builds["builds"]] != list(ORDER)
            or [row["variant"] for row in runtime_report["resident"]] != list(ORDER)
            or [row["variant"] for row in runtime_report["startup"]] != list(ORDER) * 3
            or [(row["case"], row["variant"]) for row in runtime_report["cli"]]
            != [(case, variant) for case in case_ids for variant in ORDER]):
        raise ValueError("The sample grid or execution order is incomplete")
    for row in builds["builds"]:
        verify_file(row["binary"])
        if "measurement_driver" in row:
            verify_file(row["measurement_driver"])
        expected = {name: row["variant"] == "baseline" or name == "qwen35"
                    for name in ("llama", "pi05", "qwen3vl", "qwen35")}
        if row["model_modules"] != expected:
            raise ValueError("Model source pruning evidence is invalid")
    expected_settings = {"max_context": 2048, "max_tokens": contract["workload"]["max_new_tokens"],
                         "warmups_per_case": warmups, "repeats_per_case": repeats,
                         "batch": 1, "concurrency": 1, "eos_stopping": True,
                         "metal_w8_lm_head": True, "metal_w8_mlp_block": True,
                         "streaming_output": False}
    references = {}
    settings_reference = None
    documents = {variant: [] for variant in FEATURES}
    for sample in runtime_report["resident"]:
        verify_file(sample["receipt_file"])
        if sample["receipt_file"]["path"] != sample["receipt_path"]:
            raise ValueError("Resident receipt paths differ")
        document = json.loads(Path(sample["receipt_path"]).read_text())
        if (document["schema"] != "apxinf-specialization-generation-v1" or document["status"] != "complete"
                or document["model"] != prepared["model"]
                or document["config_sha256"] != contract["expected_config_sha256"]
                or document["cases_sha256"] != prepared["cases"]["sha256"]
                or any(document["settings"].get(key) != value for key, value in expected_settings.items())):
            raise ValueError("Resident settings or input identities differ")
        if settings_reference is None:
            settings_reference = document["settings"]
        if document["settings"] != settings_reference:
            raise ValueError("Resident settings changed between processes")
        expected_grid = [(case, index, index < warmups) for case in case_ids for index in range(warmups + repeats)]
        if [(row["case"], row["repetition"], row["warmup"]) for row in document["runs"]] != expected_grid:
            raise ValueError("A resident receipt has missing, duplicate, or reordered runs")
        prefills = decodes = 0
        for row in document["runs"]:
            tokens = row["generated_token_ids"]
            if not tokens or len(tokens) > expected_settings["max_tokens"] or not row["prompt_token_ids"]:
                raise ValueError("Invalid prompt or output token counts")
            prefills += 1
            decodes += len(tokens) - 1
            check_path_calls(row["generation_path_receipt"], prefills, decodes)
            value = {"input": row["prompt_token_ids"], "output": tokens, "text": row["text"],
                     "path": path_signature(row["generation_path_receipt"])}
            if references.setdefault(row["case"], value) != value:
                raise ValueError(f"Resident input, output, or execution path differs: {row['case']}")
        documents[sample["variant"]].append(document)
    for sample in runtime_report["cli"]:
        receipt = sample["receipt"]
        reference = references[sample["case"]]
        if (receipt["format"] != "apxinf-generation-v1" or receipt["model_type"] not in ("qwen35", "qwen3_5")
                or receipt["device"] != "cpu" or receipt["dtype"] != "fp32"
                or receipt["build"]["matmul_feature"] != "accelerate"
                or receipt["build"]["metal_w8_lm_head"] is not True
                or receipt["build"]["metal_w8_mlp_block"] is not True
                or receipt["prompt_token_count"] != len(reference["input"])
                or receipt["generated_token_ids"] != reference["output"]
                or path_signature(receipt["generation_path"]) != reference["path"]):
            raise ValueError(f"CLI output or execution path differs: {sample['case']}")
        check_path_calls(receipt["generation_path"], 1, len(reference["output"]) - 1)
    return documents, references


def summarize(args):
    summary = {"accepted": False, "formal_performance_accepted": False,
               "acceptance_scope": "Prepared identities, sample completeness, and corpus equivalence only.",
               "build": {}, "runtime": {}, "comparison_scope": "Model-family source pruning only"}
    save(args.output / "summary.json", summary)
    try:
        builds = json.loads((args.output / "builds.json").read_text())
        runtime_report = json.loads((args.output / "runtime.json").read_text())
        documents, references = validate_comparison(args, builds, runtime_report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        summary["failure"] = str(error)
        save(args.output / "summary.json", summary)
        raise ValueError(f"Comparison rejected: {error}") from error
    for variant in ("baseline", "candidate"):
        selected = [row for row in builds["builds"] if row["variant"] == variant]
        summary["build"][variant] = {kind + "_seconds": stats([row[kind]["wall_seconds"] for row in selected])
                                      for kind in ("clean", "noop", "incremental")}
        summary["build"][variant]["binary_bytes"] = stats([row["binary"]["bytes"] for row in selected])
        summary["build"][variant]["modules"] = selected[0]["model_modules"]
        records = documents[variant]
        result = {"load_ms": stats([row["load_ms"] for row in records]),
                  "first_request_ms": stats([row["first_request_ms"] for row in records]),
                  "help_process_ms": stats([row["wall_seconds"] * 1000 for row in runtime_report["startup"]
                                             if row["variant"] == variant]), "cases": {}}
        names = sorted({row["case"] for document in records for row in document["runs"]})
        for name in names:
            rows = [row for document in records for row in document["runs"]
                    if row["case"] == name and not row["warmup"]]
            case = {metric: stats([row[metric] for row in rows if row[metric] is not None])
                    for metric in ("request_ms", "ttft_wall_ms", "decode_tps")
                    if any(row[metric] is not None for row in rows)}
            case["output_tokens"] = [len(row["generated_token_ids"]) for row in rows]
            case["text"] = rows[0]["text"]
            case["cli_process_ms"] = stats([row["wall_seconds"] * 1000 for row in runtime_report["cli"]
                                             if row["variant"] == variant and row["case"] == name])
            result["cases"][name] = case
        summary["runtime"][variant] = result
    summary["all_runs_equal_tokens_by_case"] = {key: True for key in references}
    summary["all_cli_outputs_equal_by_case"] = {key: True for key in references}
    summary["cli_matches_resident_by_case"] = {key: True for key in references}
    summary["limitations"] = [
        "The clean build uses an empty Cargo target directory and a warm dependency download cache.",
        "Two clean builds per selection support a bounded observation, not a universal speedup claim.",
        "Resident timing has two independent processes per selection; each contributes three measured requests per case.",
        "Resident timing belongs to the separate measurement driver; CLI timing covers the delivered executable.",
        "The baseline and candidate retain the same native inference algorithms and runtime Metal compilation.",
        "Model-family pruning does not remove every unused operator or CLI feature.",
        "Process-cold measurements do not clear filesystem or driver caches.",
        "The small real-question corpus checks equivalence, not general model quality or all supported shapes.",
        "Acceptance does not establish a speedup or quiet-host formal performance qualification.",
    ]
    summary["accepted"] = True
    save(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("snapshot", "prepare", "build", "runtime", "summarize"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--contract", type=Path, default=Path(__file__).with_name("contract.json"))
    args = parser.parse_args()
    args.snapshot = args.snapshot.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "snapshot":
        snapshot(args.source.resolve(strict=True), args.snapshot)
    elif args.phase == "prepare":
        args.model = args.model.resolve(strict=True)
        args.cases = args.cases.resolve(strict=True)
        args.contract = args.contract.resolve(strict=True)
        prepare(args)
    elif args.phase == "build":
        build(args)
    elif args.phase == "runtime":
        args.model = args.model.resolve(strict=True)
        args.cases = args.cases.resolve(strict=True)
        runtime(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
