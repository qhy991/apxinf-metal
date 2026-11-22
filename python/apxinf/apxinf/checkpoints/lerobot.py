"""Translate a LeRobot serialized processor pipeline into canonical facts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .descriptor import (
    IDENTITY,
    IDENTITY_DECLARED,
    IDENTITY_MISSING_STATS,
    MEAN_STD,
    QUANTILE,
    RESOLVED,
    NormalizationPlan,
    TokenizerSpec,
    TransformSpec,
)
from .safetensors_state import SafeTensorStateError, load_state_file

PREPROCESSOR_NAME = "policy_preprocessor.json"
POSTPROCESSOR_NAME = "policy_postprocessor.json"

_MODES = {
    "IDENTITY": IDENTITY,
    "MEAN_STD": MEAN_STD,
    "QUANTILES": QUANTILE,
    "QUANTILE": QUANTILE,
}


class LeRobotProcessorError(ValueError):
    pass


def has_processor_layout(root) -> bool:
    root = Path(root)
    return (root / PREPROCESSOR_NAME).is_file() or (root / POSTPROCESSOR_NAME).is_file()


def load_processor_plan(
    root, config: Mapping[str, Any]
) -> Tuple[NormalizationPlan, Optional[TokenizerSpec]]:
    """Load both LeRobot processor manifests and their referenced state."""
    root = Path(root)
    pre_path, post_path = root / PREPROCESSOR_NAME, root / POSTPROCESSOR_NAME
    missing = [str(path) for path in (pre_path, post_path) if not path.is_file()]
    if missing:
        raise LeRobotProcessorError(
            "LeRobot processor layout is incomplete; missing " + ", ".join(missing)
        )
    pre, post = _read_json(pre_path), _read_json(post_path)
    pre_step = _find_step(pre, "normalizer_processor", pre_path)
    post_step = _find_step(post, "unnormalizer_processor", post_path)

    state_key, state_width = _config_feature(config, "STATE", inputs=True)
    action_key, action_width = _config_feature(config, "ACTION", inputs=False)
    pre_stats, pre_source = _load_step_state(root, pre_step, pre_path)
    post_stats, post_source = _load_step_state(root, post_step, post_path)

    state = _transform_from_step(
        pre_step,
        feature_type="STATE",
        fallback_key=state_key,
        fallback_width=state_width,
        tensors=pre_stats,
        source=pre_source,
    )
    action = _transform_from_step(
        post_step,
        feature_type="ACTION",
        fallback_key=action_key,
        fallback_width=action_width,
        tensors=post_stats,
        source=post_source,
    )
    tokenizer = _tokenizer_spec(pre, pre_path)
    return NormalizationPlan(state=state, action=action), tokenizer


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LeRobotProcessorError(f"read {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("steps"), list):
        raise LeRobotProcessorError(f"{path}: expected an object with a steps list")
    return document


def _find_step(document: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    found = [
        step
        for step in document["steps"]
        if isinstance(step, dict) and step.get("registry_name") == name
    ]
    if len(found) != 1:
        raise LeRobotProcessorError(
            f"{path}: expected exactly one {name!r} step, found {len(found)}"
        )
    return found[0]


def _safe_state_path(root: Path, value: Any, manifest: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise LeRobotProcessorError(f"{manifest}: state_file must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise LeRobotProcessorError(f"{manifest}: unsafe state_file {value!r}")
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    if root_resolved != candidate and root_resolved not in candidate.parents:
        raise LeRobotProcessorError(
            f"{manifest}: state_file escapes checkpoint root: {value!r}"
        )
    return candidate


def _load_step_state(
    root: Path, step: Mapping[str, Any], manifest: Path
) -> Tuple[Optional[Mapping[str, np.ndarray]], str]:
    value = step.get("state_file")
    if value is None:
        return None, str(manifest)
    path = _safe_state_path(root, value, manifest)
    if not path.is_file():
        raise LeRobotProcessorError(
            f"{manifest}: declared state_file {value!r} does not exist at {path}"
        )
    try:
        state = load_state_file(path)
        if not state:
            raise LeRobotProcessorError(f"{path}: state_file contains no tensors")
        return state, str(path)
    except SafeTensorStateError as exc:
        raise LeRobotProcessorError(str(exc)) from exc


def _config_feature(
    config: Mapping[str, Any], feature_type: str, *, inputs: bool
) -> Tuple[str, int]:
    field = "input_features" if inputs else "output_features"
    features = config.get(field, {})
    if not isinstance(features, dict):
        raise LeRobotProcessorError(f"config.json: {field} must be an object")
    matches = []
    for key, feature in features.items():
        if isinstance(feature, dict) and str(feature.get("type", "")).upper() == feature_type:
            shape = feature.get("shape")
            if isinstance(shape, list) and shape and isinstance(shape[-1], int):
                matches.append((key, int(shape[-1])))
    if len(matches) != 1:
        raise LeRobotProcessorError(
            f"config.json: expected exactly one {feature_type} feature in {field}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _feature_from_step(
    step: Mapping[str, Any], feature_type: str, fallback_key: str, fallback_width: int
) -> Tuple[str, int]:
    config = step.get("config", {})
    features = config.get("features", {}) if isinstance(config, dict) else {}
    matches = []
    if isinstance(features, dict):
        for key, feature in features.items():
            if (
                isinstance(feature, dict)
                and str(feature.get("type", "")).upper() == feature_type
            ):
                shape = feature.get("shape")
                if isinstance(shape, (list, tuple)) and shape and isinstance(shape[-1], int):
                    matches.append((key, int(shape[-1])))
    if len(matches) > 1:
        raise LeRobotProcessorError(
            f"processor step has multiple {feature_type} features: {[key for key, _ in matches]}"
        )
    if matches and matches[0] != (fallback_key, fallback_width):
        raise LeRobotProcessorError(
            f"processor {feature_type} feature {matches[0]!r} disagrees with "
            f"config.json {(fallback_key, fallback_width)!r}"
        )
    return matches[0] if matches else (fallback_key, fallback_width)


def _transform_from_step(
    step: Mapping[str, Any],
    *,
    feature_type: str,
    fallback_key: str,
    fallback_width: int,
    tensors: Optional[Mapping[str, np.ndarray]],
    source: str,
) -> TransformSpec:
    config = step.get("config", {})
    if not isinstance(config, dict):
        raise LeRobotProcessorError("processor step config must be an object")
    norm_map = config.get("norm_map", {})
    serialized_mode = (
        norm_map.get(feature_type, "IDENTITY")
        if isinstance(norm_map, dict)
        else "IDENTITY"
    )
    mode = _MODES.get(str(serialized_mode).upper())
    if mode is None:
        raise LeRobotProcessorError(
            f"unsupported LeRobot normalization mode {serialized_mode!r} for {feature_type}"
        )
    key, width = _feature_from_step(step, feature_type, fallback_key, fallback_width)
    eps = float(config.get("eps", 1e-8))
    if mode == IDENTITY:
        return TransformSpec.identity(
            key, width, source=source, status=IDENTITY_DECLARED
        )

    names = ("mean", "std") if mode == MEAN_STD else ("q01", "q99")
    values: Dict[str, Tuple[float, ...]] = {}
    present = () if tensors is None else tuple(
        name for name in names if f"{key}.{name}" in tensors
    )
    if not present:
        if tensors is not None and feature_type == "ACTION":
            raise LeRobotProcessorError(
                f"{source}: no tensors for {key!s} ACTION {mode} normalization"
            )
        return TransformSpec.identity(
            key,
            width,
            source=source,
            status=IDENTITY_MISSING_STATS,
        )
    for name in names:
        tensor = None if tensors is None else tensors.get(f"{key}.{name}")
        if tensor is None:
            raise LeRobotProcessorError(
                f"{source}: no tensor {key}.{name!s} for {feature_type} {mode} normalization"
            )
        flat = np.asarray(tensor).reshape(-1)
        values[name] = tuple(float(value) for value in flat)
    # Canonical TransformSpec quantiles use ApxInf/OpenPI's "add eps to every
    # span" rule.  LeRobot instead substitutes eps only for a zero span.  Apply
    # that family-specific rule here so consumers never need to know which
    # exporter produced the plan.
    canonical_eps = eps
    if mode == QUANTILE:
        q01, q99 = values["q01"], values["q99"]
        values["q99"] = tuple(
            low + eps if high == low else high for low, high in zip(q01, q99)
        )
        canonical_eps = 0.0
    try:
        return TransformSpec(
            feature_key=key,
            mode=mode,
            width=width,
            eps=canonical_eps,
            values=values,
            source=source,
            status=RESOLVED,
        )
    except ValueError as exc:
        raise LeRobotProcessorError(f"{source}: {exc}") from exc


def _tokenizer_spec(document: Mapping[str, Any], path: Path) -> Optional[TokenizerSpec]:
    steps = [
        step
        for step in document["steps"]
        if isinstance(step, dict) and step.get("registry_name") == "tokenizer_processor"
    ]
    if not steps:
        return None
    if len(steps) != 1:
        raise LeRobotProcessorError(f"{path}: multiple tokenizer_processor steps")
    config = steps[0].get("config", {})
    name = config.get("tokenizer_name") if isinstance(config, dict) else None
    if not isinstance(name, str) or not name:
        raise LeRobotProcessorError(f"{path}: tokenizer_processor has no tokenizer_name")
    return TokenizerSpec(name=name, source=str(path))
