"""Translate OpenPI PI0.5 normalization statistics into canonical facts.

Layout resolution decides *which* ``norm_stats.json`` wins and
:mod:`.norm_stats` decodes its JSON envelope.  This family adapter owns the
remaining OpenPI semantics: PI0.5 consumes quantiles, action statistics are
required, state statistics are optional, and OpenPI adds ``1e-6`` to every
quantile span.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .descriptor import QUANTILE, NormalizationPlan, TransformSpec
from .norm_stats import NormStatsError, read_norm_stats

OPENPI_QUANTILE_EPS = 1e-6


class OpenPINormalizationError(ValueError):
    """OpenPI normalization statistics are absent or semantically invalid."""


def load_normalization_plan(
    path,
    *,
    action_key: str = "actions",
    state_key: Optional[str] = "state",
) -> NormalizationPlan:
    """Build PI0.5's canonical state/action transforms from ``norm_stats.json``."""
    path = Path(path)
    try:
        stats = read_norm_stats(path)
    except NormStatsError as exc:
        raise OpenPINormalizationError(str(exc)) from exc

    action = _quantile_transform(stats, action_key, path=path, required=True)
    assert action is not None
    state = (
        _quantile_transform(stats, state_key, path=path, required=False)
        if state_key is not None
        else None
    )
    return NormalizationPlan(state=state, action=action)


def _quantile_transform(
    stats: Mapping[str, Any],
    key: str,
    *,
    path: Path,
    required: bool,
) -> Optional[TransformSpec]:
    entry = stats.get(key)
    if entry is None:
        if not required:
            return None
        raise OpenPINormalizationError(
            f"{path} has no entry {key!r}; keys: {sorted(stats)}"
        )
    if not isinstance(entry, Mapping):
        raise OpenPINormalizationError(f"{path}: {key} must be a JSON object")

    values = {}
    for name in ("q01", "q99"):
        raw = entry.get(name)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise OpenPINormalizationError(
                f"{path}: {key}.{name} must be a numeric vector"
            )
        try:
            values[name] = tuple(float(value) for value in raw)
        except (TypeError, ValueError) as exc:
            raise OpenPINormalizationError(
                f"{path}: {key}.{name} must be a numeric vector"
            ) from exc

    width = len(values["q01"])
    if len(values["q99"]) != width:
        raise OpenPINormalizationError(
            f"{path}: {key}.q01 has width {width}, but q99 has width "
            f"{len(values['q99'])}"
        )
    try:
        return TransformSpec(
            feature_key=key,
            mode=QUANTILE,
            width=width,
            eps=OPENPI_QUANTILE_EPS,
            values=values,
            source=str(path),
        )
    except ValueError as exc:
        raise OpenPINormalizationError(f"{path}: {exc}") from exc
