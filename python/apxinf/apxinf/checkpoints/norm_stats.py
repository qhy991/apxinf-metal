"""Decode the JSON envelope used by ``norm_stats.json`` files.

This module owns serialization only.  It knows that exporters use either a
flat object or ``{"norm_stats": {...}}``; it does not know which feature names
PI0.5 uses, which statistics a model family consumes, or how those values map
to an ApxInf transform.  Those semantics belong to a family adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class NormStatsError(ValueError):
    """A ``norm_stats.json`` file is unreadable or has the wrong envelope."""


def read_norm_stats(path) -> Mapping[str, Any]:
    """Return the decoded feature table without interpreting its entries."""
    path = Path(path)
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise NormStatsError(f"read {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise NormStatsError(f"{path}: expected a JSON object")
    stats = document.get("norm_stats", document)
    if not isinstance(stats, dict):
        raise NormStatsError(f"{path}: norm_stats must be a JSON object")

    return stats
