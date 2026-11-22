"""Canonical, serialization-friendly checkpoint facts.

External checkpoint layouts describe the same policy facts in incompatible
ways.  Adapters translate those files into these small immutable values; policy
construction consumes the values without depending on the source layout. Values
use serialization-friendly Python types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

IDENTITY = "identity"
MEAN_STD = "mean_std"
QUANTILE = "quantile"

RESOLVED = "resolved"
IDENTITY_DECLARED = "identity_declared"
IDENTITY_MISSING_STATS = "identity_missing_stats"


@dataclass(frozen=True)
class TransformSpec:
    """One feature transform, independent of its source layout.

    ``eps`` has canonical ApxInf semantics: quantile transforms add it to every
    ``q99 - q01`` span, while mean/std transforms add it only to the input-side
    normalization denominator.  A family with a different rule must encode the
    equivalent values before constructing this descriptor; transform
    materializers never branch on :attr:`source`.
    """

    feature_key: str
    mode: str
    width: int
    eps: float
    values: Mapping[str, Tuple[float, ...]] = field(default_factory=dict)
    source: str = ""
    status: str = RESOLVED

    def __post_init__(self) -> None:
        if self.mode not in (IDENTITY, MEAN_STD, QUANTILE):
            raise ValueError(f"unsupported normalization mode {self.mode!r}")
        if self.width <= 0:
            raise ValueError(f"normalization width must be positive, got {self.width}")
        if not math.isfinite(self.eps) or self.eps < 0:
            raise ValueError(
                f"normalization epsilon must be finite and non-negative, got {self.eps}"
            )
        required = {
            IDENTITY: (),
            MEAN_STD: ("mean", "std"),
            QUANTILE: ("q01", "q99"),
        }[self.mode]
        for name in required:
            values = self.values.get(name)
            if values is None:
                raise ValueError(f"{self.mode} transform has no {name}")
            if len(values) != self.width:
                raise ValueError(
                    f"{self.feature_key}.{name} has width {len(values)}, expected {self.width}"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{self.feature_key}.{name} contains non-finite values")

    @classmethod
    def identity(cls, feature_key: str, width: int, *, source: str, status: str):
        return cls(
            feature_key=feature_key,
            mode=IDENTITY,
            width=width,
            eps=0.0,
            source=source,
            status=status,
        )


@dataclass(frozen=True)
class NormalizationPlan:
    """Canonical state-input and action-output transforms for one policy."""

    state: Optional[TransformSpec]
    action: TransformSpec


@dataclass(frozen=True)
class TokenizerSpec:
    """The tokenizer identity declared by an upstream processor pipeline."""

    name: str
    source: str
