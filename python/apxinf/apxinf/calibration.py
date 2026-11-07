"""Model-neutral orchestration for offline quantization calibration.

Models own their execution plan, preprocessing, and activation capture.  This
module owns the reusable iteration, aggregation, coverage validation, scale
generation, and manifest contract around those model-specific pieces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Protocol

__all__ = [
    "CalibrationContext",
    "CalibrationPlan",
    "CalibrationRunner",
    "CalibrationTarget",
    "ConsumerContract",
    "CaptureSite",
    "Fp8ExecutionPlan",
    "ObservationAdapter",
    "QuantizationSpec",
    "QuantizedOperator",
    "adapt_records",
    "build_calibration_document",
    "merge_records",
]


FP8_MAX = 448.0
DEFAULT_SCHEMA = "apxinf.fp8-calibration.v1"
DEFAULT_SCALE_RULE = "max(amax*margin/448,1e-8)"
_DEFAULT_OPERATOR_KINDS = frozenset({"linear", "gemm"})


class ConsumerContract(str, Enum):
    """Where scale-to-FP8-consumer coverage is validated."""

    MANIFEST = "manifest"
    RUNTIME_VALIDATED = "runtime-validated"


@dataclass(frozen=True)
class QuantizedOperator:
    """One stable logical operator in a model's actual FP8 execution plan."""

    stable_id: str
    kind: str
    output: Optional[str] = None
    quantized: bool = True

    def __post_init__(self) -> None:
        if not self.stable_id or not self.kind:
            raise ValueError("quantized operators require stable_id and kind")


@dataclass(frozen=True)
class Fp8ExecutionPlan:
    """The operators selected by the runtime, not a scan of module names."""

    operators: tuple[QuantizedOperator, ...]
    activation_mode: str = "static"

    def __post_init__(self) -> None:
        if self.activation_mode not in {"static", "dynamic"}:
            raise ValueError("activation_mode must be 'static' or 'dynamic'")
        identifiers = [operator.stable_id for operator in self.operators]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("FP8 execution plan contains duplicate stable operator IDs")


@dataclass(frozen=True)
class CaptureSite:
    """A statistic captured at a stable site and its first FP8 consumer."""

    name: str
    consumer: Optional[str] = None
    statistic: str = "absmax"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("capture site name must not be empty")
        _validate_statistic(self.statistic)


@dataclass(frozen=True)
class CalibrationPlan:
    model_family: str
    capture_sites: tuple[CaptureSite, ...]
    consumers: Mapping[str, str]
    activation_mode: str = "static"
    schema: str = DEFAULT_SCHEMA
    fp8_format: str = "e4m3fn"
    scale_rule: str = DEFAULT_SCALE_RULE
    minimum_amax: Mapping[str, float] = field(default_factory=dict)
    seed_algorithm: str = "seed-plus-sample-context-v1"
    seed_sequence: str = "[base_seed,sample_index]"
    consumer_contract: ConsumerContract = ConsumerContract.MANIFEST

    def __post_init__(self) -> None:
        if not isinstance(self.consumer_contract, ConsumerContract):
            raise ValueError("consumer_contract must be a ConsumerContract")
        if not self.model_family:
            raise ValueError("calibration plan model_family must not be empty")
        if self.activation_mode not in {"static", "dynamic"}:
            raise ValueError("activation_mode must be 'static' or 'dynamic'")
        if self.activation_mode == "dynamic" and self.capture_sites:
            raise ValueError("dynamic activation FP8 must not declare static capture sites")
        names = [site.name for site in self.capture_sites]
        if len(names) != len(set(names)):
            raise ValueError("calibration plan contains duplicate stable capture sites")
        if self.consumer_contract is ConsumerContract.MANIFEST:
            unknown_scales = sorted(set(self.consumers.values()) - set(names))
            if unknown_scales:
                raise ValueError(f"FP8 consumers reference unknown scales: {unknown_scales}")
            dangling = sorted(set(names) - set(self.consumers.values()))
            if dangling:
                raise ValueError(
                    f"generated scale has no FP8 consumer: {', '.join(dangling)}"
                )
        unknown_minima = sorted(set(self.minimum_amax) - set(names))
        if unknown_minima:
            raise ValueError(f"minimum_amax references unknown sites: {unknown_minima}")
        for name, value in self.minimum_amax.items():
            if not _finite(value) or value < 0.0:
                raise ValueError(f"minimum_amax for {name} must be finite and non-negative")

    @classmethod
    def runtime_validated_sites(
        cls,
        model_family: str,
        sites: Iterable[str],
        *,
        schema: str,
        seed_algorithm: str,
    ) -> "CalibrationPlan":
        """Adapt a legacy runtime plan that validates consumers internally."""
        capture_sites = tuple(CaptureSite(site) for site in sites)
        return cls(
            model_family=model_family,
            capture_sites=capture_sites,
            consumers={},
            schema=schema,
            seed_algorithm=seed_algorithm,
            consumer_contract=ConsumerContract.RUNTIME_VALIDATED,
        )

    @property
    def requires_calibration(self) -> bool:
        return self.activation_mode == "static" and bool(self.capture_sites)

    @property
    def sites(self) -> tuple[str, ...]:
        return tuple(site.name for site in self.capture_sites)


@dataclass(frozen=True)
class QuantizationSpec:
    """Small model override layered on conventional static-FP8 defaults."""

    model_family: str
    excluded_outputs: frozenset[str] = frozenset()
    shared_scales: Mapping[str, str] = field(default_factory=dict)
    custom_captures: tuple[CaptureSite, ...] = ()
    default_statistic: str = "absmax"
    schema: str = DEFAULT_SCHEMA
    fp8_format: str = "e4m3fn"
    scale_rule: str = DEFAULT_SCALE_RULE
    minimum_amax: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_family:
            raise ValueError("model_family must not be empty")
        _validate_statistic(self.default_statistic)

    def plan_for(self, execution: Fp8ExecutionPlan) -> CalibrationPlan:
        if execution.activation_mode == "dynamic":
            return CalibrationPlan(
                model_family=self.model_family,
                capture_sites=(),
                consumers={},
                activation_mode="dynamic",
                schema=self.schema,
                fp8_format=self.fp8_format,
                scale_rule=self.scale_rule,
            )

        custom_by_consumer = {
            site.consumer: site for site in self.custom_captures if site.consumer is not None
        }
        if len(custom_by_consumer) != sum(
            site.consumer is not None for site in self.custom_captures
        ):
            raise ValueError("quantization specification has duplicate custom consumers")

        execution_outputs = {
            operator.output for operator in execution.operators if operator.output is not None
        }
        unknown_exclusions = sorted(self.excluded_outputs - execution_outputs)
        if unknown_exclusions:
            raise ValueError(
                f"excluded outputs are absent from the FP8 execution plan: {unknown_exclusions}"
            )

        eligible_consumers = {
            operator.stable_id
            for operator in execution.operators
            if operator.quantized and operator.output not in self.excluded_outputs
        }
        unknown_shared = sorted(set(self.shared_scales) - eligible_consumers)
        if unknown_shared:
            raise ValueError(f"shared scales reference unknown FP8 consumers: {unknown_shared}")
        unknown_custom = sorted(set(custom_by_consumer) - eligible_consumers)
        if unknown_custom:
            raise ValueError(f"custom captures reference unknown FP8 consumers: {unknown_custom}")

        captures: list[CaptureSite] = []
        capture_names: set[str] = set()
        capture_statistics: dict[str, str] = {}
        consumers: dict[str, str] = {}

        def add_capture(site: CaptureSite, consumer: str) -> None:
            consumers[consumer] = site.name
            if site.name in capture_names:
                if capture_statistics[site.name] != site.statistic:
                    raise ValueError(
                        f"conflicting statistics for shared scale {site.name}: "
                        f"{capture_statistics[site.name]} vs {site.statistic}"
                    )
            else:
                captures.append(
                    CaptureSite(site.name, consumer=consumer, statistic=site.statistic)
                )
                capture_names.add(site.name)
                capture_statistics[site.name] = site.statistic

        for operator in execution.operators:
            if not operator.quantized or operator.output in self.excluded_outputs:
                continue
            custom = custom_by_consumer.get(operator.stable_id)
            if custom is not None:
                add_capture(custom, operator.stable_id)
                continue
            if operator.kind not in _DEFAULT_OPERATOR_KINDS:
                raise ValueError(
                    f"quantized {operator.kind} operator {operator.stable_id} "
                    "requires a custom capture"
                )
            name = self.shared_scales.get(
                operator.stable_id, f"{operator.stable_id}.input"
            )
            add_capture(
                CaptureSite(name, statistic=self.default_statistic),
                operator.stable_id,
            )

        for site in self.custom_captures:
            if site.name not in capture_names:
                captures.append(site)
                capture_names.add(site.name)
                capture_statistics[site.name] = site.statistic

        dangling = sorted(capture_names - set(consumers.values()))
        if dangling:
            raise ValueError(
                f"generated scale has no FP8 consumer: {', '.join(dangling)}"
            )
        missing_scales = sorted(set(consumers.values()) - capture_names)
        if missing_scales:
            raise ValueError(
                f"FP8 consumers reference capture sites not in the plan: {missing_scales}"
            )
        return CalibrationPlan(
            model_family=self.model_family,
            capture_sites=tuple(captures),
            consumers=consumers,
            schema=self.schema,
            fp8_format=self.fp8_format,
            scale_rule=self.scale_rule,
            minimum_amax=dict(self.minimum_amax),
        )


@dataclass(frozen=True)
class CalibrationContext:
    seed: int
    sample_index: int


class CalibrationTarget(Protocol):
    """Public model/policy seam used by :class:`CalibrationRunner`."""

    def collect_calibration(
        self, observation: Mapping[str, Any], context: CalibrationContext
    ) -> Mapping[str, float]: ...


class ObservationAdapter(Protocol):
    """Translate an external dataset record into a public Observation only."""

    def to_observation(self, record: Any) -> Mapping[str, Any]: ...


def adapt_records(
    records: Iterable[Any], adapter: ObservationAdapter
) -> Iterable[Mapping[str, Any]]:
    """Lazily adapt dataset records without performing model preprocessing."""
    for record in records:
        observation = adapter.to_observation(record)
        if not isinstance(observation, Mapping):
            raise TypeError("dataset adapters must return public Observation mappings")
        yield observation


class CalibrationRunner:
    """Run a model's public calibration seam over public Observations."""

    def __init__(
        self,
        target: CalibrationTarget,
        plan: CalibrationPlan,
        *,
        checkpoint: str,
        data_identity: str,
        source_revision: str,
        device: Mapping[str, str],
        margin: float = 1.1,
        seed: int = 0,
        bootstrap: bool = False,
    ) -> None:
        if not checkpoint or not data_identity or not source_revision:
            raise ValueError("checkpoint, data_identity, and source_revision are required")
        if not _finite(margin) or margin < 1.0:
            raise ValueError("margin must be finite and >= 1")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.target = target
        self.plan = plan
        self.checkpoint = checkpoint
        self.data_identity = data_identity
        self.source_revision = source_revision
        self.device = dict(device)
        self.margin = float(margin)
        self.seed = seed
        self.bootstrap = bootstrap

    def run(
        self, observations: Iterable[Mapping[str, Any]]
    ) -> Optional[dict[str, Any]]:
        if not self.plan.requires_calibration:
            return None

        aggregate: dict[str, float] = {}
        sample_count = 0
        for sample_count, observation in enumerate(observations, start=1):
            if not isinstance(observation, Mapping):
                raise TypeError("calibration datasets must yield Observation mappings")
            records = self.target.collect_calibration(
                observation,
                CalibrationContext(seed=self.seed, sample_index=sample_count - 1),
            )
            merge_records(aggregate, records)
        if sample_count == 0:
            raise ValueError("calibration dataset did not yield any Observations")
        return build_calibration_document(
            aggregate,
            plan=self.plan,
            checkpoint=self.checkpoint,
            data_identity=self.data_identity,
            source_revision=self.source_revision,
            device=self.device,
            margin=self.margin,
            seed=self.seed,
            bootstrap=self.bootstrap,
            sample_count=sample_count,
        )


def build_calibration_document(
    records: Mapping[str, float],
    *,
    plan: CalibrationPlan,
    checkpoint: str,
    data_identity: str,
    source_revision: str,
    device: Mapping[str, str],
    margin: float,
    seed: int,
    bootstrap: bool,
    sample_count: int,
) -> dict[str, Any]:
    """Validate aggregated records and build the common calibration manifest."""
    expected = set(plan.sites)
    observed = set(records)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        raise ValueError(
            f"calibration site coverage mismatch: missing={missing}, unknown={unknown}"
        )
    if sample_count < 1:
        raise ValueError("calibration sample_count must be positive")
    if not _finite(margin) or margin < 1.0:
        raise ValueError("margin must be finite and >= 1")

    scales = {}
    for name in sorted(records):
        value = float(records[name])
        if not _finite(value) or value < 0.0:
            raise ValueError(
                f"calibration returned invalid amax/statistic for {name}: {value}"
            )
        amax = max(value, float(plan.minimum_amax.get(name, 0.0)))
        scales[name] = {
            "amax": amax,
            "scale": max(amax * margin / FP8_MAX, 1.0e-8),
        }
    statistics = {site.name: site.statistic for site in plan.capture_sites}
    unique_statistics = sorted(set(statistics.values()))
    plan_document: dict[str, Any] = {"sites": list(plan.sites)}
    if plan.consumer_contract is ConsumerContract.MANIFEST:
        plan_document.update(
            consumers=dict(sorted(plan.consumers.items())),
            statistics=statistics,
        )
    return {
        "schema": plan.schema,
        "model": {"family": plan.model_family, "checkpoint": checkpoint},
        "quantization": {
            "format": plan.fp8_format,
            "statistic": unique_statistics[0]
            if len(unique_statistics) == 1
            else "per-site",
            "scale_rule": plan.scale_rule,
            "margin": margin,
        },
        "calibration_data": {
            "identity": data_identity,
            "kind": "synthetic-zero-fixture" if bootstrap else "representative",
            "production": not bootstrap,
            "sample_count": sample_count,
        },
        "seed_policy": {
            "algorithm": plan.seed_algorithm,
            "base_seed": seed,
            "sample_sequence": plan.seed_sequence,
        },
        "source_revision": source_revision,
        "device": dict(device),
        "plan": plan_document,
        "observed_sites": sorted(records),
        "scales": scales,
    }


def _validate_statistic(statistic: str) -> None:
    if statistic == "absmax":
        return
    prefix = "percentile:"
    if statistic.startswith(prefix):
        try:
            percentile = float(statistic[len(prefix) :])
        except ValueError:
            percentile = -1.0
        if 0.0 < percentile <= 100.0:
            return
    raise ValueError(f"unsupported calibration statistic: {statistic}")


def merge_records(aggregate: dict[str, float], records: Mapping[str, float]) -> None:
    for name, value in records.items():
        value = float(value)
        if not _finite(value) or value < 0.0:
            raise ValueError(
                f"calibration returned invalid amax/statistic for {name}: {value}"
            )
        aggregate[name] = max(aggregate.get(name, 0.0), value)


def _finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
