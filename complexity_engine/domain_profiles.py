"""Domain-aware configuration layer for scientific analysis pipelines."""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import SURROGATE_TYPES


# How time-series metrics treat gaps in a recording (see docs/design/time-axis-and-node-attributes.md):
# join them (consecutive samples), compute within gap-free segments, or exclude low-coverage units.
GAP_POLICIES = ("join", "segment", "exclude")

# What metric scales count: samples (the default; e.g. heartbeats for HRV), or time, where
# scales may be durations and irregularly sampled units aren't computed.
AXES = ("samples", "time")

# Metric parameters that are window sizes, and so may be given as durations under axis: time.
DURATION_PARAMETERS = ("min_scale", "max_scale")

# How a node attribute is compared across ties (see network_metrics.attribute_mixing).
ATTRIBUTE_KINDS = ("categorical", "numeric")

# Surrogate baselines (see metrics.surrogate_test): defaults for a profile's `surrogates`.
SURROGATE_DEFAULTS = {"type": "shuffle", "count": 200, "seed": 0}

_DURATION_UNITS = {"s": 1, "min": 60, "h": 3_600, "D": 86_400, "W": 604_800}
_DURATION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(s|min|h|D|W)\s*$")


def parse_duration(value: Any) -> float | None:
    """Seconds in a duration such as "30D", "12h", "15min", "2W" or "0.5s", or None if
    value isn't a duration string. Units are case-sensitive, matching the sampling report
    ("D" is days; "min" is minutes, so "m" is never ambiguous with months)."""
    if not isinstance(value, str):
        return None
    match = _DURATION_PATTERN.match(value)
    if match is None:
        return None
    return float(match.group(1)) * _DURATION_UNITS[match.group(2)]


@dataclass
class DomainProfile:
    name: str
    description: str
    recommended_metrics: tuple[str, ...]
    signal_roles: dict[str, str] = field(default_factory=dict)
    interpretation_rules: dict[str, str] = field(default_factory=dict)
    expected_columns: tuple[str, ...] = field(default_factory=tuple)
    analysis_question: str = ""
    metric_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    gap_policy: str = "join"
    min_coverage: float = 0.8
    max_joined_gap: int = 0
    axis: str = "samples"
    units: dict[str, str] = field(default_factory=dict)
    surrogates: dict[str, Any] | None = None
    node_attributes: dict[str, dict[str, str]] = field(default_factory=dict)

    def has_duration_parameters(self) -> bool:
        return any(isinstance(v, str) for params in self.metric_parameters.values() for v in params.values())

    @classmethod
    def from_mapping(cls, data: dict[str, Any], key: str | None = None) -> "DomainProfile":
        normalized_name = (data.get("name") or key or "").strip().lower()
        recommended_metrics = tuple(data.get("recommended_metrics", ()))
        signal_roles = data.get("signal_roles", {})
        interpretation_rules = data.get("interpretation_rules", {})
        expected_columns = tuple(data.get("expected_columns", ()))
        analysis_question = data.get("analysis_question", "")
        metric_parameters = data.get("metric_parameters") or {}
        if not isinstance(metric_parameters, dict) or not all(isinstance(v, dict) for v in metric_parameters.values()):
            raise ValueError(
                f"Domain profile '{normalized_name}': metric_parameters must map each metric name to a "
                "mapping of parameter names to values, e.g. {dfa_exponent: {min_scale: 30}}."
            )
        gap_policy = str(data.get("gap_policy", "join")).strip().lower()
        if gap_policy not in GAP_POLICIES:
            raise ValueError(
                f"Domain profile '{normalized_name}': gap_policy must be one of {', '.join(GAP_POLICIES)}, "
                f"got '{gap_policy}'."
            )
        min_coverage = float(data.get("min_coverage", 0.8))
        if not 0.0 < min_coverage <= 1.0:
            raise ValueError(f"Domain profile '{normalized_name}': min_coverage must be in (0, 1], got {min_coverage}.")
        max_joined_gap = data.get("max_joined_gap", 0)
        if not isinstance(max_joined_gap, int) or isinstance(max_joined_gap, bool) or max_joined_gap < 0:
            raise ValueError(
                f"Domain profile '{normalized_name}': max_joined_gap must be a non-negative whole number of "
                f"missing samples, got {max_joined_gap!r}."
            )
        axis = str(data.get("axis", "samples")).strip().lower()
        if axis not in AXES:
            raise ValueError(
                f"Domain profile '{normalized_name}': axis must be one of {', '.join(AXES)}, got '{axis}'."
            )
        for metric, parameters in metric_parameters.items():
            for parameter, value in parameters.items():
                if not isinstance(value, str):
                    continue
                where = f"Domain profile '{normalized_name}': metric_parameters {metric}.{parameter}"
                if parameter not in DURATION_PARAMETERS:
                    raise ValueError(
                        f"{where} is '{value}', but only {', '.join(DURATION_PARAMETERS)} may be given as text "
                        "(a duration)."
                    )
                seconds = parse_duration(value)
                if seconds is None or seconds <= 0:
                    raise ValueError(
                        f"{where} is '{value}', which isn't a positive duration; use a number and one of "
                        f"{', '.join(_DURATION_UNITS)}, e.g. 30D or 12h."
                    )
                if axis != "time":
                    raise ValueError(
                        f"{where} is the duration '{value}', but the profile's axis is '{axis}', where scales "
                        "count samples. Set axis: time, or give the scale as a whole number of samples."
                    )
        units = data.get("units") or {}
        if not isinstance(units, dict) or not all(isinstance(v, str) and v.strip() for v in units.values()):
            raise ValueError(
                f"Domain profile '{normalized_name}': units must map column names to unit labels, "
                "e.g. {tmax_anomaly: degC}."
            )
        surrogates = data.get("surrogates")
        if surrogates is not None:
            if surrogates is True:
                surrogates = {}
            if not isinstance(surrogates, dict) or set(surrogates) - set(SURROGATE_DEFAULTS):
                raise ValueError(
                    f"Domain profile '{normalized_name}': surrogates must be a mapping with any of "
                    f"{', '.join(SURROGATE_DEFAULTS)}, e.g. {{type: shuffle, count: 200}}."
                )
            surrogates = {**SURROGATE_DEFAULTS, **surrogates}
            surrogates["type"] = str(surrogates["type"]).strip().lower()
            if surrogates["type"] not in SURROGATE_TYPES:
                raise ValueError(
                    f"Domain profile '{normalized_name}': surrogates type must be one of "
                    f"{', '.join(SURROGATE_TYPES)}, got '{surrogates['type']}'."
                )
            for key, minimum in (("count", 19), ("seed", 0)):
                value = surrogates[key]
                if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                    raise ValueError(
                        f"Domain profile '{normalized_name}': surrogates {key} must be a whole number of at "
                        f"least {minimum}, got {value!r}"
                        + (" (with fewer than 19 surrogates, p can never reach 0.05)." if key == "count" else ".")
                    )
        node_attributes = {}
        raw_attributes = data.get("node_attributes") or {}
        if not isinstance(raw_attributes, dict):
            raise ValueError(
                f"Domain profile '{normalized_name}': node_attributes must map attribute names to a role "
                "description, or to {role: ..., kind: categorical | numeric}."
            )
        for attribute, spec in raw_attributes.items():
            spec = {"role": spec} if isinstance(spec, str) else spec
            if not isinstance(spec, dict) or set(spec) - {"role", "kind"}:
                raise ValueError(
                    f"Domain profile '{normalized_name}': node_attributes.{attribute} must be a role description "
                    "or a mapping with role and kind."
                )
            kind = str(spec.get("kind", "")).strip().lower()
            if kind and kind not in ATTRIBUTE_KINDS:
                raise ValueError(
                    f"Domain profile '{normalized_name}': node_attributes.{attribute} kind must be one of "
                    f"{', '.join(ATTRIBUTE_KINDS)}, got '{kind}'."
                )
            node_attributes[str(attribute)] = {"role": str(spec.get("role", "")), "kind": kind}
        return cls(
            name=normalized_name,
            description=str(data.get("description", "")),
            recommended_metrics=recommended_metrics,
            signal_roles={str(k): str(v) for k, v in dict(signal_roles).items()},
            interpretation_rules={str(k): str(v) for k, v in dict(interpretation_rules).items()},
            expected_columns=tuple(str(col) for col in expected_columns),
            analysis_question=str(analysis_question),
            metric_parameters={str(k): {str(p): v for p, v in v.items()} for k, v in metric_parameters.items()},
            gap_policy=gap_policy,
            min_coverage=min_coverage,
            max_joined_gap=max_joined_gap,
            axis=axis,
            units={str(k): v.strip() for k, v in units.items()},
            surrogates=surrogates,
            node_attributes=node_attributes,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "recommended_metrics": list(self.recommended_metrics),
            "expected_columns": list(self.expected_columns),
            "analysis_question": self.analysis_question,
            "signal_roles": dict(self.signal_roles),
            "interpretation_rules": dict(self.interpretation_rules),
            "metric_parameters": {k: dict(v) for k, v in self.metric_parameters.items()},
            "gap_policy": self.gap_policy,
            "min_coverage": self.min_coverage,
            "max_joined_gap": self.max_joined_gap,
            "axis": self.axis,
            "units": dict(self.units),
            "surrogates": dict(self.surrogates) if self.surrogates else None,
            "node_attributes": {k: dict(v) for k, v in self.node_attributes.items()},
        }


def physiology_profile() -> DomainProfile:
    return DomainProfile(
        name="physiology",
        description="Physiological and behavioral time-series analysis focused on variability, stress response, and temporal structure.",
        recommended_metrics=("sample_entropy", "spectral_entropy", "dfa_exponent"),
        expected_columns=("signal", "condition", "subject_id"),
        analysis_question="How do changes in physiological signal dynamics relate to condition-specific stress, arousal, or variability?",
        signal_roles={
            "primary_signal": "Main physiological signal being assessed.",
            "condition": "Experimental or task condition used for comparison.",
            "covariate": "Optional metadata such as age, group, session, or clinical variables.",
        },
        interpretation_rules={
            "sample_entropy": (
                'Higher sample entropy indicates reduced short-term predictability in the signal. '
                'Physiological regulation is classically understood as a cybernetic feedback process '
                '(Wiener, 1948; Ashby, 1956); a healthy regulatory system maintains a flexible operating '
                'range, so both unusually low entropy (rigid, overly constrained dynamics) and unusually '
                'high entropy (loss of regulatory control) can indicate impaired adaptability rather than '
                'simply "more" or "less" complexity.'
            ),
            "spectral_entropy": (
                'A more evenly distributed frequency spectrum means the signal is not dominated by one '
                'oscillatory rhythm. In feedback-control terms (Ashby, 1956), effective regulation '
                'typically distributes activity across several feedback timescales rather than a single '
                'dominant one, so a narrowing spectrum can reflect a collapse toward fewer active '
                'regulatory pathways.'
            ),
            "dfa_exponent": (
                'The DFA exponent estimates long-range temporal correlation. Values that depart from the '
                'uncorrelated random-walk baseline (~0.5) suggest history-dependent, self-organized '
                'fluctuation patterns rather than memoryless noise (Bak, 1996); interpret shifts relative '
                "to a subject's own baseline rather than the absolute value alone, since scale-free "
                'fluctuation patterns are common in healthy physiological systems.'
            ),
            "mutual_information": (
                'Mutual information between physiological signals reflects how tightly two subsystems are '
                'coupled through shared regulatory feedback (Ashby, 1956; Wiener, 1948); rising coupling '
                'under a condition can indicate coordinated regulation, while falling coupling can indicate '
                'the subsystems regulating more independently.'
            ),
        },
    )


def cognitive_profile() -> DomainProfile:
    return DomainProfile(
        name="cognitive",
        description="Cognitive and neural time-series analysis focused on signal organization, predictability, and dynamic adjustment to task demands.",
        recommended_metrics=("sample_entropy", "spectral_entropy", "dfa_exponent"),
        expected_columns=("signal", "condition", "trial"),
        analysis_question="Do cognitive or neural dynamics shift toward tighter predictive control, more distributed processing, or stronger temporal structure under task demands?",
        signal_roles={
            "primary_signal": "Main cognitive or neural signal being analyzed.",
            "condition": "Condition, task state, or group labeling used for comparison.",
            "covariate": "Optional covariates such as trial, session, or participant metadata.",
        },
        interpretation_rules={
            "sample_entropy": (
                'Under a predictive-processing account, a well-tuned cognitive system minimizes surprise '
                'through precise, expected dynamics (Friston, 2010; Clark, 2013); falling entropy under '
                'task engagement can reflect tighter predictive control rather than "less complexity," '
                'while rising entropy can reflect exploration, uncertainty, or reduced model precision.'
            ),
            "spectral_entropy": (
                'Power distributed across more frequencies suggests multiple processing rhythms are '
                'active in parallel, consistent with connectionist accounts of distributed processing '
                '(Rumelhart et al., 1986); read this as evidence about how many concurrent processes are '
                'engaged, not as a direct measure of cognitive "complexity."'
            ),
            "dfa_exponent": (
                'Long-range correlation in cognitive or neural signals reflects memory-dependent dynamics '
                'rather than momentary reactivity (Hopfield, 1982; Rumelhart et al., 1986); a shift toward '
                'stronger persistence is more consistent with the system settling into a constrained, '
                'attractor-like processing mode than with becoming more "flexible."'
            ),
            "mutual_information": (
                'Mutual information across cognitive or neural signals indexes functional integration '
                'between distributed processing streams (Rumelhart et al., 1986); increasing coupling '
                'under task demands is consistent with binding separate processes into a coordinated '
                'response, while decreasing coupling suggests more independent, modular processing.'
            ),
        },
    )


_DOMAIN_REGISTRY: dict[str, DomainProfile] = {
    "physiology": physiology_profile(),
    "cognitive": cognitive_profile(),
}


def register_domain_profile(profile: DomainProfile) -> DomainProfile:
    _DOMAIN_REGISTRY[profile.name.lower().strip()] = profile
    return profile


def load_domain_profiles(config_path: str | Path | None = None) -> dict[str, DomainProfile]:
    if config_path is None:
        here = Path(__file__).resolve().parent
        candidates = [
            # an installed copy carries config/ inside the package (see pyproject.toml)
            here / "config" / "domain_profiles.yaml",
            here / "config" / "domain_profiles.yml",
            here / "config" / "domain_profiles.json",
            # a checkout of the repository: config/ at the top level
            Path(__file__).resolve().parents[1] / "config" / "domain_profiles.yaml",
            Path(__file__).resolve().parents[1] / "config" / "domain_profiles.yml",
            Path(__file__).resolve().parents[1] / "config" / "domain_profiles.json",
            Path(__file__).resolve().parents[1] / "domain_profiles.yaml",
            Path(__file__).resolve().parents[1] / "domain_profiles.yml",
            Path(__file__).resolve().parents[1] / "domain_profiles.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break
    if config_path is None:
        return dict(_DOMAIN_REGISTRY)

    path = Path(config_path)
    raw_text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(raw_text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise ImportError("YAML domain configuration requires PyYAML to be installed.") from exc
        payload = yaml.safe_load(raw_text) or {}
    else:
        raise ValueError(f"Unsupported domain config format: {suffix}. Use .json, .yaml, or .yml")

    if not isinstance(payload, dict):
        raise ValueError("Domain profile configuration must be a dictionary keyed by domain name.")

    loaded_profiles: dict[str, DomainProfile] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"Domain profile for '{key}' must be a mapping.")
        profile = DomainProfile.from_mapping(value, key=key)
        loaded_profiles[profile.name.lower().strip()] = profile

    _DOMAIN_REGISTRY.update(loaded_profiles)
    return dict(_DOMAIN_REGISTRY)


def get_domain_profile(domain: str | None = None) -> DomainProfile:
    if domain is None:
        return _DOMAIN_REGISTRY.get("physiology", physiology_profile())

    normalized = domain.lower().strip()
    if normalized not in _DOMAIN_REGISTRY:
        supported = ", ".join(sorted(_DOMAIN_REGISTRY))
        raise ValueError(f"Unknown domain: {domain}. Supported domains: {supported}.")
    return _DOMAIN_REGISTRY[normalized]


def validate_recommended_metrics(profile: DomainProfile, known_metrics: Collection[str]) -> None:
    """Raise a clear error if a profile's recommended_metrics names a metric that
    doesn't exist in the given pipeline's metric registry.

    Config mistakes (a typo, or a metric that's valid for the other pipeline) would
    otherwise only surface deep inside metric computation -- or, for the network
    pipeline, not at all, since it doesn't gate computation on recommended_metrics
    the way the time-series pipeline does, so an unknown name there is silently
    dropped from the interpretation section rather than erroring. Domain profiles
    are meant to be editable without touching Python, so this failure should be as
    early and as clear as possible: called right after a domain profile is looked
    up, before any (potentially expensive) data loading happens.
    """
    unknown = [metric for metric in profile.recommended_metrics if metric not in known_metrics]
    if unknown:
        supported = ", ".join(sorted(known_metrics))
        raise ValueError(
            f"Domain profile '{profile.name}' lists unknown recommended_metrics {unknown}; "
            f"supported metrics are: {supported}."
        )


def validate_metric_parameters(profile: DomainProfile, accepted_parameters: Mapping[str, Collection[str]]) -> None:
    """Raise a clear error if a profile's metric_parameters names a metric the pipeline
    doesn't have, a metric the profile doesn't compute, or a parameter the metric doesn't
    accept. Without this, a misspelled parameter would be silently ignored, and the run would
    use the default setting while the profile appeared to configure it.
    """
    computed = set(profile.recommended_metrics)
    for metric, parameters in profile.metric_parameters.items():
        if metric not in accepted_parameters:
            supported = ", ".join(sorted(accepted_parameters)) or "none"
            raise ValueError(
                f"Domain profile '{profile.name}' sets metric_parameters for unknown metric '{metric}'; "
                f"metrics that take parameters here: {supported}."
            )
        if metric not in computed:
            raise ValueError(
                f"Domain profile '{profile.name}' sets metric_parameters for '{metric}', which isn't in its "
                "recommended_metrics, so the parameters would have no effect."
            )
        unknown = sorted(set(parameters) - set(accepted_parameters[metric]))
        if unknown:
            raise ValueError(
                f"Domain profile '{profile.name}' sets unknown parameter(s) {unknown} for '{metric}'; "
                f"it accepts: {', '.join(accepted_parameters[metric])}."
            )


# Load any repository-level domain config if present at import time.
load_domain_profiles()
