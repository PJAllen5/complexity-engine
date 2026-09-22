"""Domain-aware configuration layer for scientific analysis pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DomainProfile:
    name: str
    description: str
    recommended_metrics: tuple[str, ...]
    signal_roles: dict[str, str] = field(default_factory=dict)
    interpretation_rules: dict[str, str] = field(default_factory=dict)
    expected_columns: tuple[str, ...] = field(default_factory=tuple)
    analysis_question: str = ""

    @classmethod
    def from_mapping(cls, data: dict[str, Any], key: str | None = None) -> "DomainProfile":
        normalized_name = (data.get("name") or key or "").strip().lower()
        recommended_metrics = tuple(data.get("recommended_metrics", ()))
        signal_roles = data.get("signal_roles", {})
        interpretation_rules = data.get("interpretation_rules", {})
        expected_columns = tuple(data.get("expected_columns", ()))
        analysis_question = data.get("analysis_question", "")
        return cls(
            name=normalized_name,
            description=str(data.get("description", "")),
            recommended_metrics=recommended_metrics,
            signal_roles={str(k): str(v) for k, v in dict(signal_roles).items()},
            interpretation_rules={str(k): str(v) for k, v in dict(interpretation_rules).items()},
            expected_columns=tuple(str(col) for col in expected_columns),
            analysis_question=str(analysis_question),
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
            "sample_entropy": "Higher sample entropy suggests greater temporal irregularity and reduced predictability.",
            "spectral_entropy": "Higher spectral entropy indicates broader frequency content and a more dispersed signal structure.",
            "dfa_exponent": "DFA exponent reflects long-range temporal correlation; values near 0.5 indicate more random-like behavior, while larger values suggest stronger persistence.",
        },
    )


def cognitive_profile() -> DomainProfile:
    return DomainProfile(
        name="cognitive",
        description="Cognitive and neural time-series analysis focused on signal organization, predictability, and dynamic adjustment to task demands.",
        recommended_metrics=("sample_entropy", "spectral_entropy", "dfa_exponent"),
        expected_columns=("signal", "condition", "trial"),
        analysis_question="Do cognitive or neural dynamics shift toward greater flexibility, more structure, or stronger temporal regularity under task demands?",
        signal_roles={
            "primary_signal": "Main cognitive or neural signal being analyzed.",
            "condition": "Condition, task state, or group labeling used for comparison.",
            "covariate": "Optional covariates such as trial, session, or participant metadata.",
        },
        interpretation_rules={
            "sample_entropy": "Higher sample entropy can indicate more flexible and less predictable cognitive dynamics.",
            "spectral_entropy": "Higher spectral entropy suggests broader engagement across frequency bands or less dominant structure.",
            "dfa_exponent": "DFA exponent helps characterize temporal dependence and long-range structure in task-related dynamics.",
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
        candidates = [
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


# Load any repository-level domain config if present at import time.
load_domain_profiles()
