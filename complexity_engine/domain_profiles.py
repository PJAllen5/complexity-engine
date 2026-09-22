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
