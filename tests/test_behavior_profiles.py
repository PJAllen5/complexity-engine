"""The behavior profiles run end to end on data shaped like `behavetrack export-complexity` output.

The files built here mirror that export: column names, one row per frame (time series) or
per pair (networks), with `session` identifying each video or movie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from complexity_engine.analysis import analyze_dataset
from complexity_engine.domain_profiles import get_domain_profile, load_domain_profiles
from complexity_engine.network_analysis import analyze_network

BEHAVIOR_PROFILES = ["collective_behavior", "neural_activity", "animal_social_network", "neuronal_network"]


@pytest.mark.parametrize("name", BEHAVIOR_PROFILES)
def test_behavior_profiles_load_with_question_and_rules(name):
    load_domain_profiles()
    profile = get_domain_profile(name)
    assert profile.analysis_question
    # every recommended metric has interpretation text
    assert set(profile.recommended_metrics) <= set(profile.interpretation_rules)


def _collective_export(rng) -> pd.DataFrame:
    """Two videos per condition: 'schooling' holds long aligned bouts, 'swarming' is noisy."""
    rows = []
    for condition, persistence in (("schooling", 0.98), ("swarming", 0.2)):
        for v in range(2):
            p = 0.5
            for f in range(400):
                p = float(np.clip(persistence * p + (1 - persistence) * 0.5 + rng.normal(0, 0.05), 0, 1))
                rows.append({"time": round(f / 30, 4), "session": f"{condition}_{v}", "condition": condition,
                             "polarization": p, "rotation": float(rng.uniform(0, 0.3)),
                             "mean_speed_bl_s": 2 + 3 * p + float(rng.normal(0, 0.2)), "n_visible": 40})
    return pd.DataFrame(rows)


def test_collective_behavior_on_exported_frames(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "collective_timeseries.csv"
    _collective_export(rng).to_csv(path, index=False)
    result = analyze_dataset(path, signal_cols=["polarization", "mean_speed_bl_s"], condition_col="condition",
                             domain="collective_behavior", unit_cols=["session"], output_dir=tmp_path / "out")
    assert result["missing_columns"] == []
    summary = result["complexity_summary"] if "complexity_summary" in result else pd.read_csv(result["complexity_csv"])
    pol = summary[summary["signal"] == "polarization"].set_index("condition")
    # long aligned bouts are more persistent than noise
    assert pol.loc["schooling", "dfa_exponent"] > pol.loc["swarming", "dfa_exponent"]
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Tunstrøm et al., 2013" in report


def test_neural_activity_on_exported_traces(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for condition in ("control", "drug"):
        for n in range(3):
            dff = np.convolve(rng.random(600) < 0.03, np.exp(-np.arange(20) / 6), mode="full")[:600]
            dff = dff + rng.normal(0, 0.02, 600)
            rows += [{"time": round(f / 20, 3), "session": f"{condition}_movie", "neuron_id": n + 1,
                      "condition": condition, "dff": float(v)} for f, v in enumerate(dff)]
    path = tmp_path / "neural_timeseries.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    result = analyze_dataset(path, signal_cols=["dff"], condition_col="condition", domain="neural_activity",
                             unit_cols=["session", "neuron_id"], output_dir=tmp_path / "out")
    assert result["missing_columns"] == []
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "min_scale" in report  # the profile's DFA window is applied and reported


@pytest.mark.parametrize("domain", ["animal_social_network", "neuronal_network"])
def test_network_profiles_on_exported_edges(domain, tmp_path):
    rng = np.random.default_rng(2)
    edges = []
    for condition in ("a", "b"):
        for i in range(1, 13):
            for j in range(i + 1, 13):
                same_group = (i <= 6) == (j <= 6)
                if rng.random() < (0.8 if same_group else 0.1):
                    edges.append({"source": i, "target": j, "weight": round(float(rng.uniform(0.3, 1)), 3),
                                  "condition": condition})
    path = tmp_path / "network.csv"
    pd.DataFrame(edges).to_csv(path, index=False)
    result = analyze_network(path, domain=domain, output_dir=tmp_path / "out")
    assert result["missing_columns"] == []
    metrics = result["network_metrics_summary"]
    assert set(metrics["condition"]) == {"a", "b"}
    assert (metrics["modularity"] > 0.2).all()  # two planted groups
