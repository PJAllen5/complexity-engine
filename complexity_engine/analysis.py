"""Analysis workflow for time-series physiological and cognitive data."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .domain_profiles import DomainProfile, get_domain_profile, load_domain_profiles
from .metrics import compare_conditions, compute_complexity_metrics, descriptive_stats, mutual_information
from .plots import plot_complexity_metrics, plot_condition_comparison, plot_mutual_information_heatmap

_SIGNIFICANCE_ALPHA = 0.05

_SIGNIFICANCE_EXPLANATION = (
    "Welch's t-test checks whether the raw signal's mean level differs between two conditions "
    "beyond what sampling variability alone would produce; Cohen's d expresses the size of that "
    "difference in standard-deviation units (roughly: ~0.2 small, ~0.5 medium, ~0.8 large). This "
    "tests the raw signal, not the complexity metrics above, which are single summary values per "
    "condition rather than per-sample distributions."
)

_DEFAULT_METRIC_EXPLANATIONS = {
    "sample_entropy": "Sample entropy captures temporal irregularity; higher values suggest more unpredictability.",
    "spectral_entropy": "Spectral entropy captures the distribution of frequency content; higher values suggest broader frequency activity.",
    "dfa_exponent": "DFA exponent estimates long-range correlation structure; values near 0.5 often indicate random-like structure, while higher values suggest stronger persistence.",
}

_DEFAULT_CROSS_SIGNAL_EXPLANATION = (
    "Mutual information quantifies how much knowing one signal reduces uncertainty about the "
    "other; higher values indicate stronger coupling between signals, while values near zero "
    "suggest the signals vary independently at this analysis resolution."
)

_ENTROPY_COMPLEXITY_CAVEAT = (
    'Methodological note: entropy-based metrics quantify unpredictability, not complexity in the '
    'fuller sense used in complexity science (Crutchfield & Young, 1989; Gell-Mann & Lloyd, 1996). '
    'A maximally random signal has high entropy but little structured organization, so read the '
    'entropy changes below as changes in signal regularity, not as a direct verdict on how '
    '"complex" the underlying system is.'
)

_METADATA_COLUMNS = {
    "subject_id",
    "subject",
    "condition",
    "group",
    "timestamp",
    "time",
    "timepoint",
    "trial",
    "session",
    "label",
    "id",
    "participant",
}


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def resolve_signal_columns(df: pd.DataFrame, signal_cols: str | list[str] | None = None) -> list[str]:
    if signal_cols is not None:
        if isinstance(signal_cols, str):
            return [signal_cols]
        return list(signal_cols)

    candidate_columns = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col.lower() not in _METADATA_COLUMNS
    ]
    if not candidate_columns:
        raise ValueError("No numeric signal columns were found. Please specify signal_cols manually.")
    return candidate_columns


def summarize_condition(df: pd.DataFrame, signal_col: str, condition_col: str = "condition") -> pd.DataFrame:
    summary = (
        df.groupby(condition_col, dropna=False)[signal_col]
        .agg(["mean", "std", "min", "max", "median"])
        .reset_index()
    )
    return summary.rename(columns={"mean": "mean_signal", "std": "std_signal", "min": "min_signal", "max": "max_signal"})


def condition_complexity_summary(
    df: pd.DataFrame,
    signal_col: str,
    condition_col: str = "condition",
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    records = []
    for condition, group in df.groupby(condition_col, dropna=False):
        complexity = compute_complexity_metrics(group[signal_col].to_numpy(dtype=float), metrics=metrics)
        desc = descriptive_stats(group[signal_col].to_numpy(dtype=float))
        records.append(
            {
                "condition": condition,
                "signal": signal_col,
                "n_samples": int(group.shape[0]),
                **desc,
                **complexity,
            }
        )
    return pd.DataFrame.from_records(records)


def condition_significance_summary(
    df: pd.DataFrame,
    signal_col: str,
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Compare the raw signal between every pair of conditions with Welch's t-test + Cohen's d."""
    records = []
    conditions = sorted(df[condition_col].dropna().unique(), key=str)
    for condition_a, condition_b in itertools.combinations(conditions, 2):
        values_a = df.loc[df[condition_col] == condition_a, signal_col].to_numpy(dtype=float)
        values_b = df.loc[df[condition_col] == condition_b, signal_col].to_numpy(dtype=float)
        comparison = compare_conditions(values_a, values_b)
        records.append(
            {
                "signal": signal_col,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "n_a": values_a.size,
                "n_b": values_b.size,
                "mean_a": float(np.mean(values_a)) if values_a.size else np.nan,
                "mean_b": float(np.mean(values_b)) if values_b.size else np.nan,
                **comparison,
            }
        )
    return pd.DataFrame.from_records(records)


def condition_cross_signal_summary(
    df: pd.DataFrame,
    signal_cols: Sequence[str],
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Compute pairwise mutual information between signals, per condition.

    Requires at least two signal columns; returns an empty frame otherwise.
    """
    signal_cols = list(signal_cols)
    if len(signal_cols) < 2:
        return pd.DataFrame(columns=["condition", "signal_a", "signal_b", "mutual_information"])

    records = []
    for condition, group in df.groupby(condition_col, dropna=False):
        for signal_a, signal_b in itertools.combinations(signal_cols, 2):
            mi = mutual_information(
                group[signal_a].to_numpy(dtype=float),
                group[signal_b].to_numpy(dtype=float),
            )
            records.append(
                {
                    "condition": condition,
                    "signal_a": signal_a,
                    "signal_b": signal_b,
                    "mutual_information": mi,
                }
            )
    return pd.DataFrame.from_records(records)


def _missing_expected_columns(df: pd.DataFrame, domain_profile: DomainProfile) -> list[str]:
    return [col for col in domain_profile.expected_columns if col not in df.columns]


def _significance_lines_for_signal(signal: str, significance_df: pd.DataFrame | None) -> list[str]:
    if significance_df is None or significance_df.empty:
        return []
    signal_rows = significance_df[significance_df["signal"] == signal]
    lines = []
    for _, row in signal_rows.iterrows():
        p_value = row["p_value"]
        if np.isnan(p_value):
            continue
        condition_a, condition_b = row["condition_a"], row["condition_b"]
        cohens_d = row["cohens_d"]
        d_text = f"{cohens_d:.2f}" if not np.isnan(cohens_d) else "n/a"
        p_text = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
        verdict = "differs significantly" if p_value < _SIGNIFICANCE_ALPHA else "does not differ significantly"
        lines.append(
            f"- Raw signal {verdict} between {condition_a} and {condition_b} "
            f"(Welch's t-test {p_text}, Cohen's d={d_text})."
        )
    return lines


def _generate_interpretation_lines(
    complexity_df: pd.DataFrame,
    domain_profile: DomainProfile,
    cross_signal_df: pd.DataFrame | None = None,
    significance_df: pd.DataFrame | None = None,
) -> list[str]:
    lines = ["## Interpretation", ""]
    if domain_profile.analysis_question:
        lines.append(f"Guiding question: {domain_profile.analysis_question}")
        lines.append("")

    lines.append(_ENTROPY_COMPLEXITY_CAVEAT)
    lines.append("")

    if significance_df is not None and not significance_df.empty:
        lines.append(_SIGNIFICANCE_EXPLANATION)
        lines.append("")

    metrics_of_interest = list(domain_profile.recommended_metrics) or list(_DEFAULT_METRIC_EXPLANATIONS)
    for metric in metrics_of_interest:
        explanation = domain_profile.interpretation_rules.get(metric) or _DEFAULT_METRIC_EXPLANATIONS.get(metric)
        if explanation:
            lines.append(f"- {metric}: {explanation}")
    lines.append("")

    signals = sorted(complexity_df["signal"].unique()) if not complexity_df.empty else []
    multi_signal = len(signals) > 1
    for signal in signals:
        signal_rows = complexity_df[complexity_df["signal"] == signal].sort_values("condition")
        signal_lines = []
        if signal_rows.shape[0] >= 2:
            first = signal_rows.iloc[0]
            second = signal_rows.iloc[-1]
            for metric in metrics_of_interest:
                if metric not in signal_rows.columns:
                    continue
                first_val = float(first[metric])
                second_val = float(second[metric])
                if np.isnan(first_val) or np.isnan(second_val) or first_val == second_val:
                    continue
                direction = "higher" if second_val > first_val else "lower"
                rule = domain_profile.interpretation_rules.get(metric, "")
                metric_label = metric.replace("_", " ")
                signal_lines.append(
                    f"- {metric_label} is {direction} in {second['condition']} than in {first['condition']}. {rule}".rstrip()
                )
        signal_lines.extend(_significance_lines_for_signal(signal, significance_df))
        if not signal_lines:
            continue
        if multi_signal:
            lines.append(f"### {signal}")
        lines.extend(signal_lines)
        lines.append("")

    if cross_signal_df is not None and not cross_signal_df.empty:
        lines.append("### Cross-signal relationships")
        mi_explanation = domain_profile.interpretation_rules.get("mutual_information") or _DEFAULT_CROSS_SIGNAL_EXPLANATION
        lines.append(f"- mutual_information: {mi_explanation}")
        for (signal_a, signal_b), pair_rows in cross_signal_df.groupby(["signal_a", "signal_b"]):
            pair_rows = pair_rows.sort_values("condition")
            if pair_rows.shape[0] < 2:
                continue
            first = pair_rows.iloc[0]
            second = pair_rows.iloc[-1]
            first_val = float(first["mutual_information"])
            second_val = float(second["mutual_information"])
            if first_val == second_val:
                continue
            direction = "higher" if second_val > first_val else "lower"
            lines.append(
                f"- Mutual information between {signal_a} and {signal_b} is {direction} in "
                f"{second['condition']} than in {first['condition']}."
            )
        lines.append("")

    return lines


def analyze_dataset(
    data_path: str | Path,
    signal_col: str | None = None,
    signal_cols: str | list[str] | None = None,
    condition_col: str = "condition",
    output_dir: str | Path = "outputs",
    domain: str | None = None,
    domain_config: str | Path | None = None,
) -> dict[str, pd.DataFrame | str]:
    load_domain_profiles(domain_config)
    domain_profile = get_domain_profile(domain)
    df = load_dataset(data_path)
    selected_signals = resolve_signal_columns(df, signal_cols=signal_cols if signal_cols is not None else signal_col)
    missing_columns = _missing_expected_columns(df, domain_profile)

    summary_frames = []
    complexity_frames = []
    significance_frames = []
    for signal in selected_signals:
        if signal not in df.columns:
            raise ValueError(f"Signal column '{signal}' was not found in the dataset.")
        summary_frames.append(summarize_condition(df, signal_col=signal, condition_col=condition_col).assign(signal=signal))
        complexity_frames.append(
            condition_complexity_summary(
                df,
                signal_col=signal,
                condition_col=condition_col,
                metrics=domain_profile.recommended_metrics or None,
            )
        )
        significance_frames.append(condition_significance_summary(df, signal_col=signal, condition_col=condition_col))

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    complexity = pd.concat(complexity_frames, ignore_index=True) if complexity_frames else pd.DataFrame()
    significance = pd.concat(significance_frames, ignore_index=True) if significance_frames else pd.DataFrame()
    cross_signal = condition_cross_signal_summary(df, selected_signals, condition_col=condition_col)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "condition_summary.csv"
    complexity_path = output_dir / "complexity_summary.csv"
    significance_path = output_dir / "significance_summary.csv"
    cross_signal_path = output_dir / "cross_signal_summary.csv"
    summary.to_csv(summary_path, index=False)
    complexity.to_csv(complexity_path, index=False)
    significance.to_csv(significance_path, index=False)
    cross_signal.to_csv(cross_signal_path, index=False)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}
    for signal in selected_signals:
        comparison_path = plot_condition_comparison(df, signal, condition_col, plots_dir / f"{signal}_condition_comparison.png")
        plot_paths[f"{signal}_condition_comparison"] = comparison_path
        metrics_plot_path = plot_complexity_metrics(
            complexity, signal, domain_profile.recommended_metrics or list(_DEFAULT_METRIC_EXPLANATIONS), plots_dir / f"{signal}_complexity_metrics.png"
        )
        if metrics_plot_path:
            plot_paths[f"{signal}_complexity_metrics"] = metrics_plot_path
    if not cross_signal.empty:
        heatmap_path = plot_mutual_information_heatmap(cross_signal, plots_dir / "cross_signal_mutual_information.png")
        if heatmap_path:
            plot_paths["cross_signal_mutual_information"] = heatmap_path

    report_path = output_dir / "report.md"
    report_lines = [
        "# Complexity Analysis Report",
        "",
        f"- Domain: {domain_profile.name}",
        f"- Domain description: {domain_profile.description}",
        f"- Analysis question: {domain_profile.analysis_question}" if domain_profile.analysis_question else "",
        f"- Data source: {Path(data_path).name}",
        f"- Signal columns: {', '.join(selected_signals)}",
        f"- Condition column: {condition_col}",
        "",
    ]
    if domain_profile.signal_roles:
        report_lines.append("## Domain signal roles")
        for role, description in domain_profile.signal_roles.items():
            report_lines.append(f"- {role}: {description}")
        report_lines.append("")
    if missing_columns:
        report_lines.append(
            f"Note: the '{domain_profile.name}' domain profile expects columns "
            f"[{', '.join(domain_profile.expected_columns)}]; missing from this dataset: {', '.join(missing_columns)}."
        )
        report_lines.append("")
    report_lines.extend(
        [
            "## Condition summary",
            summary.to_string(index=False),
            "",
            "## Complexity metrics",
            complexity.to_string(index=False),
            "",
        ]
    )
    if not significance.empty:
        report_lines.extend(
            [
                "## Statistical significance",
                significance.to_string(index=False),
                "",
            ]
        )
    if not cross_signal.empty:
        report_lines.extend(
            [
                "## Cross-signal metrics",
                cross_signal.to_string(index=False),
                "",
            ]
        )
    if plot_paths:
        report_lines.append("## Visualizations")
        for label, path in plot_paths.items():
            relative_path = Path(path).relative_to(output_dir)
            report_lines.append(f"![{label}]({relative_path.as_posix()})")
        report_lines.append("")
    report_lines.extend(
        _generate_interpretation_lines(complexity, domain_profile, cross_signal_df=cross_signal, significance_df=significance)
    )
    report_lines = [line for line in report_lines if line]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "data": df,
        "selected_signals": selected_signals,
        "domain_profile": domain_profile.as_dict(),
        "missing_columns": missing_columns,
        "condition_summary": summary,
        "complexity_summary": complexity,
        "significance_summary": significance,
        "cross_signal_summary": cross_signal,
        "summary_csv": str(summary_path),
        "complexity_csv": str(complexity_path),
        "significance_csv": str(significance_path),
        "cross_signal_csv": str(cross_signal_path),
        "plot_paths": plot_paths,
        "report_md": str(report_path),
    }
