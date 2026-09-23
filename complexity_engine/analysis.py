"""Analysis workflow for time-series physiological and cognitive data."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .domain_profiles import DomainProfile, get_domain_profile, load_domain_profiles, validate_recommended_metrics
from .metrics import _METRIC_FUNCTIONS, compare_conditions, compute_complexity_metrics, descriptive_stats, mutual_information
from .plots import plot_complexity_metrics, plot_condition_comparison, plot_mutual_information_heatmap
from .report_export import export_html_report

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

_UNDEFINED_METRIC_REASONS = {
    "sample_entropy": "needs at least 5 samples and at least one pair of matching templates at lengths 2 and 3",
    "spectral_entropy": "needs at least 2 samples and a non-constant signal",
    "dfa_exponent": "needs at least 16 samples and non-zero fluctuation at every scale",
}

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
    "source_file",
}

# Columns that identify one independent recording; all present ones are used together,
# since trials and sessions are typically nested within a subject.
_UNIT_COLUMN_CANDIDATES = ("subject_id", "subject", "participant", "session", "trial")

_TIME_COLUMN_CANDIDATES = ("timestamp", "time", "timepoint")


def _load_single_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def load_dataset(data_path: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    """Load one file, or merge several files from the same multi-session/multi-subject study.

    A single path keeps the original single-file schema exactly (no added columns).
    Multiple paths are concatenated and must share the same columns -- a mismatch is a
    strong signal of a mismatched file being merged by mistake, so it's a hard error
    rather than a silent NaN-filled merge.
    """
    if isinstance(data_path, (str, Path)):
        return _load_single_file(Path(data_path))

    paths = [Path(p) for p in data_path]
    if not paths:
        raise ValueError("No data files were provided.")
    frames = [_load_single_file(path) for path in paths]

    column_sets = [frozenset(frame.columns) for frame in frames]
    if len(set(column_sets)) > 1:
        details = "; ".join(f"{path.name}: {sorted(cols)}" for path, cols in zip(paths, column_sets))
        raise ValueError(f"Cannot merge files with different columns: {details}")

    for path, frame in zip(paths, frames):
        frame["source_file"] = path.name
    return pd.concat(frames, ignore_index=True)


def _describe_data_source(data_path: str | Path | Sequence[str | Path]) -> str:
    if isinstance(data_path, (str, Path)):
        return Path(data_path).name
    return ", ".join(Path(p).name for p in data_path)


def _as_code_block(text: str) -> list[str]:
    """Wrap fixed-width text (e.g. a DataFrame.to_string() dump) in a fenced code
    block, so it renders as a monospace, whitespace-preserved block in Markdown/HTML
    instead of collapsing into an unreadable paragraph.
    """
    return ["```", text, "```", ""]


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    """Keep at most one blank line in a row, and drop leading/trailing blanks.

    A single blank line between blocks is meaningful in Markdown (it's what
    separates a list from unrelated following text -- without it, a paragraph
    right after a list item gets parsed as a "lazy continuation" of that list
    item instead of its own block), so report generation shouldn't strip every
    blank line the way earlier versions did.
    """
    collapsed: list[str] = []
    for line in lines:
        if line == "" and (not collapsed or collapsed[-1] == ""):
            continue
        collapsed.append(line)
    while collapsed and collapsed[0] == "":
        collapsed.pop(0)
    while collapsed and collapsed[-1] == "":
        collapsed.pop()
    return collapsed


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


def clean_dataset(df: pd.DataFrame, required_cols: Sequence[str]) -> tuple[pd.DataFrame, dict]:
    """Drop rows missing a value in any required column; detect (but don't drop) exact
    duplicate rows.

    Duplicates are only flagged, not removed: without a timestamp or other unique key,
    two different observations can legitimately share the same signal value, especially
    in time series with limited measurement precision, so silently deleting "duplicate"
    rows risks deleting real data rather than errors. Missing-value rows are safe to drop
    automatically since a NaN signal value breaks the downstream metrics outright.

    Returns the cleaned frame alongside a provenance record of what was found/dropped and
    why, so cleaning is never silent -- see analyze_dataset()'s "Data provenance" report
    section.
    """
    provenance: dict = {"input_rows": int(df.shape[0])}

    provenance["duplicate_rows_detected"] = int(df.duplicated().sum())

    checked_cols = [col for col in required_cols if col in df.columns]
    if checked_cols:
        missing_mask = df[checked_cols].isna().any(axis=1)
    else:
        missing_mask = pd.Series(False, index=df.index)
    provenance["rows_dropped_missing_values"] = int(missing_mask.sum())
    provenance["missing_value_columns_checked"] = checked_cols
    df = df.loc[~missing_mask]

    provenance["output_rows"] = int(df.shape[0])
    return df.reset_index(drop=True), provenance


def summarize_condition(df: pd.DataFrame, signal_col: str, condition_col: str = "condition") -> pd.DataFrame:
    summary = (
        df.groupby(condition_col, dropna=False)[signal_col]
        .agg(["mean", "std", "min", "max", "median"])
        .reset_index()
    )
    return summary.rename(columns={"mean": "mean_signal", "std": "std_signal", "min": "min_signal", "max": "max_signal"})


def resolve_unit_columns(
    df: pd.DataFrame,
    unit_cols: str | Sequence[str] | None = None,
    condition_col: str = "condition",
) -> list[str]:
    """Return the columns that identify one independent recording (subject, trial, ...).

    None auto-detects any of _UNIT_COLUMN_CANDIDATES present in the data (matched
    case-insensitively); an explicit empty list disables per-unit computation, treating
    each condition as a single recording.
    """
    if unit_cols is None:
        by_lower = {str(col).lower(): col for col in df.columns}
        resolved = [by_lower[name] for name in _UNIT_COLUMN_CANDIDATES if name in by_lower]
    else:
        resolved = [unit_cols] if isinstance(unit_cols, str) else list(unit_cols)
        missing = [col for col in resolved if col not in df.columns]
        if missing:
            raise ValueError(f"Unit column(s) not found in the dataset: {', '.join(missing)}")
    return [col for col in resolved if col != condition_col]


def resolve_time_column(df: pd.DataFrame) -> str | None:
    by_lower = {str(col).lower(): col for col in df.columns}
    return next((by_lower[name] for name in _TIME_COLUMN_CANDIDATES if name in by_lower), None)


def _time_sort_key(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return values
    try:
        return pd.to_datetime(values)
    except (ValueError, TypeError):
        return values.astype(str)


def unit_complexity_summary(
    df: pd.DataFrame,
    signal_col: str,
    condition_col: str = "condition",
    metrics: Sequence[str] | None = None,
    unit_cols: Sequence[str] = (),
    time_col: str | None = None,
) -> pd.DataFrame:
    """Compute complexity metrics separately within each recording unit of each condition.

    Sample entropy, spectral entropy, and DFA depend on sample order, so joining several
    independent recordings (subjects, trials, sessions) end to end would insert an
    artificial jump at every join that the metrics read as real dynamics. Rows within a
    unit are sorted by time_col when given; otherwise file order is kept.
    """
    group_cols = [condition_col, *unit_cols]
    records = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        if time_col is not None:
            group = group.sort_values(time_col, key=_time_sort_key, kind="stable")
        values = group[signal_col].to_numpy(dtype=float)
        records.append(
            {
                "condition": keys[0],
                **dict(zip(unit_cols, keys[1:])),
                "signal": signal_col,
                "n_samples": int(values.size),
                **compute_complexity_metrics(values, metrics=metrics),
            }
        )
    return pd.DataFrame.from_records(records)


def condition_complexity_summary(
    df: pd.DataFrame,
    signal_col: str,
    condition_col: str = "condition",
    metrics: Sequence[str] | None = None,
    unit_cols: Sequence[str] = (),
    time_col: str | None = None,
    unit_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-condition complexity summary: descriptive stats pooled over every sample in the
    condition, and each complexity metric averaged across recording units (NaN units skipped).
    """
    if unit_summary is None:
        unit_summary = unit_complexity_summary(
            df, signal_col, condition_col=condition_col, metrics=metrics, unit_cols=unit_cols, time_col=time_col
        )
    metric_cols = list(metrics) if metrics else list(_METRIC_FUNCTIONS)
    records = []
    for condition, group in df.groupby(condition_col, dropna=False):
        units = unit_summary[unit_summary["condition"] == condition]
        records.append(
            {
                "condition": condition,
                "signal": signal_col,
                "n_samples": int(group.shape[0]),
                "n_units": int(units.shape[0]),
                **descriptive_stats(group[signal_col].to_numpy(dtype=float)),
                **{metric: float(units[metric].mean()) for metric in metric_cols},
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


def _unit_method_note(unit_complexity: pd.DataFrame, unit_cols: Sequence[str], time_col: str | None) -> str:
    order = f"sorted by {time_col}" if time_col else "in file order (no time column found)"
    if not unit_cols:
        return (
            "Complexity metrics are computed on each condition's samples as a single series, "
            f"{order}. No recording unit columns were used, so if a condition contains several "
            "independent recordings they are joined end to end, which distorts order-dependent "
            "metrics (sample entropy, spectral entropy, DFA)."
        )
    lengths = unit_complexity["n_samples"] if not unit_complexity.empty else pd.Series(dtype=int)
    length_text = (
        f"{int(lengths.min())}-{int(lengths.max())} samples per unit" if lengths.min() != lengths.max()
        else f"{int(lengths.min())} samples per unit"
    ) if not lengths.empty else "no units"
    return (
        f"Complexity metrics are computed separately within each recording unit ({', '.join(unit_cols)}; "
        f"{length_text}), {order}, then averaged across units per condition (n_units). "
        "Per-unit values are in complexity_by_unit.csv."
    )


def _undefined_metric_lines(unit_complexity: pd.DataFrame, metrics: Sequence[str]) -> list[str]:
    """Explain every metric that was undefined (NaN) for some recordings, so a blank or
    missing value is never mistaken for a real result."""
    lines = []
    if unit_complexity.empty:
        return lines
    for metric in metrics:
        if metric not in unit_complexity.columns:
            continue
        for signal, rows in unit_complexity.groupby("signal", sort=True):
            n_undefined = int(rows[metric].isna().sum())
            if not n_undefined:
                continue
            reason = _UNDEFINED_METRIC_REASONS.get(metric, "could not be computed")
            all_undefined = rows.groupby("condition", sort=True)[metric].apply(lambda v: v.isna().all())
            empty_conditions = [str(c) for c in all_undefined[all_undefined].index]
            if len(empty_conditions) == all_undefined.size:
                consequence = "it has no value for any condition"
            elif empty_conditions:
                consequence = (
                    f"it has no value for {', '.join(empty_conditions)}, and the remaining averages "
                    "use only the recordings where it is defined"
                )
            else:
                consequence = "per-condition averages use only the recordings where it is defined"
            lines.append(
                f"- {metric} is undefined for {n_undefined} of {rows.shape[0]} recording(s) of {signal} "
                f"({metric} {reason}); {consequence}."
            )
    if lines:
        lines.insert(0, "Undefined metrics:")
        lines.append("")
    return lines


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
    data_path: str | Path | Sequence[str | Path],
    signal_col: str | None = None,
    signal_cols: str | list[str] | None = None,
    condition_col: str = "condition",
    output_dir: str | Path = "outputs",
    domain: str | None = None,
    domain_config: str | Path | None = None,
    unit_cols: str | Sequence[str] | None = None,
) -> dict[str, pd.DataFrame | str]:
    load_domain_profiles(domain_config)
    domain_profile = get_domain_profile(domain)
    validate_recommended_metrics(domain_profile, _METRIC_FUNCTIONS.keys())
    df = load_dataset(data_path)
    selected_signals = resolve_signal_columns(df, signal_cols=signal_cols if signal_cols is not None else signal_col)
    missing_columns = _missing_expected_columns(df, domain_profile)
    df, provenance = clean_dataset(df, required_cols=[*selected_signals, condition_col])
    selected_units = resolve_unit_columns(df, unit_cols, condition_col=condition_col)
    time_col = resolve_time_column(df)

    summary_frames = []
    complexity_frames = []
    unit_complexity_frames = []
    significance_frames = []
    for signal in selected_signals:
        if signal not in df.columns:
            raise ValueError(f"Signal column '{signal}' was not found in the dataset.")
        summary_frames.append(summarize_condition(df, signal_col=signal, condition_col=condition_col).assign(signal=signal))
        metrics = domain_profile.recommended_metrics or None
        unit_complexity = unit_complexity_summary(
            df, signal_col=signal, condition_col=condition_col, metrics=metrics, unit_cols=selected_units, time_col=time_col
        )
        unit_complexity_frames.append(unit_complexity)
        complexity_frames.append(
            condition_complexity_summary(
                df, signal_col=signal, condition_col=condition_col, metrics=metrics, unit_summary=unit_complexity
            )
        )
        significance_frames.append(condition_significance_summary(df, signal_col=signal, condition_col=condition_col))

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    complexity = pd.concat(complexity_frames, ignore_index=True) if complexity_frames else pd.DataFrame()
    unit_complexity = pd.concat(unit_complexity_frames, ignore_index=True) if unit_complexity_frames else pd.DataFrame()
    significance = pd.concat(significance_frames, ignore_index=True) if significance_frames else pd.DataFrame()
    cross_signal = condition_cross_signal_summary(df, selected_signals, condition_col=condition_col)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "condition_summary.csv"
    complexity_path = output_dir / "complexity_summary.csv"
    unit_complexity_path = output_dir / "complexity_by_unit.csv"
    significance_path = output_dir / "significance_summary.csv"
    cross_signal_path = output_dir / "cross_signal_summary.csv"
    summary.to_csv(summary_path, index=False)
    complexity.to_csv(complexity_path, index=False)
    unit_complexity.to_csv(unit_complexity_path, index=False)
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
        f"- Data source: {_describe_data_source(data_path)}",
        f"- Signal columns: {', '.join(selected_signals)}",
        f"- Condition column: {condition_col}",
        f"- Recording unit columns: {', '.join(selected_units) if selected_units else 'none'}",
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
    report_lines.append("## Data provenance")
    report_lines.append(f"- Input rows: {provenance['input_rows']}")
    report_lines.append(f"- Duplicate rows detected (not removed automatically): {provenance['duplicate_rows_detected']}")
    report_lines.append(
        f"- Rows dropped for missing values in {provenance['missing_value_columns_checked']}: "
        f"{provenance['rows_dropped_missing_values']}"
    )
    report_lines.append(f"- Rows analyzed: {provenance['output_rows']}")
    report_lines.append("")
    report_lines.append("## Condition summary")
    report_lines.extend(_as_code_block(summary.to_string(index=False)))
    report_lines.append("## Complexity metrics")
    report_lines.append(_unit_method_note(unit_complexity, selected_units, time_col))
    report_lines.append("")
    report_lines.extend(_as_code_block(complexity.to_string(index=False)))
    report_lines.extend(
        _undefined_metric_lines(unit_complexity, domain_profile.recommended_metrics or list(_METRIC_FUNCTIONS))
    )
    if not significance.empty:
        report_lines.append("## Statistical significance")
        report_lines.extend(_as_code_block(significance.to_string(index=False)))
    if not cross_signal.empty:
        report_lines.append("## Cross-signal metrics")
        report_lines.extend(_as_code_block(cross_signal.to_string(index=False)))
    if plot_paths:
        report_lines.append("## Visualizations")
        for label, path in plot_paths.items():
            relative_path = Path(path).relative_to(output_dir)
            report_lines.append(f"![{label}]({relative_path.as_posix()})")
        report_lines.append("")
    report_lines.extend(
        _generate_interpretation_lines(complexity, domain_profile, cross_signal_df=cross_signal, significance_df=significance)
    )
    report_lines = _collapse_blank_lines(report_lines)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    html_path = export_html_report(report_path)

    return {
        "data": df,
        "selected_signals": selected_signals,
        "domain_profile": domain_profile.as_dict(),
        "missing_columns": missing_columns,
        "provenance": provenance,
        "condition_summary": summary,
        "complexity_summary": complexity,
        "complexity_by_unit": unit_complexity,
        "unit_columns": selected_units,
        "time_column": time_col,
        "significance_summary": significance,
        "cross_signal_summary": cross_signal,
        "summary_csv": str(summary_path),
        "complexity_csv": str(complexity_path),
        "complexity_by_unit_csv": str(unit_complexity_path),
        "significance_csv": str(significance_path),
        "cross_signal_csv": str(cross_signal_path),
        "plot_paths": plot_paths,
        "report_md": str(report_path),
        "report_html": html_path,
    }
