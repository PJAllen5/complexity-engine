"""Complexity engineering MVP package for scientific time-series analysis."""

from .analysis import analyze_dataset, load_dataset, resolve_signal_columns
from .metrics import (
    compare_conditions,
    compute_complexity_metrics,
    dfa_exponent,
    mutual_information,
    sample_entropy,
    spectral_entropy,
)

__all__ = [
    "analyze_dataset",
    "load_dataset",
    "resolve_signal_columns",
    "compute_complexity_metrics",
    "sample_entropy",
    "spectral_entropy",
    "dfa_exponent",
    "mutual_information",
    "compare_conditions",
]
