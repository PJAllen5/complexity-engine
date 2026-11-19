"""Run settings shared by the command line and the app.

Each pipeline has one settings object. The CLI builds it from its arguments and the app builds it
from its widgets; both then call the analysis with settings.analysis_kwargs(), so the two can't
drift apart. settings.cli_args() turns settings back into the equivalent command, which the app
records in settings.json so any run can be repeated from the command line.
"""

from __future__ import annotations

import argparse
import shlex
from dataclasses import asdict, dataclass
from typing import Any


def _split(text: str | None) -> list[str]:
    return [item.strip() for item in (text or "").split(",") if item.strip()]


@dataclass
class TimeSeriesSettings:
    """Settings for analyze_dataset / complexity-analysis.

    unit_cols: None auto-detects recording units; an empty list means none (CLI: --unit-cols none).
    """

    data: list[str]
    signal_cols: list[str] | None = None
    condition_col: str = "condition"
    unit_cols: list[str] | None = None
    domain: str = "physiology"
    domain_config: str | None = None
    output_dir: str = "outputs"

    command = "complexity-analysis"

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> "TimeSeriesSettings":
        unit_cols = None
        if args.unit_cols is not None:
            unit_cols = [] if args.unit_cols.strip().lower() == "none" else _split(args.unit_cols)
        return cls(
            data=_split(args.data),
            signal_cols=_split(args.signal_cols) or None,
            condition_col=args.condition_col,
            unit_cols=unit_cols,
            domain=args.domain,
            domain_config=args.domain_config,
            output_dir=args.output_dir,
        )

    def analysis_kwargs(self) -> dict[str, Any]:
        return {
            "data_path": self.data[0] if len(self.data) == 1 else list(self.data),
            "signal_cols": self.signal_cols,
            "condition_col": self.condition_col,
            "output_dir": self.output_dir,
            "domain": self.domain,
            "domain_config": self.domain_config,
            "unit_cols": self.unit_cols,
        }

    def cli_args(self) -> list[str]:
        args = ["--data", ",".join(self.data)]
        if self.signal_cols:
            args += ["--signal-cols", ",".join(self.signal_cols)]
        args += ["--condition-col", self.condition_col]
        if self.unit_cols is not None:
            args += ["--unit-cols", ",".join(self.unit_cols) or "none"]
        args += ["--domain", self.domain]
        if self.domain_config:
            args += ["--domain-config", self.domain_config]
        args += ["--output-dir", self.output_dir]
        return args


@dataclass
class NetworkSettings:
    """Settings for analyze_network / complexity-network.

    weight_col and condition_col: None means no such column (CLI: an empty string).
    """

    data: list[str]
    source_col: str = "source"
    target_col: str = "target"
    weight_col: str | None = "weight"
    condition_col: str | None = "condition"
    directed: bool = False
    compute_centrality: bool = False
    compute_small_world: bool = False
    compute_communities: bool = False
    nodes: str | None = None
    node_col: str = "node"
    attributes: list[str] | None = None
    permutations: int = 1000
    path_sources: int = 50
    random_seed: int = 42
    domain: str = "network"
    domain_config: str | None = None
    output_dir: str = "outputs/network"

    command = "complexity-network"

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> "NetworkSettings":
        return cls(
            data=_split(args.data),
            source_col=args.source_col,
            target_col=args.target_col,
            weight_col=args.weight_col or None,
            condition_col=args.condition_col or None,
            directed=args.directed,
            compute_centrality=args.compute_centrality,
            compute_small_world=args.compute_small_world,
            compute_communities=args.compute_communities,
            nodes=args.nodes,
            node_col=args.node_col,
            attributes=_split(args.attributes) or None,
            permutations=args.permutations,
            path_sources=args.path_sources,
            random_seed=args.random_seed,
            domain=args.domain,
            domain_config=args.domain_config,
            output_dir=args.output_dir,
        )

    def analysis_kwargs(self) -> dict[str, Any]:
        return {
            "data_path": self.data[0] if len(self.data) == 1 else list(self.data),
            "source_col": self.source_col,
            "target_col": self.target_col,
            "weight_col": self.weight_col,
            "condition_col": self.condition_col,
            "directed": self.directed,
            "compute_centrality": self.compute_centrality,
            "compute_small_world": self.compute_small_world,
            "compute_communities": self.compute_communities,
            "random_seed": self.random_seed,
            "output_dir": self.output_dir,
            "domain": self.domain,
            "domain_config": self.domain_config,
            "node_table": self.nodes,
            "node_col": self.node_col,
            "attributes": self.attributes,
            "permutations": self.permutations,
            "path_sources": self.path_sources,
        }

    def cli_args(self) -> list[str]:
        args = ["--data", ",".join(self.data), "--source-col", self.source_col, "--target-col", self.target_col,
                "--weight-col", self.weight_col or "", "--condition-col", self.condition_col or ""]
        for flag, on in (("--directed", self.directed), ("--compute-centrality", self.compute_centrality),
                         ("--compute-small-world", self.compute_small_world),
                         ("--compute-communities", self.compute_communities)):
            if on:
                args.append(flag)
        if self.nodes:
            args += ["--nodes", self.nodes, "--node-col", self.node_col]
        if self.attributes:
            args += ["--attributes", ",".join(self.attributes)]
        args += ["--permutations", str(self.permutations), "--path-sources", str(self.path_sources),
                 "--random-seed", str(self.random_seed), "--domain", self.domain]
        if self.domain_config:
            args += ["--domain-config", self.domain_config]
        args += ["--output-dir", self.output_dir]
        return args


def command_line(settings: TimeSeriesSettings | NetworkSettings) -> str:
    """The shell command that repeats a run, e.g. 'complexity-analysis --data a.csv ...'."""
    return shlex.join([settings.command, *settings.cli_args()])


def as_dict(settings: TimeSeriesSettings | NetworkSettings) -> dict[str, Any]:
    return asdict(settings)


SETTINGS_FOR = {"timeseries": TimeSeriesSettings, "network": NetworkSettings}
__all__ = ["TimeSeriesSettings", "NetworkSettings", "command_line", "as_dict", "SETTINGS_FOR"]
