"""Write two real social networks with published results as edge-list CSVs.

- Zachary's karate club (Zachary, 1977): friendships among 34 members of a university karate
  club that later split in two after a dispute. Community detection recovers that split
  with at most one member misclassified (Girvan & Newman, 2002; Newman & Girvan, 2004), which
  makes it a known-answer check for community structure. The network ships with networkx,
  so nothing is downloaded. Zachary's ties are written unweighted, as in those analyses,
  and the faction each member joined goes to a separate node file.
- ca-GrQc (Leskovec, Kleinberg & Faloutsos, 2007): co-authorships among 5,242 authors in
  arXiv's General Relativity and Quantum Cosmology section, 1993-2003, from the Stanford
  Network Analysis Project (SNAP), which publishes its size and clustering. SNAP lists each
  tie in both directions; it is written once per pair. The 12 self-loops are kept, since
  SNAP's edge count includes them.

Output, under data/external/ (not committed; cite the papers above when you use the data):
- karate_club_edges.csv: source, target (members numbered 1-34, as in Zachary, 1977)
- karate_club_factions.csv: node, faction ("Mr. Hi" or "Officer", networkx's labels)
- ca_grqc_edges.csv: source, target

Usage:
    python data/fetch_social_networks.py
"""

from __future__ import annotations

import gzip
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import networkx as nx
import pandas as pd

SNAP_URL = "https://snap.stanford.edu/data/ca-GrQc.txt.gz"

DATA_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = DATA_DIR / "external"
SNAP_CACHE = EXTERNAL_DIR / "snap" / "ca-GrQc.txt.gz"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _download_snap() -> Path:
    if not SNAP_CACHE.exists():
        SNAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(SNAP_URL, context=_ssl_context()) as response:
                SNAP_CACHE.write_bytes(response.read())
        except urllib.error.URLError as error:
            if isinstance(error.reason, ssl.SSLCertVerificationError):
                raise SystemExit(
                    f"Couldn't verify SNAP's TLS certificate ({error.reason}). This Python has no "
                    "CA certificates configured: `pip install certifi`, or on macOS run "
                    "'Install Certificates.command' from your Python folder. Alternatively, download "
                    f"{SNAP_URL} yourself to {SNAP_CACHE} and rerun."
                ) from error
            raise
    return SNAP_CACHE


def write_karate_club() -> None:
    graph = nx.karate_club_graph()
    edges = pd.DataFrame([(u + 1, v + 1) for u, v in graph.edges()], columns=["source", "target"])
    factions = pd.DataFrame(
        [(node + 1, data["club"]) for node, data in graph.nodes(data=True)], columns=["node", "faction"]
    )
    edges.to_csv(EXTERNAL_DIR / "karate_club_edges.csv", index=False)
    factions.to_csv(EXTERNAL_DIR / "karate_club_factions.csv", index=False)
    print(f"Karate club: {len(factions)} members, {len(edges)} ties, factions {factions['faction'].value_counts().to_dict()}")


def write_ca_grqc() -> None:
    with gzip.open(_download_snap(), "rt") as handle:
        pairs = pd.read_csv(handle, sep="\t", comment="#", header=None, names=["source", "target"])
    lower = pairs[["source", "target"]].min(axis=1)
    upper = pairs[["source", "target"]].max(axis=1)
    edges = pd.DataFrame({"source": lower, "target": upper}).drop_duplicates().sort_values(["source", "target"])
    edges.to_csv(EXTERNAL_DIR / "ca_grqc_edges.csv", index=False)
    self_loops = int((edges["source"] == edges["target"]).sum())
    print(f"ca-GrQc: {len(pairs)} directed rows -> {len(edges)} undirected ties ({self_loops} self-loops)")


def main() -> int:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    write_karate_club()
    write_ca_grqc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
