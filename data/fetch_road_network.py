"""Write the California road network (SNAP roadNet-CA) as an edge-list CSV.

roadNet-CA (Leskovec, Lang, Dasgupta & Mahoney, 2009): intersections and endpoints of
California's roads as nodes, road segments as edges; 1,965,206 nodes and 2,766,607 edges.
SNAP lists each segment in both directions; it's written once per pair. This is the scale test
for the network pipeline's large-network mode (roadmap items E6/B5).

Output, under data/external/ (not committed; about 40 MB; cite the paper above):
- roadnet_ca_edges.csv: source, target

Usage:
    python data/fetch_road_network.py
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SNAP_URL = "https://snap.stanford.edu/data/roadNet-CA.txt.gz"

DATA_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = DATA_DIR / "external"
CACHE = EXTERNAL_DIR / "snap" / "roadNet-CA.txt.gz"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def main() -> None:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(SNAP_URL, context=_ssl_context()) as response:
                CACHE.write_bytes(response.read())
        except urllib.error.URLError as error:
            raise SystemExit(f"Couldn't download {SNAP_URL} ({error.reason}). Download it yourself to {CACHE} and rerun.") from error
    edges = pd.read_csv(CACHE, sep="\t", comment="#", header=None, names=["source", "target"])
    pairs = pd.DataFrame({"source": np.minimum(edges["source"], edges["target"]),
                          "target": np.maximum(edges["source"], edges["target"])}).drop_duplicates()
    pairs.to_csv(EXTERNAL_DIR / "roadnet_ca_edges.csv", index=False)
    print(f"Wrote roadnet_ca_edges.csv: {len(pairs)} edges")


if __name__ == "__main__":
    main()
