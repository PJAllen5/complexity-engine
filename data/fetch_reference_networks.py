"""Write classic reference networks with published results as edge-list CSVs.

All four are Mark Newman's network data collection, downloaded from the Netzschleuder
mirror (https://networks.skewed.de), since the original umich.edu pages refuse scripted
downloads. Netzschleuder numbers nodes from 0 and ships each network as edges.csv,
nodes.csv and gprops.csv in a zip; the zips are cached under data/external/netzschleuder/.

- C. elegans neural network (White et al., 1986), as used by Watts & Strogatz (1998).
  Newman's version has 297 neurons and is directed and weighted (synapse counts). It's
  written undirected and unweighted, without self-loops, as Watts & Strogatz analyzed it.
- US western power grid (Watts & Strogatz, 1998): 4,941 transformers, substations and
  generators, linked by high-voltage lines.
- Doubtful Sound bottlenose dolphins (Lusseau et al., 2003): 62 dolphins linked by frequent
  association. Used by the animal_social_network validation.
- Political blogs (Adamic & Glance, 2005): directed hyperlinks among blogs about the 2004 US
  election, with each blog's political leaning. Used by the attribute-mixing validation.

Output, under data/external/ (not committed; cite the papers above when you use the data):
- celegans_edges.csv, power_grid_edges.csv, dolphins_edges.csv: source, target (undirected)
- dolphins_nodes.csv: node, label (the dolphin's name, e.g. SN100)
- polblogs_edges.csv: source, target (directed)
- polblogs_nodes.csv: node, label, leaning (0 = liberal, 1 = conservative, as in the source),
  source (the blog list the entry came from)

Usage:
    python data/fetch_reference_networks.py
"""

from __future__ import annotations

import io
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

MIRROR = "https://networks.skewed.de/net/{name}/files/{name}.csv.zip"

DATA_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = DATA_DIR / "external"
CACHE_DIR = EXTERNAL_DIR / "netzschleuder"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _download(name: str) -> zipfile.ZipFile:
    cached = CACHE_DIR / f"{name}.csv.zip"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        url = MIRROR.format(name=name)
        try:
            with urllib.request.urlopen(url, context=_ssl_context()) as response:
                cached.write_bytes(response.read())
        except urllib.error.URLError as error:
            raise SystemExit(
                f"Couldn't download {url} ({error.reason}). Download it yourself to {cached} and rerun."
            ) from error
    return zipfile.ZipFile(cached)


def _read(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    """Read a Netzschleuder CSV, whose header line starts with '# '."""
    text = archive.read(member).decode("utf-8")
    frame = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    return frame.rename(columns=lambda c: c.lstrip("#").strip())


def _undirected_simple(edges: pd.DataFrame) -> pd.DataFrame:
    """Each unordered pair once, without self-loops (direction and weights dropped)."""
    pairs = pd.DataFrame({
        "source": edges[["source", "target"]].min(axis=1),
        "target": edges[["source", "target"]].max(axis=1),
    })
    pairs = pairs[pairs["source"] != pairs["target"]]
    return pairs.drop_duplicates().sort_values(["source", "target"]).reset_index(drop=True)


def write_undirected(name: str, output: str) -> None:
    edges = _undirected_simple(_read(_download(name), "edges.csv"))
    edges.to_csv(EXTERNAL_DIR / output, index=False)
    print(f"Wrote {output}: {len(edges)} edges")


def write_dolphin_names() -> None:
    nodes = _read(_download("dolphins"), "nodes.csv").rename(columns={"index": "node"})
    nodes[["node", "label"]].to_csv(EXTERNAL_DIR / "dolphins_nodes.csv", index=False)
    print(f"Wrote dolphins_nodes.csv: {len(nodes)} named dolphins")


def write_polblogs() -> None:
    archive = _download("polblogs")
    edges = _read(archive, "edges.csv")[["source", "target"]]
    nodes = _read(archive, "nodes.csv").rename(columns={"index": "node", "value": "leaning"})
    edges.to_csv(EXTERNAL_DIR / "polblogs_edges.csv", index=False)
    nodes[["node", "label", "leaning", "source"]].to_csv(EXTERNAL_DIR / "polblogs_nodes.csv", index=False)
    print(f"Wrote polblogs_edges.csv ({len(edges)} edges) and polblogs_nodes.csv ({len(nodes)} nodes)")


def main() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    write_undirected("celegansneural", "celegans_edges.csv")
    write_undirected("power", "power_grid_edges.csv")
    write_undirected("dolphins", "dolphins_edges.csv")
    write_dolphin_names()
    write_polblogs()


if __name__ == "__main__":
    main()
