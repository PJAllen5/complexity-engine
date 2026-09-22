"""Cross-checks inline citations like "(Friston, 2010)" in domain-profile and
analysis text against papers/references.bib, so a citation can't silently drift
out of sync with the bibliography (wrong year, removed entry, typo'd author).

Scans the project's source files for anything that looks like a citation rather
than a fixed, hand-maintained list, so new citations added to these files are
checked automatically without updating this test.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BIB_PATH = REPO_ROOT / "papers" / "references.bib"

# Files known to contain domain-grounded, citation-bearing prose.
SOURCE_FILES = [
    REPO_ROOT / "config" / "domain_profiles.yaml",
    REPO_ROOT / "config" / "domain_profiles.json",
    REPO_ROOT / "complexity_engine" / "domain_profiles.py",
    REPO_ROOT / "complexity_engine" / "analysis.py",
    REPO_ROOT / "complexity_engine" / "metrics.py",
]

_CITATION_GROUP_RE = re.compile(r"\(([^()]*\b\d{4}[a-z]?\b[^()]*)\)")
_CITATION_ITEM_RE = re.compile(r"^(.*?),\s*(\d{4})[a-z]?$")
_LATEX_ACCENT_RE = re.compile(r"\{\\['\"^~`]?([a-zA-Z])\}")


def normalize_name(name: str) -> str:
    name = _LATEX_ACCENT_RE.sub(r"\1", name)
    name = name.replace("{", "").replace("}", "").replace("\\", "")
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return name.strip().lower()


def parse_bib_entries(bib_text: str) -> list[dict]:
    blocks = re.split(r"\n(?=@\w+\{)", bib_text)
    entries = []
    for block in blocks:
        key_match = re.search(r"@\w+\{([^,\s]+),", block)
        author_match = re.search(r"author\s*=\s*\{(.*)\},\s*$", block, re.MULTILINE)
        year_match = re.search(r"year\s*=\s*\{(\d{4})\}", block)
        if not (key_match and author_match and year_match):
            continue
        surnames = [normalize_name(part.split(",")[0]) for part in author_match.group(1).split(" and ")]
        entries.append({"key": key_match.group(1), "surnames": surnames, "year": int(year_match.group(1))})
    return entries


def extract_citations(text: str) -> list[tuple[list[str], int, bool, str]]:
    """Return (surnames, year, is_et_al, raw_text) for every citation-like group found."""
    citations = []
    for group in _CITATION_GROUP_RE.findall(text):
        for item in group.split(";"):
            item = item.strip()
            match = _CITATION_ITEM_RE.match(item)
            if not match:
                continue
            authors_part, year = match.group(1), int(match.group(2))
            is_et_al = "et al." in authors_part
            authors_part = authors_part.replace("et al.", "").strip()
            if "&" in authors_part:
                surnames = [normalize_name(p) for p in authors_part.split("&")]
            else:
                surnames = [normalize_name(authors_part)]
            citations.append((surnames, year, is_et_al, item))
    return citations


def find_matching_entry(surnames: list[str], year: int, is_et_al: bool, entries: list[dict]) -> dict | None:
    for entry in entries:
        if entry["year"] != year:
            continue
        if is_et_al:
            if surnames[0] in entry["surnames"]:
                return entry
        elif all(surname in entry["surnames"] for surname in surnames):
            return entry
    return None


def test_bibliography_parses_known_entries():
    entries = parse_bib_entries(BIB_PATH.read_text(encoding="utf-8"))
    assert len(entries) >= 29
    assert any(e["key"] == "shannon1948mathematical" and e["year"] == 1948 for e in entries)
    assert any(e["key"] == "barabasi1999emergence" and "albert" in e["surnames"] for e in entries)


def test_citation_parser_handles_all_current_formats():
    sample = (
        "Single (Wiener, 1948). Two joined (Watts & Strogatz, 1998). "
        "Et al. (Rumelhart et al., 1986). Multiple in one group "
        "(Friston, 2010; Clark, 2013; Hohwy, 2013)."
    )
    citations = extract_citations(sample)
    assert (["wiener"], 1948, False, "Wiener, 1948") in citations
    assert (["watts", "strogatz"], 1998, False, "Watts & Strogatz, 1998") in citations
    assert (["rumelhart"], 1986, True, "Rumelhart et al., 1986") in citations
    assert len(citations) == 6


def test_all_inline_citations_match_the_bibliography():
    entries = parse_bib_entries(BIB_PATH.read_text(encoding="utf-8"))
    assert entries, "Expected to parse at least one bibliography entry"

    failures = []
    for path in SOURCE_FILES:
        text = path.read_text(encoding="utf-8")
        for surnames, year, is_et_al, raw in extract_citations(text):
            if find_matching_entry(surnames, year, is_et_al, entries) is None:
                failures.append(f"{path.relative_to(REPO_ROOT)}: no bibliography entry matches '({raw})'")

    assert not failures, "\n" + "\n".join(failures)
