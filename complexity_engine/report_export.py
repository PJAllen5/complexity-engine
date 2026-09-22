"""Exports a generated report.md (from either pipeline) as a single, self-contained
HTML file: the plot PNGs it references are base64-inlined, so the result is one
file a researcher can send to a collaborator, rather than a Markdown file plus a
directory of images that have to be gathered separately.

Data-shape-agnostic -- operates purely on the already-written report.md and the
image files next to it, so it's shared by both analyze_dataset() and
analyze_network() rather than duplicated.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")

_CSS = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 60rem;
       margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #1a1a1a; }
h1, h2, h3 { border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
pre { background: #f5f5f5; padding: 0.75rem 1rem; overflow-x: auto; border-radius: 4px; }
code { font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.9em; }
img { max-width: 100%; height: auto; display: block; margin: 1rem 0; }
ul { padding-left: 1.5rem; }
"""


def _inline_images(markdown_text: str, base_dir: Path) -> str:
    """Replace ![alt](relative/path.png) with a base64 data URI, resolved against
    base_dir (the report's own directory). Leaves anything that isn't a local,
    existing image file untouched rather than raising -- a missing plot shouldn't
    block exporting the rest of the report.
    """

    def replace(match: re.Match[str]) -> str:
        alt, src = match.group(1), match.group(2)
        image_path = (base_dir / src).resolve()
        if not image_path.is_file():
            return match.group(0)
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        suffix = image_path.suffix.lstrip(".").lower() or "png"
        return f"![{alt}](data:image/{suffix};base64,{encoded})"

    return _IMAGE_RE.sub(replace, markdown_text)


def export_html_report(report_md_path: str | Path) -> str:
    """Convert a report.md into a self-contained report.html next to it."""
    report_md_path = Path(report_md_path)
    markdown_text = report_md_path.read_text(encoding="utf-8")
    markdown_text = _inline_images(markdown_text, report_md_path.parent)

    body = markdown.markdown(markdown_text, extensions=["fenced_code", "tables"])
    title = report_md_path.stem.replace("_", " ").title()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""

    html_path = report_md_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    return str(html_path)
