from complexity_engine.report_export import export_html_report


def test_export_html_report_inlines_images_and_writes_next_to_markdown(tmp_path):
    # A minimal 1x1 PNG, base64-decoded, so the image path actually resolves.
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
        "53de0000000c4944415408d763f8ffff3f0005fe02fea7d0e5e70000000049454e44ae426082"
    )
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "chart.png").write_bytes(png_bytes)

    report_md = tmp_path / "report.md"
    report_md.write_text(
        "# Title\n\n"
        "- a: 1\n\n"
        "## Table\n```\ncol1 col2\n1 2\n```\n\n"
        "## Visualizations\n![chart](plots/chart.png)\n",
        encoding="utf-8",
    )

    html_path = export_html_report(report_md)

    assert html_path == str(tmp_path / "report.html")
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<h1>Title</h1>" in html
    assert "<pre><code>col1 col2" in html
    assert "data:image/png;base64," in html
    assert "plots/chart.png" not in html


def test_export_html_report_leaves_missing_images_untouched(tmp_path):
    report_md = tmp_path / "report.md"
    report_md.write_text("# Title\n\n![missing](plots/does_not_exist.png)\n", encoding="utf-8")

    html_path = export_html_report(report_md)

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "plots/does_not_exist.png" in html
    assert html_path.endswith("report.html")
