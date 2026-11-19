"""The app's page (docs/design/app.md). Start it with `complexity-app`.

Thin by design: widgets -> settings -> runner.run_analysis -> display. All analysis logic lives in
the engine, and the runner gives results identical to the CLI's (tests/test_app_runner.py).
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

from complexity_engine.app.choices import (
    DATA_TYPES,
    combined_preview,
    domain_choices,
    example_upload,
    timeseries_defaults,
)
from complexity_engine.app.runner import Upload, run_analysis
from complexity_engine.settings import TimeSeriesSettings

st.set_page_config(page_title="Complexity Engine", layout="wide")
state = st.session_state
state.setdefault("runs", [])          # finished runs, kept in memory for this session only
state.setdefault("example", None)     # the bundled example, when "Try an example" was clicked

st.title("Complexity Engine")
st.caption(
    "Runs on this computer: uploaded files never leave it, and nothing is saved to disk. "
    "Results are kept until you close the app."
)

analysis_type = st.radio("Analysis", ["Time series", "Network"], horizontal=True, key="analysis_type")
if analysis_type == "Network":
    st.info("The network page is coming next (docs/design/app.md, PR 3). Use `complexity-network` for now.")
    st.stop()

# 1. Upload --------------------------------------------------------------------------------------
st.header("1. Data")
files = st.file_uploader("CSV or Excel files (several files with the same columns are merged)",
                         type=DATA_TYPES, accept_multiple_files=True, key="uploads")
if st.button("Try an example", key="example_button", help="Load the bundled physiology example."):
    state.example = example_upload("timeseries")
    if state.example is None:
        st.warning("The bundled examples are only in a checkout of the repository, not in an installed copy.")
uploads = [Upload(f.name, f.getvalue()) for f in files] if files else ([state.example] if state.example else [])
if not uploads:
    st.stop()

try:
    preview = combined_preview(uploads)
except ValueError as error:
    st.error(str(error))
    st.stop()
st.write(f"{', '.join(u.safe_name for u in uploads)}: {len(preview):,} rows, {preview.shape[1]} columns")
st.dataframe(preview.head(20), width="stretch")

# 2. Domain ----------------------------------------------------------------------------------------
st.header("2. Domain")
config_file = st.file_uploader("Optional: your own domain profile file (YAML or JSON)",
                               type=["yaml", "yml", "json"], key="domain_config")
custom = Upload(config_file.name, config_file.getvalue()) if config_file else None
try:
    choices = domain_choices("timeseries", custom)
except (ValueError, UnicodeDecodeError) as error:
    st.error(f"Couldn't read the profile file: {error}")
    st.stop()
names = [c.name for c in choices]
domain = st.selectbox("Domain profile", names, index=names.index("physiology") if "physiology" in names else 0,
                      key="domain")
chosen = next(c for c in choices if c.name == domain)
st.markdown(f"**{chosen.description}**")
if chosen.question:
    st.caption(f"Analysis question: {chosen.question}")

# 3. Columns -------------------------------------------------------------------------------------
st.header("3. Columns")
defaults = timeseries_defaults(preview)
columns = list(preview.columns)
numeric = [c for c in columns if preview[c].dtype.kind in "iuf"]
signals = st.multiselect("Signal columns (the measurements to analyze)", numeric, default=defaults.signals,
                         key="signals")
condition = st.selectbox("Condition column (the groups or states to compare)", columns,
                         index=columns.index(defaults.condition) if defaults.condition in columns else 0,
                         key="condition")
unit_mode = st.radio("Recording units (one independent recording each, such as a subject or trial)",
                     ["Detect automatically", "Choose columns", "None: each condition is one recording"],
                     key="unit_mode")
unit_cols = None
if unit_mode == "Choose columns":
    unit_cols = st.multiselect("Unit columns", [c for c in columns if c != condition], default=defaults.units,
                               key="unit_cols")
elif unit_mode.startswith("None"):
    unit_cols = []
if unit_mode == "Detect automatically":
    st.caption(f"Detected: {', '.join(defaults.units) or 'none'}. "
               f"Time column: {defaults.time or 'none (file order is used)'}.")

# 4. Run ------------------------------------------------------------------------------------------
st.header("4. Run")
if not signals:
    st.warning("Choose at least one signal column.")
    st.stop()
settings = TimeSeriesSettings(
    data=[u.safe_name for u in uploads],
    # Leaving the engine's own choice in place passes None, exactly as the CLI does with no option.
    signal_cols=None if signals == defaults.signals else signals,
    condition_col=condition,
    unit_cols=unit_cols,
    domain=domain,
)
if st.button("Run analysis", type="primary", key="run"):
    with st.spinner("Analyzing... long recordings or surrogate baselines can take a few minutes."):
        try:
            result = run_analysis("timeseries", uploads, settings, domain_config=custom)
        except ValueError as error:
            st.error(str(error))
            st.stop()
    label = f"Run {len(state.runs) + 1} · {datetime.now():%H:%M:%S} · {domain} · {', '.join(settings.data)}"
    state.runs.insert(0, (label, result))

# 5. Results --------------------------------------------------------------------------------------
if not state.runs:
    st.stop()
st.header("5. Results")
labels = [label for label, _ in state.runs]
picked = st.selectbox("Run (this session's runs, newest first)", labels, key="picked_run")
result = dict(state.runs)[picked]
st.success(f"Finished in {result.seconds:.1f} s.")
st.code(result.command, language="bash")
left, right = st.columns(2)
left.download_button("Download all results (.zip)", result.zip_bytes, file_name="complexity_results.zip",
                     mime="application/zip", key=f"zip_{picked}")
right.download_button("Download report (.html)", result.report_html, file_name="report.html",
                      mime="text/html", key=f"html_{picked}")

report_tab, tables_tab, plots_tab = st.tabs(["Report", "Tables", "Plots"])
with report_tab:
    components.html(result.report_html, height=900, scrolling=True)
with tables_tab:
    for name, table in result.tables.items():
        if not table.empty:
            st.subheader(name)
            st.dataframe(table, width="stretch")
with plots_tab:
    for name, image in result.plots.items():
        st.image(image, caption=name)

if len(state.runs) > 1:
    with st.expander("Compare two runs"):
        first, second = st.columns(2)
        for column, key, default in ((first, "compare_a", 0), (second, "compare_b", 1)):
            label = column.selectbox("Run", labels, index=default, key=key)
            column.caption(dict(state.runs)[label].command)
            column.dataframe(dict(state.runs)[label].tables.get("complexity_summary.csv"),
                             width="stretch")
