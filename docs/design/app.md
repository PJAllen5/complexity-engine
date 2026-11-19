# Design: a local app for running the engine without the command line

Status: proposed 2026-11-18. Version 1 runs locally for one user (PJ). Hosting it for other
researchers is a later step, planned for at the end of this doc and kept out of version 1.

## Goal

Researchers should be able to analyze a dataset without the command line: upload a CSV, say which
domain it belongs to and which columns are which, and get back the same report and files the CLI
produces, plus one download with everything in it.

The app is only an interface. It doesn't change any analysis. The science is judged by the
validations; the app is judged on whether it delivers that science intact.

## What "done" means for version 1

1. **Parity:** every option of `complexity-analysis` and `complexity-network` can be set in the app.
2. **Same results:** given the same file and settings, the app writes CSVs identical to the CLI's.
   An automated test checks this on validation datasets that CI already has (gait, lynx, karate
   club, dolphins).
3. **Downloads:** each run offers a zip with the report (HTML and Markdown), every CSV, every
   plot and a `settings.json` recording every option used, plus the self-contained `report.html`
   on its own.
4. **Understandable failures:** when a run fails, the app shows the engine's own error message
   (they already explain what's wrong, e.g. "Domain profile 'x' sets axis: time, which needs a
   readable time column"), not a stack trace.

## Why Streamlit

Streamlit turns a Python script into a web page that runs locally (`streamlit run`) and can be
hosted later as-is. Everything stays in Python, which fits the project, and version 1 needs no web
server, JavaScript or database.

The trade-off is that a run blocks the page until it finishes. Most runs take seconds; surrogate
baselines and very large networks take minutes. For version 1 the page shows a progress message
and the elapsed time, and that's acceptable. If hosted use needs runs in the background, see
"When to outgrow Streamlit" below.

## What the user sees

One page, with two analysis types chosen at the top: **Time series** and **Network**.

1. **Upload.** One or more CSV or Excel files. Several files are merged, as `--data a.csv,b.csv`
   does. A preview shows the first rows and each column's type.
2. **Domain.** A list of the configured profiles, each with its description and analysis question
   from the YAML, so the choice is informed. An optional upload of a custom profile file stands in
   for `--domain-config`.
3. **Columns.** Pickers filled from the uploaded file's columns, with the engine's own
   auto-detection as the default. The user can always see and change what will be used.
   - Time series: signal columns (several allowed), condition column, recording-unit columns (or
     none).
   - Network: source, target, optional weight and condition columns, directed or not; optional node
     table upload with its id column and the attributes to compare.
4. **Options** (network only, collapsed by default): centrality, communities, small-world
   statistics, permutations, path sources, random seed. Each has the CLI help text as its tooltip.
5. **Run.** A button, then a progress message with the elapsed time.
6. **Results.** The rendered report (the same `report.html`, shown in the page), then the main
   tables (condition summary, complexity by unit, significance; network metrics, attribute mixing)
   as sortable tables, then the plots.
7. **Download.** "Download all results (.zip)" and "Download report (.html)". The zip includes
   `settings.json`: the analysis type, file names, domain (and custom profile file, if any), every
   column choice and option, the random seed, and the engine version. It also lists the equivalent
   `complexity-analysis` or `complexity-network` command, so the run can be repeated exactly from the
   command line, by you or by a reviewer.

The profile settings in effect (metric settings, transforms, surrogates, gap policy) are already
printed at the top of every report, so the user can see exactly what was run.

## How it's built

```
complexity_engine/
  app/
    runner.py      # no Streamlit: settings in, analysis run in a temporary folder, results out
    streamlit_app.py   # the page: widgets -> settings -> runner -> display and downloads
```

- **`runner.py` holds everything except the page.** It takes the uploaded files and the chosen
  settings, writes the uploads to a fresh temporary folder, calls `analyze_dataset` or
  `analyze_network` with exactly the keyword arguments the CLI would pass, writes `settings.json`
  from the same settings object, zips the outputs, and returns the results. Because it doesn't import Streamlit, it can be tested like any other module,
  and the parity test compares its CSVs with the CLI's.
- **The CLI and the app build their arguments the same way.** Each CLI option maps to one function
  argument, so a small shared settings object (one for each pipeline), used by both, keeps them from
  drifting apart. When a new CLI flag is added, the parity test fails until the app has it too.
- **`streamlit_app.py` stays thin:** widgets, a call to the runner, and display. Streamlit's built-in
  test harness (`streamlit.testing`) gives one smoke test that the page loads and a run completes.
- **Packaging:** `streamlit` becomes an optional extra (`pip install -e ".[app]"`), so the engine
  and CI don't depend on it, and a new command `complexity-app` starts the page.

## Files and data in version 1

- Uploads and outputs go to a temporary folder for each run, which the runner deletes before
  returning. Results, including the zip, are held in memory for the session and disappear when the
  app closes (see Decisions, 3). Nothing is kept between sessions.
- Streamlit's default upload limit is 200 MB. Version 1 keeps it, and states it on the page. The
  large validation inputs (the 660 MB resting-EEG CSV, the 41 MB road network) are command-line
  jobs for now.
- Running locally means data never leaves the machine. That matters because some of the engine's
  data is human-subject physiology.

## Not in version 1

Hosting, user accounts, several users at once, saving past runs, background jobs, PDF export,
editing domain profiles in the app, and the GHCN and other fetch scripts. Each can be added later
without changing the runner.

## Before hosting it for others

These come from the design discussion on 2026-11-18 and need to be in place before anyone other
than PJ uploads data:

1. **Retention, unconditional and time-limited.** Delete each upload as soon as its analysis
   finishes. Delete the report when it's downloaded or after 24 hours, whichever comes first.
   Never keep data based on what's in it: the app can't tell whether a file is identifiable.
2. **Consent and de-identification notice.** Before upload, the user confirms they have the right to
   process the data and that it's de-identified. This is a policy safeguard; retention (1) is the
   technical one.
3. **Encrypted connections** (HTTPS) and no logging of file contents.
4. **Limits.** Upload size, run time and memory caps, so one large job can't take the server down.
5. **Licences.** Some datasets forbid redistribution (WESAD is non-commercial). Results returned to
   the uploader are fine, but the app must never share one user's data or results with another.
6. **Where results and zips live.** Version 1 keeps each session's results, zip included, in
   memory, which is fine for one local user. Hosted, many sessions with large files would exhaust a
   server's memory, so decide then between memory, the server's disk or separate file storage, and
   how retention (1) applies to each. (Agreed 2026-11-18 to revisit at hosting time.)
7. **Cost and where to host.** Free tiers (Streamlit Community Cloud, Hugging Face Spaces) are
   public by default and have small CPU and memory limits. A small paid server gives control over
   retention and limits. Decide once the expected users and file sizes are known.

### When to outgrow Streamlit

If hosted use needs several simultaneous users or runs longer than a page should wait, move the
runner behind a small web API with a job queue: upload, get a job id, come back for the results.
The runner doesn't change; only the layer in front of it does. That's why it's kept separate from
the page.

## Delivery plan

Three PRs, each under about 500 lines:

1. **Runner and shared settings,** with no page yet. The CLI uses the shared settings objects, and
   the parity test compares runner and CLI outputs on the validation datasets. This PR proves the
   "same results" criterion before any interface exists.
2. **Time-series page:** upload, preview, domain and column choices, run, report, downloads, and the
   `complexity-app` command, plus a Streamlit smoke test.
3. **Network page:** edge list, node table, attributes and options.

## Decisions (2026-11-18)

1. **`settings.json` in every zip:** see "What the user sees", step 7.
2. **One-click demos:** the page offers the bundled examples (`physio_example.csv`,
   `network_example.csv`) as "Try an example" buttons, so a first-time user sees a full run
   without their own data.
3. **Runs are kept for the session:** each finished run stays in a list for the rest of the
   session, so runs can be compared side by side. They're kept **in memory, not on disk**: the
   runner deletes each run's temporary folder before returning (implemented in PR 1), and the
   session's runs disappear when the app closes. This replaces "deleted when the next run
   starts" in "Files and data in version 1".
