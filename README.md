# Report Agent

![Report Agent](img/report-agent-header.svg)

Concise, modular system to generate **weekly data reports** directly from ClickHouse + dbt docs using OpenAI's **Code Interpreter** (Python tool).

The model receives **raw tables** (CSV) plus neutral context (schema, meta, optional dbt docs), then decides how to analyze, visualize, and summarize — **no precomputed metrics**. Each run produces:

* Per-metric HTML reports with BD-friendly narrative and plots.
* A unified weekly report HTML that synthesizes findings across all metrics.

---

## Key Features

* **Free-form analysis**: Model runs Python in a sandbox (Responses API + `code_interpreter`), no fixed toolchain.
* **Parallel processing**: Process multiple metrics concurrently with `--max-workers` flag for faster execution.
* **Optimized prompts**: Reduced token usage (~30-40%) through prompt optimization while maintaining all functionality.
* **Raw inputs**: Time series (date, value, optional label) and snapshots (value, optional change_pct, optional label).
* **Neutral context**: Attaches `*.schema.json` (dtypes/examples/roles), `*.meta.json` (coverage/kind), and optional `*.docs.md`.
* **Evidence-first visuals**: Weekly total + WoW movers plots required by the prompt for time series.
* **Static HTML**: Dark-themed per-metric pages + a summary page, all assets referenced via relative paths.

---

## Repository Layout

```
report_agent/
  cli/
    main.py                       # CLI entrypoint (installed as `report-agent`)

  connectors/
    db/
      clickhouse_connector.py     # read-only ClickHouse client
    llm/
      base.py                     # abstract LLMConnector
      openai.py                   # OpenAICodeInterpreterConnector (Responses API + CI)

  metrics/
    metrics.yml                   # metric list + kind + history_days
    metrics_loader.py             # fetch_time_series() and fetch_snapshot()
    metrics_registry.py           # loads metrics.yml, helpers (list_models, get_kind, etc.)

  dbt_context/
    from_docs_json.py             # optional dbt manifest/docs ingestion

  nlg/
    prompt_builder.py             # builds CI prompts (time_series vs snapshot)
    html_report.py                # render per-metric HTML
    report_service.py             # run CI -> download plots -> save CSV/text -> HTML
    summary_service.py            # unified weekly report synthesizing all metrics
    templates/
      ci_report_prompt.j2         # time-series CI prompt (weekly report + required plots)
      ci_snapshot_prompt.j2       # snapshot CI prompt (single KPI)
      report_page.html.j2         # per-metric HTML template (dark theme)
      summary_prompt.j2           # weekly report LLM prompt
      summary_page.html.j2        # weekly report HTML template

  utils/
    config_loader.py              # loads .env and returns config dict
```

**Output structure after a run (default `reports/`):**

```
reports/
  2025-11-26_api_p2p_discv4_clients_daily.html
  2025-11-26_api_execution_transactions_active_accounts_7d.html
  portfolio_summary.html

  plots/
    api_p2p_discv4_clients_daily_headline_weekly.png
    api_p2p_discv4_clients_daily_top5_wow.png
    ...

  data/
    api_p2p_discv4_clients_daily.csv
    api_execution_transactions_active_accounts_7d.csv

  text/
    api_p2p_discv4_clients_daily.txt
    api_execution_transactions_active_accounts_7d.txt
```

---

## Requirements

* Python 3.10+
* Install package (from repo root):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

Dependencies (via `pyproject.toml`) include: `openai`, `clickhouse-connect`, `jinja2`, `markdown`, `httpx`, `pandas`, etc.

---

## Configuration

Set credentials in `.env` (loaded by `utils/config_loader.py`):

```bash
# OpenAI
OPENAI_API_KEY=...

# Optional overrides
# OPENAI_MODEL=gpt-4.1
# OPENAI_SUMMARY_MODEL=gpt-4.1-mini

# ClickHouse
CLICKHOUSE_HOST=...
CLICKHOUSE_USER=...
CLICKHOUSE_PASSWORD=...
CLICKHOUSE_DB_READ=dbt
CLICKHOUSE_DB_WRITE=playground_max
CLICKHOUSE_SECURE=true

# dbt docs / manifest (optional)
DBT_MANIFEST_PATH=/path/to/manifest.json
# or
# DBT_DOCS_BASE_URL=https://your-dbt-docs-root/

# Custom base (optional; e.g., Azure/OpenAI-compatible)
# OPENAI_BASE_URL=https://.../v1
```

Define metrics in `report_agent/metrics/metrics.yml`:

```yaml
metrics:
  # Time-series with optional label dimension
  - model: api_p2p_discv4_clients_daily
    kind: time_series
    history_days: 180

  - model: api_execution_transactions_active_accounts_by_sector_daily
    kind: time_series
    history_days: 180

  # Snapshot / number display
  - model: api_execution_transactions_active_accounts_7d
    kind: snapshot
```

**Conventions:**
* `kind: time_series`: table has a date column; value is the main measure; label (if present) is a dimension.
* `kind: snapshot`: no date required; typically value, optional change_pct, optional label.

---

## How It Works

1.  **CLI loads config + registry**: `report-agent` uses `load_configs()` and `MetricsRegistry` to discover metrics, kinds, and history windows. Metrics can be processed sequentially or in parallel (default: 3 workers).
2.  **Fetch data from ClickHouse**:
    * Time series: `fetch_time_series(model, lookback_days)` queries by `date >= today() - INTERVAL ...`.
    * Snapshots: `fetch_snapshot(model)` selects the whole table (no date filter).
3.  **Prepare CI inputs**: For each metric, the connector writes:
    * `{model}.csv` — raw rows
    * `{model}.schema.json` — dtypes + sample values + simple roles (time/measure/dimension/delta)
    * `{model}.meta.json` — counts, (optional) date range, kind
    * `{model}.docs.md` — dbt model + column docs (if available)
4.  **Run Code Interpreter**:
    * Files uploaded to OpenAI as container files.
    * `build_ci_prompt()` selects the correct prompt template (time series vs snapshot).
    * Responses API runs with `code_interpreter` and returns a BD-facing narrative.
5.  **Persist outputs**:
    * Narrative → `reports/text/<model>.txt`
    * Plots → downloaded via `download_artifacts()` into `reports/plots/`
    * Data → `reports/data/<model>.csv`
    * Per-metric HTML → `html_report.render_html_report()` writes `YYYY-MM-DD_<model>.html`
6.  **Weekly report (optional)**:
    * Uses `summary_service.generate_weekly_report()` on the collected metric texts + dbt docs.
    * LLM synthesizes findings across all metrics and creates a unified weekly report.

---

## Quick Start

1.  Install and configure:
    ```bash
    pip install -e .
    # create and fill .env
    ```
2.  Define metrics in `report_agent/metrics/metrics.yml`.
3.  Run all metrics + summary (with parallel processing):
    ```bash
    report-agent
    ```
4.  Run with custom number of parallel workers:
    ```bash
    report-agent --max-workers 5
    ```
5.  Run a single metric:
    ```bash
    report-agent --metric api_p2p_discv4_clients_daily
    ```
6.  Change output directory / skip summary:
    ```bash
    report-agent --out-dir gnosis_reports --no-summary
    ```
7.  Enable verbose logging:
    ```bash
    report-agent --verbose
    ```

---

## Performance & Optimization

* **Parallel Processing**: By default, metrics are processed with 3 parallel workers. Use `--max-workers N` to control concurrency. With 3 workers, 3 metrics complete in roughly the time of the slowest single metric.
* **Prompt Optimization**: Prompts have been optimized to reduce token usage by ~30-40% while maintaining all functionality, resulting in lower API costs.
* **Cost Efficiency**: Optimized prompts combined with parallel processing provide faster execution and reduced costs per report.

---

## Troubleshooting

* **ClickHouse UNKNOWN_IDENTIFIER date**: Mark snapshot tables as `kind: snapshot` in `metrics.yml` so they don't get a `WHERE date` filter.
* **Plots not generated**: Plots are generated when possible but not guaranteed due to Code Interpreter limitations. Reports are complete and useful even without plots. You'll see a warning message if plots are missing for a metric.
* **Plots not visible in HTML**: Check that PNGs exist in `reports/plots/` and that you open the HTML from the same directory tree (paths are relative).
* **No weekly report**: Ensure you didn't pass `--no-summary` and that at least one metric completed successfully.
* **Parallel processing issues**: If you encounter issues with parallel processing, try running with `--max-workers 1` for sequential execution.

---

## Roadmap (short)

* More metric kinds and templates (e.g. funnels, distributions).
* Slack / email delivery on schedule.
* Lightweight consistency checks between narrative and numbers.
* Support for additional LLM providers (e.g. Gemini) behind the same connector interface.
* Better Frontend
