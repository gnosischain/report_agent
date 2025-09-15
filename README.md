# Report Agent (code-interpreter)

Concise, modular system to generate **weekly data reports** directly from ClickHouse + dbt docs using OpenAI’s **Code Interpreter** (python tool). The model receives **raw tables** (CSV) plus neutral context (schema, meta, optional dbt docs), then decides how to analyze, visualize, and summarize — **no precomputed metrics**. The run produces a **self-contained HTML report** with a BD-friendly narrative and plots.

---

## Key Features

- **Free-form analysis**: model runs Python in a sandbox (Responses API + `code_interpreter`), no fixed toolchain.
- **Raw, long-format input**: supports multiple rows per `date` (e.g., `date, metric, label, value`).
- **Neutral context**: attaches `*.schema.json` (dtypes/examples), `*.meta.json` (coverage/columns), and optional `*.docs.md`.
- **Evidence-first visuals**: plots are required to visualize the headline & highlights (weekly total + WoW movers).
- **HTML output**: renders narrative + plot gallery into a portable static HTML page.
- **Safe defaults**: read-only SQL; no network in sandbox; plots saved to a writable relative `plots/` folder.

---

## Repository Layout

```
report_agent/
  analysis/
    agent_tools.py
    analyzer.py
    metrics_loader.py        # fetch raw rows from ClickHouse (no aggregation)
    metrics_registry.py      # reads configs/metrics.yml (history window)
    thresholds.py
  configs/
    metrics.yml              # metric list + history days
  connectors/
    clickhouse_connector.py  # read-only client
    llm_connector/
      base.py
      code_interpreter.py    # Responses API: mounts CSV/schema/meta/docs in container,
                             # runs analysis, captures citations, download_artifacts()
      openai.py
  dbt_context/
    from_docs_json.py        # optional dbt manifest/catalog ingestion
  nlg/
    prompt_builder.py        # builds interpreter prompt
    html_report.py           # render self-contained HTML report
    report_service.py        # one-shot: run -> download plots -> save CSV -> HTML
    templates/
      ci_report_prompt.j2    # BD-style prompt (no process talk; headline + sections; required plots)
      report_page.html.j2    # HTML page template (dark theme, gallery)
  utils/
    config_loader.py         # loads .env and config
```

_Output structure after a run:_
```
reports/
  YYYY-MM-DD_<model>.html
  plots/
    <model>_headline_weekly.png
    <model>_top5_wow.png
    ... (any additional cfile_*.png created)
  data/
    <model>.csv
```

---

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
  - Recent OpenAI SDK (`openai>=1.40.0`)
  - `httpx` (container file downloads)
  - `markdown` (HTML narrative)

---

## Configuration

Set credentials in `.env` (loaded by `utils/config_loader.py`):

```
# OpenAI
OPENAI_API_KEY=...

# ClickHouse
CLICKHOUSE_HOST=...
CLICKHOUSE_PORT=...
CLICKHOUSE_USER=...
CLICKHOUSE_PASSWORD=...
CLICKHOUSE_DATABASE=...

# dbt docs (optional)
DBT_MANIFEST_URL=...
DBT_CATALOG_URL=...

# Custom base (optional; e.g., Azure)
# OPENAI_BASE_URL=https://.../v1
```

Define metrics in `report_agent/configs/metrics.yml` (history only):

```yaml
metrics:
  - model: api_p2p_discv4_clients_daily
    history_days: 180
```

> **Conventions:** Every table has a `date` column. Multiple rows per `date` are allowed (long format).

---

## How It Works

1. **Load configs** → find metric + `history_days`.
2. **Fetch raw rows** from ClickHouse; no precompute, no pivot.
3. **Create files**:
   - `{model}.csv` (raw rows)
   - `{model}.schema.json` (dtype flags + examples)
   - `{model}.meta.json` (row/col counts, date coverage, columns, history_days)
   - `{model}.docs.md` (optional dbt docs)
4. **Start a `code_interpreter` container** with `file_ids` mounted.
5. **Model executes Python**:
   - infers roles, analyzes weekly trends/segments,
   - creates two **headline plots**:
     - weekly total with last two weeks highlighted + WoW annotation
     - WoW top movers by segment (grouped bars, top 5 by abs. change)
   - saves plots to `plots/` and **displays** them (so they’re cited).
6. **Artifacts**:
   - Parse citations → `container_id` + `file_id` for each plot.
   - `download_artifacts()` pulls images to `reports/plots/`.
7. **HTML report**:
   - `render_html_report()` or `generate_html_report()` combines narrative + gallery,
   - links data CSV; all asset paths are relative for local viewing.

---

## Quick Start (concise)

1. `pip install -r requirements.txt` and set `.env`
2. Add your model to `configs/metrics.yml`
3. Run:

```python
from report_agent.connectors.llm_connector.code_interpreter import CodeInterpreterConnector
from report_agent.nlg.report_service import generate_html_report
import os

ci = CodeInterpreterConnector(api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4.1")
print("Saved:", generate_html_report("api_p2p_discv4_clients_daily", ci, out_dir="reports"))
```

---

## Best Practices

- **History window**: ≥180 days recommended for robust trends.
- **Deterministic plot names**: template nudges `<model>_headline_weekly.png` and `<model>_top5_wow.png`.
- **Container lifetime**: download artifacts right after a run (containers expire quickly).

---

## Troubleshooting

- **attachments error**: don’t use `attachments=`; mount files via `tools[0].container.file_ids`.
- **Missing `tools[0].container`**: include `{"type":"code_interpreter","container":{"type":"auto"}}`.
- **“.pdf only” error**: don’t pass `input_file` items for CSV/PNG; use container `file_ids`.
- **Plots not visible in HTML**: ensure downloads to `reports/plots/`; paths are relative in the page.
- **No data returned**: check ClickHouse creds/table and `history_days`.

---

## Roadmap (short)

- Multi-metric bundle in one HTML
- Scheduled runs + Slack/Email delivery
- Lightweight evaluation hooks (consistency checks on numbers in narrative).
- Onchain context

---