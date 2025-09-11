# Report Agent (code-interpreter)

Concise, modular system to generate **weekly data reports** directly from ClickHouse + dbt docs using OpenAI’s **Code Interpreter** (python tool). The model receives **raw tables** (CSV) plus neutral context (schema, meta, dbt docs), then decides how to analyze, visualize, and summarize — **no precomputed metrics**.

---

## Key Features

- **Free-form analysis**: model runs Python in a sandbox (Responses API + code_interpreter).
- **Raw, long-format input**: supports multiple rows per `date` (e.g., `date, metric, label, value`).
- **Neutral context**: attaches `*.schema.json` (dtypes/examples), `*.meta.json` (coverage/columns), and optional `*.docs.md` (dbt model + columns).
- **No tool lock-in**: we do not restrict to specific analysis functions; the model chooses.
- **Safe defaults**: read-only SQL, no network calls in the sandbox, plots saved to `plots/`.

---

## Repository Layout

```
report_agent/
  analysis/
    agent_tools.py           # (function-calling tools; not used in code-interpreter path)
    analyzer.py              # pure-Python helpers (used in tools path)
    metrics_loader.py        # fetch raw rows from ClickHouse
    metrics_registry.py      # read configs/metrics.yml
    thresholds.py            # (tools path)
  configs/
    metrics.yml              # metric list + history window
  connectors/
    clickhouse_connector.py  # read-only client
    llm_connector/
      base.py                # base LLM connector
      code_interpreter.py    # Responses API, container w/ file_ids, runs full analysis
      openai.py              # (function-calling path)
  dbt_context/
    from_docs_json.py        # optional docs from manifest/catalog
  nlg/
    prompt_builder.py        # builds prompts
    templates/
      ci_report_prompt.j2    # free-form code-interpreter prompt (no precomputation)
  utils/
    config_loader.py         # loads .env and config
```

---

## Requirements

- Python 3.10+ recommended
- `pip install -r requirements.txt`
- Recent OpenAI SDK (`openai>=1.40.0`)

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
```

Define metrics in `report_agent/configs/metrics.yml` (history only — no thresholds needed for interpreter):

```yaml
metrics:
  - model: api_p2p_discv4_clients_daily
    history_days: 180
```

> **Conventions:** Every table has a `date` column. Multiple rows per `date` are allowed (long format).

---

## How It Works (Code Interpreter Path)

1. **Load configs** → metric entry + `history_days`.
2. **Fetch raw rows** from ClickHouse (no aggregation).
3. **Create files**:
   - `{model}.csv` (raw rows)
   - `{model}.schema.json` (dtype flags + small examples)
   - `{model}.meta.json` (row/col counts, date coverage, columns, history_days)
   - `{model}.docs.md` (dbt docs)
4. **Create a response** using OpenAI **Responses API**:
   - `tools=[{"type":"code_interpreter","container":{"type":"auto","file_ids":[...]}}]`
   - `tool_choice="required"`, `input=rendered prompt`
5. **Model executes Python**: loads CSV, infers roles, analyzes, plots to `plots/`, writes final narrative:
   - First line: `Meaningful change this week: YES/NO — <reason>`
6. **Return** the final text (and references to plots saved by the model).

---

## Quick Start (Notebook)

```python
from report_agent.connectors.llm_connector.code_interpreter import CodeInterpreterConnector
from report_agent.utils.config_loader import load_configs
import os

cfg = load_configs()
api_key = os.getenv("OPENAI_API_KEY") or cfg.get("llm", {}).get("api_key")
model_name = (cfg.get("llm", {}).get("model") or "gpt-4.1")

ci = CodeInterpreterConnector(api_key=api_key, model_name=model_name)
report_text = ci.run_report("api_p2p_discv4_clients_daily")
print(report_text)
```

---

## Best Practices

- **History window**: set `history_days` generously (e.g., 180–365) to enable robust trend judgment.
- **Long format**: include all raw columns; let the model pivot/group as needed.
- **Plots**: model saves figures to `plots/` (relative, writable).
- **Docs fallback**: if dbt docs are unavailable, the schema/meta still provide sufficient context.

---

## Troubleshooting

- **“attachments” arg error**: ensure you’re not using `attachments=` with Responses; use `container.file_ids`.
- **“Missing tools[0].container”**: include `{"type":"code_interpreter","container":{"type":"auto"}}`.
- **“Invalid input: expected .pdf”**: don’t pass files as `input_file` items; mount via `container.file_ids`.
- **PermissionError on plots**: use relative `plots/` directory (template already instructs this).
- **No data returned**: verify ClickHouse creds/database/table and `history_days` window.

---

## Roadmap (Short)

- Retrieve container-generated plots to local `reports/plots/` via Containers API.
- Scheduled runs + Slack/Email delivery.
- Multi-metric bundles (e.g., a P2P weekly pack).
- Lightweight evaluation hooks (consistency checks on numbers in narrative).

---