# Report Agent Roadmap (v0)

> **Note:** A sandbox version using the OpenAI Code Interpreter tool is being developed in parallel on the `code‑interpreter` branch.

_A high-level, evolving plan for our AI-powered weekly reporting system. Focus here is on priorities & rationale — exact file layout to follow._

---

## Architecture Overview

1. **Config Loader**  
   Central source of truth for all credentials, endpoints, and metric definitions.  
   **File:** `utils/config_loader.py`

2. **Connectors**  
   - **ClickHouseConnector** (`connectors/clickhouse_connector.py`):  
     Read-only + write clients, TLS support, automatic DataFrame conversion.  
   - **LLMConnector** (`connectors/llm_connector/`):  
     Base class + provider-specific implementations (Gemini, OpenAI, etc.), registers analysis tools via function-calling.

3. **Metadata Layer (`dbt_context/`)**  
   Ingests your dbt model definitions and column docs from `manifest.json` & `catalog.json`.  
   **Adapters:**  
   - `from_docs_json.py`: fetch & parse live GitHub-Pages JSON  

4. **Analysis Layer (`analysis/`)**  
   Your “toolbox” of pure-Python functions that compute metrics, detect anomalies, and summarize trends.  
   - **Registry** (`metrics_registry.py`): loads `configs/metrics.yml` → metric list  
   - **Loader** (`metrics_loader.py`): fetches full DataFrames for each metric  
   - **Analyzer** (`analyzer.py`):  
     - `compute_weekly_delta(df, date_col, value_col, window)`  
     - `detect_anomalies(df, date_col, value_col, z_thresh)`  
     - `moving_average(df, date_col, value_col, window)`  
     - `summary_statistics(df, value_col)`  
     - `compare_periods(df, date_col, value_col, period)`

5. **Orchestration & NLG**  
   Coordinates the end-to-end workflow:  
   - Chooses which metrics to report on  
   - Loads time-series data and metadata  
   - Invokes the analysis toolbox (via LLM function-calling and local execution)  
   - Guides the LLM to generate narratives  
   - Emits structured output for downstream distribution (distribution to be discussed later)

---

## First Milestones

1. **Metadata Layer (dbt_context)**  
   - **from_docs_json**: fetch & parse `manifest.json` (+ `catalog.json`) from dbt-cerebro docs.

2. **Analysis Layer (TBD)**  
   - Define which functions we’ll support (e.g. delta, anomaly, trend summary).  
   - Decide:  
     - Pure Python pre-aggregation, or  
     - LLM-orchestrated multi-step via function-calling.  
   - Sketch inputs/outputs but leave implementation details flexible.

3. **LLM Orchestration (function‑calling)**  
   - Register a suite of analysis tools:  
     - `compute_weekly_delta(...)`  
     - `detect_anomalies(...)`  
     - `summarize_trends(...)`  
     - _[add more as needed]_  
   - Use a **chat loop**:  
     1. Send prompt + tool specs  
     2. LLM calls a tool → we execute & return JSON  
     3. Repeat until LLM returns a final text message  
   - This gives the model **agency** to choose its analysis path while ensuring numeric precision.

4. **History / State (TBD)**  
   - Options:  
     - A `report_history` table in ClickHouse  
     - ...  
   - Goals: record `(run_date, metric, pct_change, alerted)` so we only surface **new** anomalies.

---

## Next Steps

1. **Finalize Analysis Functions**  
   - Define which functions we’ll support  
   - Decide:  
     - Pure Python pre-aggregation, or  
     - LLM-orchestrated multi-step via function-calling.  
   - Sketch inputs/outputs but leave implementation details flexible.

2. **LLM Tool Registration**  
   - Function schemas from docstrings and signatures.  
   - Ensure descriptions clearly explain inputs and outputs.

3. **Prompt Templates**  
   - Jinja templates that blend metadata and data summaries.  
   - Craft prompts that steer the LLM through analysis → narrative.

4. **Chat‑Loop Orchestration**  
   - Implement the function‑calling loop: send prompt + tools → handle calls → feed results → final narrative / Repeat Loop  
   - Log every tool invocation for auditability.

5. **State Management**  
   - Prototype a simple `report_history` store to record past alerts.  
   - Ensure only new or significant changes surface in each run.

---
