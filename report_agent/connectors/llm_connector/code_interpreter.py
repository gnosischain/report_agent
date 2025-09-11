# report_agent/connectors/llm_connector/code_interpreter.py
import json
import os
import tempfile
from pathlib import Path
import pandas as pd
from openai import OpenAI

from report_agent.connectors.llm_connector.base import LLMConnector
from report_agent.analysis.metrics_loader import MetricsLoader
from report_agent.analysis.metrics_registry import MetricsRegistry
from report_agent.nlg.prompt_builder import build_ci_prompt
from report_agent.dbt_context.from_docs_json import load_manifest, get_model_node, get_column_metadata
from report_agent.utils.config_loader import load_configs


class CodeInterpreterConnector(LLMConnector):
    """
    Free-form code-interpreter connector:
    - Attaches raw CSV (+ schema/meta/docs) with enough history via the container file_ids
    - Lets the model decide how to analyze
    - No precomputed KPIs
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4.1"):
        super().__init__(api_key, model_name)
        self.client = OpenAI(api_key=api_key)
        # Default tools (we’ll rebuild per-call to inject file_ids)
        self.tools = [{"type": "code_interpreter", "container": {"type": "auto"}}]
        self.fn_map = {}

    def register_tools(self, functions):
        self.tools = [{"type": "code_interpreter", "container": {"type": "auto"}}]
        self.fn_map = {}

    def _generate(self, *args, **kwargs):
        raise NotImplementedError("Not used in CodeInterpreterConnector")

    def _df_to_schema_json(self, df: pd.DataFrame) -> dict:
        def is_datetime(s: pd.Series) -> bool:
            return pd.api.types.is_datetime64_any_dtype(s) or s.name == "date"

        def is_numeric(s: pd.Series) -> bool:
            return pd.api.types.is_numeric_dtype(s)

        schema = {}
        for col in df.columns:
            s = df[col]
            ex = s.dropna().astype(str).unique()[:3].tolist()
            schema[col] = {
                "dtype": str(s.dtype),
                "is_datetime": bool(is_datetime(s)),
                "is_numeric": bool(is_numeric(s)),
                "examples": ex,
            }
        return schema

    def run_report(self, model_name: str, lookback_days: int = None) -> str:
        # Load configs/registry and fetch raw data (enough history, raw rows)
        cfg = load_configs()
        registry = MetricsRegistry()
        loader = MetricsLoader()
        history = int(lookback_days or registry.get_history_days(model_name))

        df = loader.fetch_time_series(model_name, lookback_days=history)
        if df is None or df.empty:
            return f"No data returned for model '{model_name}' in the last {history} days."

        # Ensure date column is present as promised
        if "date" not in df.columns:
            return f"Model '{model_name}' has no 'date' column; cannot proceed."

        # Prepare files (CSV + schema + meta + docs)
        tmpdir = tempfile.mkdtemp(prefix=f"{model_name.replace('.', '_')}_")
        csv_path = os.path.join(tmpdir, f"{model_name}.csv")
        schema_path = os.path.join(tmpdir, f"{model_name}.schema.json")
        meta_path = os.path.join(tmpdir, f"{model_name}.meta.json")
        docs_path = None

        # CSV
        df.to_csv(csv_path, index=False)

        # Schema
        schema_json = self._df_to_schema_json(df)
        Path(schema_path).write_text(json.dumps(schema_json, indent=2))

        # Meta
        meta = {
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
            "columns": list(df.columns),
            "date_min": str(pd.to_datetime(df["date"], errors="coerce").min()),
            "date_max": str(pd.to_datetime(df["date"], errors="coerce").max()),
            "history_days": history,
        }
        Path(meta_path).write_text(json.dumps(meta, indent=2))

        # dbt docs (best-effort)
        docs_filename = None
        try:
            manifest = load_manifest(cfg)
            node = get_model_node(manifest, model_name)
            col_meta = get_column_metadata(node)
            lines = [f"# {model_name}", node.get("description", "").strip(), "\n## Columns:"]
            for col, info in (col_meta or {}).items():
                lines.append(f"- **{col}** ({info.get('data_type')}): {info.get('description')}")
            docs_path = os.path.join(tmpdir, f"{model_name}.docs.md")
            Path(docs_path).write_text("\n".join([l for l in lines if l]))
            docs_filename = os.path.basename(docs_path)
        except Exception:
            # Silent fallback; schema/meta/CSV are enough
            pass

        prompt = build_ci_prompt(
            model=model_name,
            history_days=history,
            csv_filename=os.path.basename(csv_path),
            schema_filename=os.path.basename(schema_path),
            meta_filename=os.path.basename(meta_path),
            docs_filename=docs_filename,
        )

        # 4) Upload files and collect file_ids for the container
        file_ids = []
        with open(csv_path, "rb") as f_csv:
            csv_file = self.client.files.create(file=f_csv, purpose="assistants")
            file_ids.append(csv_file.id)
        with open(schema_path, "rb") as f_schema:
            schema_file = self.client.files.create(file=f_schema, purpose="assistants")
            file_ids.append(schema_file.id)
        with open(meta_path, "rb") as f_meta:
            meta_file = self.client.files.create(file=f_meta, purpose="assistants")
            file_ids.append(meta_file.id)

        if docs_path:
            with open(docs_path, "rb") as f_docs:
                docs_file = self.client.files.create(file=f_docs, purpose="assistants")
                file_ids.append(docs_file.id)

        # 5) Create the response with a code_interpreter container that includes these files
        tools = [{
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "file_ids": file_ids,
            },
        }]

        instructions = (
            "You are a data analyst. Always use the python tool to load the attached files and execute your "
            "analysis. Do not stop after verifying the data; complete all steps (load, analyze, visualize if useful, "
            "and write the final narrative) within this single response."
        )

        resp = self.client.responses.create(
            model=self.model_name,
            tools=tools,
            tool_choice="required",     # ensure python tool is used
            max_tool_calls=8,           # allow iterative runs if needed
            parallel_tool_calls=False,  # keep it simple/serial
            instructions=instructions,  # global instruction
            input=prompt,               # task-specific guidance + filenames
            temperature=0.2,
        )

        # 6) Return final text (SDK versions differ; handle a few shapes safely)
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        try:
            parts = []
            output = getattr(resp, "output", None)
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict) and item.get("type") == "message":
                        for c in item.get("content", []) or []:
                            if c.get("type") in ("output_text", "text"):
                                parts.append(c.get("text", ""))
                    elif hasattr(item, "content"):
                        for c in (item.content or []):
                            if getattr(c, "type", None) in ("output_text", "text"):
                                parts.append(getattr(c, "text", ""))
            if parts:
                return "\n".join(p for p in parts if p)
        except Exception:
            pass

        # last resort
        return str(resp)