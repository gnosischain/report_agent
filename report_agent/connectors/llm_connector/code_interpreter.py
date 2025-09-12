# report_agent/connectors/llm_connector/code_interpreter.py
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, List

import httpx
import pandas as pd
from openai import OpenAI

from report_agent.connectors.llm_connector.base import LLMConnector
from report_agent.analysis.metrics_loader import MetricsLoader
from report_agent.analysis.metrics_registry import MetricsRegistry
from report_agent.nlg.prompt_builder import build_ci_prompt
from report_agent.dbt_context.from_docs_json import (
    load_manifest,
    get_model_node,
    get_column_metadata,
)
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
        self._api_key = api_key  # keep for HTTP fallback downloads
        # Default tools (we rebuild per-call to inject file_ids)
        self.tools = [{"type": "code_interpreter", "container": {"type": "auto"}}]
        self.fn_map = {}
        self._last_artifacts = None

    def register_tools(self, functions):
        # No JSON function-calling tools in this connector; code interpreter only.
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
        # 1) Load configs/registry and fetch raw data (enough history, raw rows)
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

        # 2) Prepare files (CSV + schema + meta + optional docs)
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

        # Optional: dbt docs (best-effort)
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

        # 3) Build the free-form prompt
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

        # Strong nudge: must use python; complete all steps now
        instructions = (
            "You are a data analyst. Always use the python tool to load the attached files and execute your analysis. "
            "Perform all verification silently. Do not include logs, code, or step-by-step narration. Output only the "
            "final BD-facing report as specified, and produce exactly two supporting figures: (1) Headline weekly "
            "total trend with the last two weeks highlighted and WoW annotation, (2) Last week vs prior week grouped "
            "bars for top 5 movers by absolute change. Save to plots/{{model}}_headline_weekly.png and "
            "plots/{{model}}_top5_wow.png and display both so they are cited."
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

        # Collect container artifacts (citations)
        try:
            self._last_artifacts = self._extract_container_artifacts(resp)
        except Exception:
            self._last_artifacts = None

        # 6) Return final text (SDK versions differ; handle a few shapes safely)
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        # Fallback: try to gather any text parts
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

        # Last resort
        return str(resp)

    # ---------- Artifacts (citations) ----------

    def _extract_container_artifacts(self, resp):
        """
        Walk the Responses API output and collect any container file citations.
        Returns a dict: {"container_ids": [...], "files": [{"container_id","file_id","filename"}]}
        """
        artifacts = {"container_ids": [], "files": []}
        seen_containers = set()

        output = getattr(resp, "output", None)
        if isinstance(output, list):
            for item in output:
                # Dict-shape
                if isinstance(item, dict) and item.get("type") == "message":
                    for content in (item.get("content") or []):
                        anns = content.get("annotations") or []
                        for a in anns:
                            if a.get("type") == "container_file_citation":
                                cid = a.get("container_id")
                                fid = a.get("file_id")
                                fname = a.get("filename")
                                if cid and cid not in seen_containers:
                                    seen_containers.add(cid)
                                if cid and fid:
                                    artifacts["files"].append({
                                        "container_id": cid,
                                        "file_id": fid,
                                        "filename": fname,
                                    })
                # Attr-shape
                elif hasattr(item, "content"):
                    for content in (item.content or []):
                        anns = getattr(content, "annotations", None) or []
                        for a in anns:
                            if getattr(a, "type", None) == "container_file_citation":
                                cid = getattr(a, "container_id", None)
                                fid = getattr(a, "file_id", None)
                                fname = getattr(a, "filename", None)
                                if cid and cid not in seen_containers:
                                    seen_containers.add(cid)
                                if cid and fid:
                                    artifacts["files"].append({
                                        "container_id": cid,
                                        "file_id": fid,
                                        "filename": fname,
                                    })

        artifacts["container_ids"] = list(seen_containers)
        return artifacts

    def get_last_artifacts(self):
        """Return artifacts collected from the most recent run (or None)."""
        return getattr(self, "_last_artifacts", None)

    # ---------- Downloads ----------

    def _get_base_url(self) -> str:
        """
        Return a usable base URL string for raw HTTP calls, honoring OPENAI_BASE_URL if set.
        Handles httpx.URL objects from the SDK.
        """
        env_base = os.getenv("OPENAI_BASE_URL")
        if env_base:
            return env_base.rstrip("/")

        base = getattr(self.client, "base_url", None)
        if base is None:
            return "https://api.openai.com/v1"
        # base can be an httpx.URL; stringify safely
        base_str = str(base)
        return base_str.rstrip("/")

    def download_artifacts(
        self,
        output_dir: str = "reports/plots",
        include_extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".csv", ".json", ".md"),
    ) -> List[str]:
        """
        Download container-generated files from the last run to local disk.

        Returns a list of saved file paths (or 'ERROR:<filename>:<exc>' strings on failure).

        Strategy:
        1) Prefer SDK container-files retrieval if available.
        2) Fallback to direct HTTP GET:
           GET {base_url}/containers/{container_id}/files/{file_id}/content
        """
        arts = self.get_last_artifacts()
        if not arts or not arts.get("files"):
            return []

        os.makedirs(output_dir, exist_ok=True)
        saved: List[str] = []

        base_url = self._get_base_url()
        api_key = os.getenv("OPENAI_API_KEY") or self._api_key or ""
        org = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")

        # Try to discover SDK method names (versions differ)
        sdk_cf = getattr(self.client, "container_files", None)
        sdk_retrieve = getattr(sdk_cf, "retrieve_content", None)
        sdk_content = getattr(sdk_cf, "content", None)

        for f in arts["files"]:
            cid = f.get("container_id")
            fid = f.get("file_id")
            fname = f.get("filename") or (fid + ".bin")

            # filter by extension if provided
            if include_extensions:
                try:
                    low = fname.lower()
                    if not any(low.endswith(ext) for ext in include_extensions):
                        continue
                except Exception:
                    pass

            out_path = Path(output_dir) / fname
            try:
                content_bytes = None

                # 1) Prefer SDK method (if present in this version)
                try:
                    if callable(sdk_retrieve):
                        resp = sdk_retrieve(container_id=cid, file_id=fid)
                        # Might return Response-like object; try common attributes
                        content_bytes = getattr(resp, "read", None) and resp.read() or getattr(resp, "content", None)
                    elif callable(sdk_content):
                        resp = sdk_content(container_id=cid, file_id=fid)
                        content_bytes = getattr(resp, "read", None) and resp.read() or getattr(resp, "content", None)
                except Exception:
                    content_bytes = None  # fall through to HTTP

                # 2) Raw HTTP fallback
                if content_bytes is None:
                    url = f"{base_url}/containers/{cid}/files/{fid}/content"
                    headers = {"Authorization": f"Bearer {api_key}"}
                    if org:
                        headers["OpenAI-Organization"] = org
                    with httpx.Client(timeout=60.0) as http:
                        r = http.get(url, headers=headers)
                        r.raise_for_status()
                        content_bytes = r.content

                Path(out_path).write_bytes(content_bytes)
                saved.append(str(out_path))
            except Exception as e:
                saved.append(f"ERROR:{fname}:{e}")

        return saved