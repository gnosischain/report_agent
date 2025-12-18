# report_agent/connectors/llm/openai.py
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

import httpx
import pandas as pd
from openai import OpenAI

log = logging.getLogger(__name__)

from report_agent.connectors.llm.base import LLMConnector
from report_agent.metrics.metrics_loader import MetricsLoader
from report_agent.metrics.metrics_registry import MetricsRegistry
from report_agent.nlg.prompt_builder import build_ci_prompt
from report_agent.dbt_context.from_docs_json import (
    load_manifest,
    get_model_node,
    get_column_metadata,
    build_model_catalog,
    save_catalog_to_file,
)
from report_agent.utils.config_loader import load_configs


class OpenAICodeInterpreterConnector(LLMConnector):
    """
    OpenAI Responses API + Code Interpreter connector.

    Responsibilities:
      - Fetch raw data for a dbt/ClickHouse model
      - Write CSV + schema + meta (+ optional dbt docs) to a temp dir
      - Upload those files to OpenAI as container files
      - Run a code_interpreter job with instructions + filenames
      - Extract final text and remember artifacts for download
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4.1"):
        super().__init__(api_key, model_name)
        # Disable automatic retries to save credits
        self.client = OpenAI(
            api_key=api_key,
            max_retries=0,  # Disable retries
            http_client=httpx.Client(
                timeout=httpx.Timeout(300.0, connect=10.0),  # 5 min total, 10s connect
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            ),
        )
        self._api_key = api_key  # keep for HTTP fallback downloads
        self.tools = [{"type": "code_interpreter", "container": {"type": "auto"}}]
        self._last_artifacts: Optional[dict] = None
        self._last_dataframe: Optional[pd.DataFrame] = None
        self._last_model_name: Optional[str] = None
        self._last_validation: Optional[dict] = None

    # ---- Tools API (no-op here) ----

    def register_tools(self, functions):
        # No JSON function-calling tools in this connector; code interpreter only.
        self.tools = [{"type": "code_interpreter", "container": {"type": "auto"}}]

    # ---- Helpers ----

    def _df_to_schema_json(self, df: pd.DataFrame) -> dict:
        """
        Build a lightweight schema description for the dataframe, including
        simple role hints that the LLM can use to interpret columns.
        """

        def is_datetime(s: pd.Series) -> bool:
            return pd.api.types.is_datetime64_any_dtype(s) or s.name == "date"

        def is_numeric(s: pd.Series) -> bool:
            return pd.api.types.is_numeric_dtype(s)

        schema = {}
        for col in df.columns:
            s = df[col]
            ex = s.dropna().astype(str).unique()[:3].tolist()

            # Simple role hints for the LLM
            role = "other"
            if col == "date":
                role = "time"
            elif col == "label":
                role = "dimension"
            elif col == "value":
                role = "measure"
            elif col == "change_pct":
                role = "delta"

            schema[col] = {
                "dtype": str(s.dtype),
                "is_datetime": bool(is_datetime(s)),
                "is_numeric": bool(is_numeric(s)),
                "examples": ex,
                "role": role,
            }
        return schema

    # ---- Public API expected by callers ----

    def run_report(self, model_name: str, lookback_days: int | None = None) -> dict:
        """
        High-level entrypoint used by the CLI / report_service.

        - Looks up metric kind (time_series vs snapshot).
        - Fetches the appropriate data slice.
        - Writes CSV/schema/meta/docs to a temp dir.
        - Invokes the Responses API with code_interpreter.
        - Parses structured output and validates significance.
        - Returns dict with narrative, structured data, and validation results.
        
        Returns:
            {
                "narrative": str,
                "structured": dict,
                "validation_status": str ("valid" | "warnings" | "errors"),
                "validation_warnings": list,
            }
        """
        # 1) Load configs/registry and fetch raw data
        cfg = load_configs()
        registry = MetricsRegistry()
        loader = MetricsLoader()

        kind = registry.get_kind(model_name)  # "time_series" or "snapshot"

        if kind == "time_series":
            history = int(lookback_days or registry.get_history_days(model_name))
            df = loader.fetch_time_series(model_name, lookback_days=history)
        else:
            # Snapshots: ignore lookback_days/history; just fetch the snapshot table
            history = 0
            df = loader.fetch_snapshot(model_name)

        if df is None or df.empty:
            return {
                "narrative": f"No data returned for model '{model_name}'",
                "structured": {},
                "validation_status": "errors",
                "validation_warnings": [f"No data returned for model '{model_name}'"],
            }

        # For time series, we require a date column; for snapshots we allow tables without date
        if kind == "time_series" and "date" not in df.columns:
            return {
                "narrative": f"Model '{model_name}' has no 'date' column; cannot proceed.",
                "structured": {},
                "validation_status": "errors",
                "validation_warnings": [f"Model '{model_name}' has no 'date' column"],
            }

        # Store the dataframe for potential reuse (e.g., saving CSV without re-fetching)
        self._last_dataframe = df.copy()  # Store a copy to avoid mutations
        self._last_model_name = model_name

        # 2) Prepare files (CSV + schema + meta + optional docs)
        tmpdir = tempfile.mkdtemp(prefix=f"{model_name.replace('.', '_')}_")
        csv_path = os.path.join(tmpdir, f"{model_name}.csv")
        schema_path = os.path.join(tmpdir, f"{model_name}.schema.json")
        meta_path = os.path.join(tmpdir, f"{model_name}.meta.json")
        docs_path: Optional[str] = None

        # CSV
        df.to_csv(csv_path, index=False)

        # Schema
        schema_json = self._df_to_schema_json(df)
        Path(schema_path).write_text(json.dumps(schema_json, indent=2))

        # Meta (include kind + optional date range)
        if "date" in df.columns:
            date_min = str(pd.to_datetime(df["date"], errors="coerce").min())
            date_max = str(pd.to_datetime(df["date"], errors="coerce").max())
        else:
            date_min = None
            date_max = None

        meta = {
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
            "columns": list(df.columns),
            "date_min": date_min,
            "date_max": date_max,
            "history_days": history if kind == "time_series" else None,
            "kind": kind,
        }
        Path(meta_path).write_text(json.dumps(meta, indent=2))

        # Optional: dbt docs (best-effort)
        docs_filename = None
        try:
            manifest = load_manifest(cfg)
            node = get_model_node(manifest, model_name)
            col_meta = get_column_metadata(node)
            lines = [
                f"# {model_name}",
                (node.get("description", "") or "").strip(),
                "\n## Columns:",
            ]
            for col, info in (col_meta or {}).items():
                lines.append(
                    f"- **{col}** ({info.get('data_type')}): {info.get('description')}"
                )
            docs_path = os.path.join(tmpdir, f"{model_name}.docs.md")
            Path(docs_path).write_text("\n".join([l for l in lines if l]))
            docs_filename = os.path.basename(docs_path)
        except Exception:
            # Silent fallback; schema/meta/CSV are enough
            pass

        # Build model catalog (optional context for LLM)
        # Note: General insight models are not included in per-metric reports - they focus on single metric analysis
        catalog_filename = None
        pre_fetched_models = {}  # Empty - not used for per-metric reports
        catalog = {}
        try:
            # Build model catalog for model_catalog.json (optional context for LLM)
            catalog = build_model_catalog(cfg)
            if catalog:
                catalog_path = os.path.join(tmpdir, "model_catalog.json")
                save_catalog_to_file(catalog, catalog_path)
                catalog_filename = "model_catalog.json"
        except Exception as e:
            # Log but don't fail - catalog is optional
            log.warning(f"Could not build model catalog: {e}")

        # 3) Build the free-form prompt (template depends on kind)
        prompt = build_ci_prompt(
            model=model_name,
            kind=kind,
            history_days=history,
            csv_filename=os.path.basename(csv_path),
            schema_filename=os.path.basename(schema_path),
            meta_filename=os.path.basename(meta_path),
            docs_filename=docs_filename,
            has_catalog=bool(catalog_filename),
            pre_fetched_models={},  # Empty - general insight models not used for per-metric reports
            catalog=catalog if catalog_filename else None,
        )

        # 4) Upload files and collect file_ids for the container
        file_ids: List[str] = []
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
        
        # Upload catalog if available
        if catalog_filename:
            catalog_path = os.path.join(tmpdir, catalog_filename)
            with open(catalog_path, "rb") as f_catalog:
                catalog_file = self.client.files.create(file=f_catalog, purpose="assistants")
                file_ids.append(catalog_file.id)
        
        # Note: General insight models are not used for per-metric reports
        
        # 5) Create the response with a code_interpreter container that includes these files
        tools = [
            {
                "type": "code_interpreter",
                "container": {
                    "type": "auto",
                    "file_ids": file_ids,
                },
            }
        ]

        instructions = (
            "You are a data analyst. Always use the python tool to load the attached files. "
            "Follow the task specification given in the input."
        )

        try:
            resp = self.client.responses.create(
            model=self.model_name,
            tools=tools,
            tool_choice="required",
            max_tool_calls=8,
            parallel_tool_calls=False,
            instructions=instructions,
            input=prompt,
            temperature=0.2,
        )
        except Exception as e:
            # If API call fails, return error dict instead of raising
            error_msg = str(e)
            log.error(f"API call failed for {model_name}: {error_msg}")
            return {
                "narrative": f"Error: {error_msg}",
                "structured": {},
                "validation_status": "errors",
                "validation_warnings": [f"API call failed: {error_msg}"],
            }

        # Collect container artifacts (citations)
        try:
            self._last_artifacts = self._extract_container_artifacts(resp)
        except Exception:
            self._last_artifacts = None

        # 6) Parse structured output and validate
        narrative, structured_data = self._parse_structured_output(resp)
        
        # 7) Validate significance assessment
        validation_result = self._validate_significance(narrative, structured_data, df)
        self._last_validation = validation_result
        
        return {
            "narrative": narrative,
            "structured": structured_data,
            "validation_status": validation_result["status"],
            "validation_warnings": validation_result.get("warnings", []),
        }

    # ---------- Artifacts (citations) ----------

    def _extract_container_artifacts(self, resp) -> dict | None:
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
                                    artifacts["files"].append(
                                        {
                                            "container_id": cid,
                                            "file_id": fid,
                                            "filename": fname,
                                        }
                                    )
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
                                    artifacts["files"].append(
                                        {
                                            "container_id": cid,
                                            "file_id": fid,
                                            "filename": fname,
                                        }
                                    )

        artifacts["container_ids"] = list(seen_containers)
        return artifacts

    def get_last_artifacts(self):
        """Return artifacts collected from the most recent run (or None)."""
        return getattr(self, "_last_artifacts", None)

    def get_last_dataframe(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Return the dataframe from the most recent run if it matches the model.
        
        This allows callers to reuse the dataframe that was already fetched for LLM processing,
        avoiding duplicate database queries.
        
        Args:
            model_name: The model name to check against
            
        Returns:
            DataFrame if available and matches model, None otherwise
        """
        if getattr(self, "_last_model_name", None) == model_name:
            df = getattr(self, "_last_dataframe", None)
            if df is not None:
                return df.copy()  # Return a copy to avoid mutations
        return None

    def get_last_validation(self) -> Optional[dict]:
        """Return validation results from the most recent run (or None)."""
        return getattr(self, "_last_validation", None)

    # ---------- Structured Output Parsing & Validation ----------

    def _parse_structured_output(self, resp) -> tuple[str, dict]:
        """
        Extract JSON and narrative from LLM response.
        
        Returns:
            tuple: (narrative_text, structured_dict)
        """
        # Try to get text from response
        text = getattr(resp, "output_text", None)
        if not text or not isinstance(text, str):
            # Try alternative extraction methods
            try:
                parts: List[str] = []
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
                    text = "\n".join(p for p in parts if p)
            except Exception:
                pass
        
        if not text:
            text = str(resp)
        
        # Try to extract JSON block
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                structured = json.loads(json_match.group(1))
                narrative = text.replace(json_match.group(0), "").strip()
                return narrative, structured
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to find JSON anywhere (without code block markers)
        json_match = re.search(r'\{[^{}]*"significance"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                structured = json.loads(json_match.group(0))
                narrative = text.replace(json_match.group(0), "").strip()
                return narrative, structured
            except json.JSONDecodeError:
                pass
        
        # No structured output found - return text as narrative
        return text, {}

    def _validate_significance(self, narrative: str, structured: dict, df: pd.DataFrame) -> dict:
        """
        Validate that significance assessment is justified by the data.
        Prevents over-interpretation.
        
        Returns:
            dict with status ("valid" | "warnings" | "errors") and warnings list
        """
        warnings = []
        errors = []
        
        if not structured:
            return {"status": "errors", "warnings": ["No structured output found"], "errors": ["No structured output found"]}
        
        significance = structured.get("significance", "").upper()
        stat_evidence = structured.get("statistical_evidence", {})
        
        # Calculate actual statistics from data
        if "date" in df.columns and "value" in df.columns:
            try:
                # Aggregate by week
                df_copy = df.copy()
                df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
                df_copy = df_copy.dropna(subset=["date"])
                
                if len(df_copy) > 0:
                    df_copy["week"] = df_copy["date"].dt.to_period("W")
                    
                    # Handle multiple rows per date (aggregate by date first, then by week)
                    if "label" in df_copy.columns:
                        # Group by date and label, sum values, then group by week
                        daily = df_copy.groupby(["date", "label"])["value"].sum().reset_index()
                        daily["week"] = daily["date"].dt.to_period("W")
                        weekly = daily.groupby("week")["value"].sum().reset_index()
                    else:
                        daily = df_copy.groupby("date")["value"].sum().reset_index()
                        daily["week"] = daily["date"].dt.to_period("W")
                        weekly = daily.groupby("week")["value"].sum().reset_index()
                    
                    if len(weekly) >= 4:
                        last_week = weekly.iloc[-1]["value"]
                        prev_week = weekly.iloc[-2]["value"] if len(weekly) >= 2 else None
                        four_week_values = weekly.iloc[-4:]["value"]
                        four_week_avg = four_week_values.mean()
                        four_week_std = four_week_values.std()
                        
                        if prev_week and four_week_std > 0:
                            wow_change_pct = ((last_week - prev_week) / prev_week) * 100 if prev_week != 0 else 0
                            std_devs_away = (last_week - four_week_avg) / four_week_std
                            
                            # Validate significance assessment
                            if significance == "HIGH":
                                # HIGH should have strong evidence
                                if abs(wow_change_pct) < 15:
                                    warnings.append(f"HIGH significance but only {wow_change_pct:.1f}% change (expected >15%)")
                                if abs(std_devs_away) < 2:
                                    warnings.append(f"HIGH significance but only {std_devs_away:.1f} std devs from avg (expected >2)")
                            
                            if significance == "MEDIUM":
                                # MEDIUM should have some evidence
                                if abs(wow_change_pct) < 5:
                                    warnings.append(f"MEDIUM significance but only {wow_change_pct:.1f}% change (expected >5%)")
                            
                            # Check if within normal variation
                            if abs(std_devs_away) < 1 and significance in ["HIGH", "MEDIUM"]:
                                warnings.append(f"Significance {significance} but change is within normal variation (±1 std dev)")
                            
                            # Check if it's trend continuation (not unusual)
                            if len(weekly) >= 3:
                                trend_direction = "increasing" if weekly.iloc[-1]["value"] > weekly.iloc[-2]["value"] else "decreasing"
                                prev_trend = "increasing" if weekly.iloc[-2]["value"] > weekly.iloc[-3]["value"] else "decreasing"
                                if trend_direction == prev_trend and significance == "HIGH":
                                    warnings.append(f"HIGH significance but change continues existing {trend_direction} trend (may not be unusual)")
            except Exception as e:
                # Don't fail validation on calculation errors, just log
                log.debug(f"Error calculating validation statistics: {e}")
        
        # Check significance is present
        if not significance or significance not in ["HIGH", "MEDIUM", "LOW", "NONE"]:
            errors.append("Missing or invalid significance assessment (must be HIGH/MEDIUM/LOW/NONE)")
        
        status = "errors" if errors else ("warnings" if warnings else "valid")
        return {
            "status": status,
            "warnings": warnings,
            "errors": errors,
        }

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
        sdk_retrieve = getattr(sdk_cf, "retrieve_content", None) if sdk_cf else None
        sdk_content = getattr(sdk_cf, "content", None) if sdk_cf else None

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
                        content_bytes = (
                            getattr(resp, "read", None)
                            and resp.read()
                            or getattr(resp, "content", None)
                        )
                    elif callable(sdk_content):
                        resp = sdk_content(container_id=cid, file_id=fid)
                        content_bytes = (
                            getattr(resp, "read", None)
                            and resp.read()
                            or getattr(resp, "content", None)
                        )
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