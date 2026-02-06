# report_agent/pipeline/stages/context_builder.py
"""
ContextBuilder stage: Prepares all context files and prompts for LLM analysis.

This stage is responsible for:
- Writing CSV, schema, meta files to a temp directory
- Optionally building dbt docs and model catalog
- Building the appropriate prompt template
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from report_agent.config import AppConfig
from report_agent.dbt_context.from_docs_json import (
    load_manifest,
    get_model_node,
    get_column_metadata,
    build_model_catalog,
    save_catalog_to_file,
)
from report_agent.nlg.prompt_builder import build_ci_prompt
from report_agent.pipeline.models import MetricData, AnalysisContext
from report_agent.pipeline.exceptions import ContextBuildError

log = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds all context files and prompts for LLM analysis.
    
    Args:
        config: AppConfig instance with application configuration.
    
    Usage:
        builder = ContextBuilder(config=config)
        context = builder.build(metric_data)
    """
    
    def __init__(self, config: AppConfig):
        self._config = config
        # Convert to dict for backward compatibility with dbt_context functions
        self.cfg = config.to_dict()
    
    def build(self, data: MetricData) -> AnalysisContext:
        """
        Build analysis context from metric data.
        
        Args:
            data: MetricData from the DataFetcher stage
            
        Returns:
            AnalysisContext ready for LLM analysis
            
        Raises:
            ContextBuildError: If context building fails
        """
        model = data.model_name
        df = data.df
        
        try:
            # Create temp directory
            tmpdir = Path(tempfile.mkdtemp(prefix=f"{model.replace('.', '_')}_"))
            
            # Write CSV
            csv_path = tmpdir / f"{model}.csv"
            df.to_csv(csv_path, index=False)
            
            # Write schema
            schema_path = tmpdir / f"{model}.schema.json"
            schema_json = self._df_to_schema_json(df)
            schema_path.write_text(json.dumps(schema_json, indent=2))
            
            # Write meta
            meta_path = tmpdir / f"{model}.meta.json"
            meta = self._build_meta(df, data.kind, data.history_days)
            meta_path.write_text(json.dumps(meta, indent=2))
            
            # Optional: dbt docs
            docs_path = self._build_docs(model, tmpdir)
            
            # Optional: catalog
            catalog_path, catalog = self._build_catalog(tmpdir)
            
            # Build prompt
            prompt = build_ci_prompt(
                model=model,
                kind=data.kind,
                history_days=data.history_days or 0,
                csv_filename=csv_path.name,
                schema_filename=schema_path.name,
                meta_filename=meta_path.name,
                docs_filename=docs_path.name if docs_path else None,
                has_catalog=catalog_path is not None,
                pre_fetched_models={},  # Not used for per-metric reports
                catalog=catalog if catalog_path else None,
                config=self._config,
            )
            
            log.debug(f"Built context for model '{model}' in {tmpdir}")
            
            return AnalysisContext(
                model_name=model,
                kind=data.kind,
                history_days=data.history_days or 0,
                csv_path=csv_path,
                schema_path=schema_path,
                meta_path=meta_path,
                docs_path=docs_path,
                catalog_path=catalog_path,
                prompt=prompt,
                df=df,
                temp_dir=tmpdir,
            )
        except Exception as e:
            raise ContextBuildError(f"Failed to build context for '{model}': {e}", model)
    
    def _df_to_schema_json(self, df: pd.DataFrame) -> dict:
        """
        Build a lightweight schema description for the dataframe,
        including simple role hints that the LLM can use to interpret columns.
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
    
    def _build_meta(self, df: pd.DataFrame, kind: str, history_days: Optional[int]) -> dict:
        """Build metadata dictionary for the LLM."""
        if "date" in df.columns:
            date_min = str(pd.to_datetime(df["date"], errors="coerce").min())
            date_max = str(pd.to_datetime(df["date"], errors="coerce").max())
        else:
            date_min = None
            date_max = None

        return {
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
            "columns": list(df.columns),
            "date_min": date_min,
            "date_max": date_max,
            "history_days": history_days if kind == "time_series" else None,
            "kind": kind,
        }
    
    def _build_docs(self, model: str, tmpdir: Path) -> Optional[Path]:
        """Build dbt docs file (best-effort, returns None on failure)."""
        try:
            manifest = load_manifest(self.cfg)
            node = get_model_node(manifest, model)
            col_meta = get_column_metadata(node)
            lines = [
                f"# {model}",
                (node.get("description", "") or "").strip(),
                "\n## Columns:",
            ]
            for col, info in (col_meta or {}).items():
                lines.append(
                    f"- **{col}** ({info.get('data_type')}): {info.get('description')}"
                )
            docs_path = tmpdir / f"{model}.docs.md"
            docs_path.write_text("\n".join([l for l in lines if l]))
            return docs_path
        except Exception as e:
            log.debug(f"Could not build dbt docs for '{model}': {e}")
            return None
    
    def _build_catalog(self, tmpdir: Path) -> tuple[Optional[Path], dict]:
        """Build model catalog (best-effort, returns (None, {}) on failure)."""
        try:
            catalog = build_model_catalog(self.cfg)
            if catalog:
                catalog_path = tmpdir / "model_catalog.json"
                save_catalog_to_file(catalog, str(catalog_path))
                return catalog_path, catalog
        except Exception as e:
            log.debug(f"Could not build model catalog: {e}")
        return None, {}
