from __future__ import annotations

from importlib.resources import files
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from report_agent.config import AppConfig
from report_agent.dbt_context.from_docs_json import load_manifest, get_model_node, get_column_metadata

template_dir = files("report_agent.nlg") / "templates"

env = Environment(
    loader=FileSystemLoader(str(template_dir)),
    autoescape=True,
)


def build_ci_prompt(
    model: str,
    history_days: int,
    csv_filename: str,
    schema_filename: str,
    meta_filename: str,
    config: AppConfig,
    docs_filename: str = None,
    kind: str = "time_series",
    slim_catalog: dict = None,
    pre_fetched_models: dict = None,
) -> str:
    """
    Build the CI prompt for a given model.

    kind:
      - "time_series": weekly trend analysis with plots
      - "snapshot":    one-off KPI snapshot (value + change_pct, etc.)

    slim_catalog: {model_name: description} dict for inline prompt embedding
    pre_fetched_models: Dict mapping model_name -> csv_filename for pre-fetched related models
    config: AppConfig instance for accessing dbt manifest settings
    """
    if kind == "snapshot":
        template_name = "ci_snapshot_prompt.j2"
    else:
        template_name = "ci_report_prompt.j2"

    slim_catalog = slim_catalog or {}

    model_description = slim_catalog.get(model, "")

    if not model_description:
        try:
            cfg_dict = config.to_dict()
            manifest = load_manifest(cfg_dict)
            node = get_model_node(manifest, model)
            if node:
                model_description = node.get("description", "")
        except Exception:
            pass

    template = env.get_template(template_name)
    return template.render(
        model=model,
        model_description=model_description,
        history_days=history_days,
        csv_filename=csv_filename,
        schema_filename=schema_filename,
        meta_filename=meta_filename,
        docs_filename=docs_filename,
        kind=kind,
        slim_catalog=slim_catalog,
        pre_fetched_models=pre_fetched_models or {},
    )