from jinja2 import Environment, FileSystemLoader
from report_agent.dbt_context.from_docs_json import load_manifest, get_model_node, get_column_metadata
from report_agent.utils.config_loader import load_configs
from importlib.resources import files

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
    docs_filename: str = None,
    kind: str = "time_series",
    has_catalog: bool = False,
    pre_fetched_models: dict = None,
    catalog: dict = None,
) -> str:
    """
    Build the CI prompt for a given model.

    kind:
      - "time_series": weekly trend analysis with plots
      - "snapshot":    one-off KPI snapshot (value + change_pct, etc.)
    
    pre_fetched_models: Dict mapping model_name -> csv_filename for pre-fetched related models
    catalog: Full model catalog dict (for template to reference model descriptions)
    """
    if kind == "snapshot":
        template_name = "ci_snapshot_prompt.j2"
    else:
        template_name = "ci_report_prompt.j2"

    # Extract model description from catalog or dbt
    model_description = ""
    if catalog and model in catalog:
        model_description = catalog[model].get("description", "")
    
    if not model_description:
        # Fallback: try to load from manifest
        try:
            cfg = load_configs()
            manifest = load_manifest(cfg)
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
        has_catalog=has_catalog,
        pre_fetched_models=pre_fetched_models or {},
        catalog=catalog or {},
    )