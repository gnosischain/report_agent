from jinja2 import Environment, FileSystemLoader
from report_agent.dbt_context.from_docs_json import load_manifest, get_model_node, get_column_metadata
from report_agent.utils.config_loader import load_configs
from importlib.resources import files

template_dir = files("report_agent.nlg") / "templates"

env = Environment(
    loader=FileSystemLoader(str(template_dir)),
    autoescape=True,
)

def build_prompt(model: str, df_sample, delta, anomalies, alerts) -> str:
    """
    Renders a Jinja template into the LLM prompt.
    Passes in:
      • model_name
      • model_description
      • column_metadata
      • df_sample (as list of dicts)
      • delta, anomalies, alerts dicts
    """
    cfg      = load_configs()
    manifest = load_manifest(cfg)
    node     = get_model_node(manifest, model)
    col_meta = get_column_metadata(node)

    template = env.get_template("weekly_report_prompt.j2")
    return template.render(
        model_name=model,
        model_desc=node["description"],
        columns=col_meta,
        sample=df_sample.to_dict(orient="records"),
        delta=delta,
        anomalies=anomalies,
        alerts=alerts,
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

    template = env.get_template(template_name)
    return template.render(
        model=model,
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