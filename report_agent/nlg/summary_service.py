from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from importlib.resources import files
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
import markdown  

from report_agent.dbt_context.from_docs_json import (
    load_manifest,
    get_model_node,
    get_column_metadata,
)
from report_agent.utils.config_loader import load_configs

_template_dir = files("report_agent.nlg") / "templates"
_env = Environment(loader=FileSystemLoader(str(_template_dir)), autoescape=True)


@dataclass
class MetricDoc:
    description: str
    columns: Dict[str, Dict[str, str]]  


def _load_metric_texts(metric_reports: List[Tuple[str, Path]], out_dir: Path) -> Dict[str, str]:
    """
    Read per-metric report text from reports/text/<metric>.txt,
    using the same out_dir as used for HTML reports.
    """
    text_dir = out_dir / "text"
    texts: Dict[str, str] = {}

    for metric, _html_path in metric_reports:
        text_path = text_dir / f"{metric}.txt"
        if text_path.exists():
            texts[metric] = text_path.read_text(encoding="utf-8")
        else:
            continue

    return texts


def _load_metric_docs(metric_names: List[str]) -> Dict[str, MetricDoc]:
    """
    Use dbt manifest helpers to get model + column docs for each metric.
    """
    cfg = load_configs()
    manifest = load_manifest(cfg)

    metric_docs: Dict[str, MetricDoc] = {}

    for metric in metric_names:
        try:
            node = get_model_node(manifest, metric)
            col_meta = get_column_metadata(node) or {}
            columns = {
                col: {
                    "data_type": (info.get("data_type") or ""),
                    "description": (info.get("description") or ""),
                }
                for col, info in col_meta.items()
            }
            metric_docs[metric] = MetricDoc(
                description=(node.get("description") or ""),
                columns=columns,
            )
        except Exception:
            metric_docs[metric] = MetricDoc(description="", columns={})

    return metric_docs


def _build_summary_prompt(metric_texts: Dict[str, str], metric_docs: Dict[str, MetricDoc]) -> str:
    tpl = _env.get_template("summary_prompt.j2")
    docs_for_template = {
        m: {
            "description": d.description,
            "columns": d.columns,
        }
        for m, d in metric_docs.items()
    }
    return tpl.render(metric_reports=metric_texts, metric_docs=docs_for_template)


def _parse_highlighted_metrics(output: str) -> List[str]:
    """
    Parse the first line 'HIGHLIGHTED_METRICS: a, b, c' from the model output.
    Returns the list of metric names; if parsing fails, returns [].
    """
    lines = output.splitlines()
    if not lines:
        return []

    first = lines[0].strip()
    prefix = "HIGHLIGHTED_METRICS:"
    if not first.startswith(prefix):
        return []

    tail = first[len(prefix):].strip()
    if not tail:
        return []

    return [m.strip() for m in tail.split(",") if m.strip()]


def _strip_highlight_header(output: str) -> str:
    """
    Remove the 'HIGHLIGHTED_METRICS: ...' line and any immediately following blank lines.
    Returns the cleaned summary body (Markdown).
    """
    lines = output.splitlines()
    if not lines:
        return output

    lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def _render_summary_page(
    summary_html: str,
    highlighted_metrics: List[str],
    per_metric_reports: List[Tuple[str, Path]],
    out_path: Path,
) -> Path:
    """
    Render the final summary HTML page, including links to per-metric reports
    and embedding plots for highlighted metrics.
    """
    tpl = _env.get_template("summary_page.html.j2")

    per_metric_rel = [(m, p.name) for m, p in per_metric_reports]

    html = tpl.render(
        summary_html=summary_html,
        highlighted_metrics=highlighted_metrics,
        per_metric_reports=per_metric_rel,
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_portfolio_summary(
    metric_reports: List[Tuple[str, Path]],
    out_dir: str = "reports",
) -> Path:
    """
    Generate a cross-metric portfolio summary.

    metric_reports: list of (metric_name, html_path) for per-metric reports
    out_dir: base reports directory (same used for HTML + plots + text)

    Returns the path to the summary HTML.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Load per-metric report texts (from reports/text/<metric>.txt)
    metric_texts = _load_metric_texts(metric_reports, out_root)
    if not metric_texts:
        raise ValueError("No metric report texts found; cannot build portfolio summary.")

    # 2) Load dbt docs for these metrics
    metric_docs = _load_metric_docs(list(metric_texts.keys()))

    # 3) Build the summary prompt
    prompt = _build_summary_prompt(metric_texts, metric_docs)

    # 4) Call LLM (plain chat, no code interpreter needed)
    cfg = load_configs()
    api_key = cfg["llm"]["api_key"]
    model_name = cfg["llm"].get("summary_model") or cfg["llm"]["model"]

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a senior data analyst writing a weekly portfolio summary."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    full_output = resp.choices[0].message.content or ""

    # 5) Parse highlighted metrics and clean body (Markdown)
    highlighted = _parse_highlighted_metrics(full_output)
    summary_markdown = _strip_highlight_header(full_output)

    # 5b) Convert Markdown -> HTML for rendering
    summary_html = markdown.markdown(summary_markdown, extensions=["extra"])

    # 6) Render summary HTML page
    summary_path = out_root / "portfolio_summary.html"
    return _render_summary_page(
        summary_html=summary_html,
        highlighted_metrics=highlighted,
        per_metric_reports=metric_reports,
        out_path=summary_path,
    )