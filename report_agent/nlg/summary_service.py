from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
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
from report_agent.metrics.metrics_registry import MetricsRegistry
from report_agent.utils.config_loader import load_configs

_template_dir = files("report_agent.nlg") / "templates"
_env = Environment(loader=FileSystemLoader(str(_template_dir)), autoescape=True)


@dataclass
class MetricDoc:
    description: str
    columns: Dict[str, Dict[str, str]]  # col -> {data_type, description}


def _load_metric_texts(metric_reports: List[Tuple[str, Path]], out_dir: Path) -> Dict[str, str]:
    """
    Read per-metric analytical findings from reports/text/<metric>.txt,
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
    Returns the cleaned weekly report body (Markdown).
    """
    lines = output.splitlines()
    if not lines:
        return output

    lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def _to_posix_relpath(p: Path, start: Path) -> str:
    """
    Return a browser-friendly (POSIX) relative path from start -> p.
    Works across drives (falls back to relpath).
    """
    p = p.resolve()
    start = start.resolve()
    try:
        rel = p.relative_to(start)
    except Exception:
        import os
        rel = Path(os.path.relpath(p, start))
    return rel.as_posix()


def _collect_plots_for_metrics(
    metrics: List[str],
    out_root: Path,
) -> Dict[str, List[str]]:
    """
    Read per-metric plot lists from out_root/plots_index/<metric>.txt.

    Each index file contains one path per line (as written by generate_html_report).
    We normalize each path to be browser-friendly and relative to out_root (reports/).
    """
    plots_index_dir = out_root / "plots_index"
    plots_by_metric: Dict[str, List[str]] = {}

    for metric in metrics:
        paths: List[str] = []
        idx_path = plots_index_dir / f"{metric}.txt"
        if idx_path.exists():
            for line in idx_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                p = Path(line)
                if p.exists():
                    paths.append(_to_posix_relpath(p, out_root))
        plots_by_metric[metric] = paths

    return plots_by_metric


def _render_summary_page(
    summary_html: str,
    highlighted_metrics: List[str],
    per_metric_reports: List[Tuple[str, Path]],
    plots_by_metric: Dict[str, List[str]],
    out_path: Path,
) -> Path:
    """
    Render the final weekly report HTML page, including links to per-metric reports
    and embedding plots for highlighted metrics.
    """
    tpl = _env.get_template("summary_page.html.j2")

    # Get display names for all metrics
    try:
        registry = MetricsRegistry()
    except Exception:
        registry = None
    
    # Map metric -> (display_name, html_filename)
    per_metric_rel = []
    for m, p in per_metric_reports:
        display_name = registry.get_display_name(m) if registry else m
        per_metric_rel.append((m, display_name, p.name))
    
    # Get display names for highlighted metrics
    highlighted_with_names = []
    for metric in highlighted_metrics:
        display_name = registry.get_display_name(metric) if registry else metric
        highlighted_with_names.append((metric, display_name))

    # Get current date for the weekly report header
    report_date = datetime.now().strftime("%B %d, %Y")
    
    html = tpl.render(
        summary_html=summary_html,
        highlighted_metrics=highlighted_with_names,
        per_metric_reports=per_metric_rel,
        plots_by_metric=plots_by_metric,
        generated_at=report_date,
    )
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_weekly_report(
    metric_reports: List[Tuple[str, Path]],
    out_dir: str = "reports",
) -> Path:
    """
    Generate a unified weekly report that synthesizes analytical findings across all metrics.

    metric_reports: list of (metric_name, html_path) for per-metric reports
    out_dir: base reports directory (same used for HTML + plots + text)

    Returns the path to the weekly report HTML (saved as index.html).
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Copy Gnosis logo to reports directory if it exists
    logo_src = Path(__file__).parent.parent.parent / "img" / "Gnosis (1).svg"
    logo_dst = out_root / "img" / "Gnosis (1).svg"
    if logo_src.exists():
        logo_dst.parent.mkdir(exist_ok=True)
        shutil.copy2(logo_src, logo_dst)

    metric_texts = _load_metric_texts(metric_reports, out_root)
    if not metric_texts:
        raise ValueError("No metric analytical findings found; cannot build weekly report.")

    metric_docs = _load_metric_docs(list(metric_texts.keys()))

    prompt = _build_summary_prompt(metric_texts, metric_docs)

    cfg = load_configs()
    api_key = cfg["llm"]["api_key"]
    model_name = cfg["llm"].get("summary_model") or cfg["llm"]["model"]

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a senior data analyst writing a unified weekly report."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    full_output = resp.choices[0].message.content or ""

    highlighted = _parse_highlighted_metrics(full_output)
    summary_markdown = _strip_highlight_header(full_output)

    summary_html = markdown.markdown(summary_markdown, extensions=["extra"])

    # Collect plots for highlighted metrics (use model names for file lookup)
    plots_by_metric = _collect_plots_for_metrics(highlighted, out_root)

    # Save as index.html to make it the main entry point
    summary_path = out_root / "index.html"
    return _render_summary_page(
        summary_html=summary_html,
        highlighted_metrics=highlighted,
        per_metric_reports=metric_reports,
        plots_by_metric=plots_by_metric,
        out_path=summary_path,
    )