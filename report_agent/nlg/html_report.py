import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from jinja2 import Environment, FileSystemLoader, select_autoescape
from importlib.resources import files

from report_agent.metrics.metrics_registry import MetricsRegistry

try:
    import markdown as _md
    _MD_ENABLED = True
except Exception:
    _MD_ENABLED = False

template_dir = files("report_agent.nlg") / "templates"

_env = Environment(
    loader=FileSystemLoader(str(template_dir)),
    autoescape=select_autoescape(["html", "xml"]),
)

def _md_to_html(text: str) -> str:
    if not text:
        return ""
    if _MD_ENABLED:
        return _md.markdown(text, extensions=["extra", "tables", "fenced_code"])
    import html as _html
    return f"<pre>{_html.escape(text)}</pre>"

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
        # fallback if on different drives, etc.
        rel = Path(os.path.relpath(p, start))
    return rel.as_posix()

def render_html_report(
    model: str,
    narrative_markdown: str,
    image_paths,
    data_csv_path: str | None = None,
    out_dir: str = "reports",
    title_prefix: str = "Weekly Report — "
) -> str:
    """
    Render a self-contained HTML report and return its file path.
    All asset paths (images, CSV) are normalized to be relative to the HTML directory.
    """
    out_dir_p = Path(out_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    narrative_html = _md_to_html(narrative_markdown)

    rel_images = []
    for p in (image_paths or []):
        p_path = Path(p)
        if not p_path.exists():
            continue
        rel_images.append(_to_posix_relpath(p_path, out_dir_p))

    rel_csv = None
    if data_csv_path:
        csv_path = Path(data_csv_path)
        if csv_path.exists():
            rel_csv = _to_posix_relpath(csv_path, out_dir_p)

    # Get display name for the model
    try:
        registry = MetricsRegistry()
        display_name = registry.get_display_name(model)
    except Exception:
        display_name = model

    template = _env.get_template("report_page.html.j2")
    now = datetime.now()
    html = template.render(
        title=f"{title_prefix}{display_name}",
        model=model,
        display_name=display_name,
        generated_at=now.strftime("%Y-%m-%d %H:%M"),
        narrative_html=narrative_html,
        image_paths=rel_images,
        data_csv_path=rel_csv,
    )

    out_path = out_dir_p / f"{now.strftime('%Y-%m-%d')}_{model}.html"
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


def generate_index_page(
    out_dir: str = "reports",
    summary_path: str | None = None,
) -> str:
    """
    Generate an index.html page that lists all reports organized by date.
    
    Scans the output directory for HTML report files and creates a navigable index
    with search/filter functionality.
    
    Args:
        out_dir: Base output directory containing HTML reports
        summary_path: Optional path to portfolio summary (relative to out_dir)
    
    Returns:
        Path to the generated index.html file
    """
    out_dir_p = Path(out_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)
    
    # Scan for HTML report files (excluding index.html and portfolio_summary.html)
    report_files: List[Tuple[str, str, datetime]] = []
    for html_file in out_dir_p.glob("*.html"):
        if html_file.name in ("index.html", "portfolio_summary.html"):
            continue
        
        # Parse date and model from filename: YYYY-MM-DD_model.html
        try:
            parts = html_file.stem.split("_", 1)
            if len(parts) == 2:
                date_str, model = parts
                report_date = datetime.strptime(date_str, "%Y-%m-%d")
                report_files.append((model, html_file.name, report_date))
        except (ValueError, IndexError):
            # Skip files that don't match the expected pattern
            continue
    
    # Load registry to get display names
    try:
        registry = MetricsRegistry()
    except Exception:
        registry = None
    
    # Group by date (most recent first) and add display names
    reports_by_date = defaultdict(list)
    for model, filename, report_date in sorted(report_files, key=lambda x: x[2], reverse=True):
        date_key = report_date.strftime("%Y-%m-%d")
        display_name = registry.get_display_name(model) if registry else model
        reports_by_date[date_key].append({
            "model": model,
            "display_name": display_name,
            "filename": filename,
            "date": report_date,
        })
    
    # Sort dates (most recent first)
    sorted_dates = sorted(reports_by_date.keys(), reverse=True)
    
    # Render index template
    template = _env.get_template("index.html.j2")
    html = template.render(
        reports_by_date=reports_by_date,
        sorted_dates=sorted_dates,
        summary_path=summary_path,
        total_reports=len(report_files),
    )
    
    index_path = out_dir_p / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return str(index_path)