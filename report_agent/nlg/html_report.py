from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from importlib.resources import files

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

    template = _env.get_template("report_page.html.j2")
    now = datetime.now()
    html = template.render(
        title=f"{title_prefix}{model}",
        model=model,
        generated_at=now.strftime("%Y-%m-%d %H:%M"),
        narrative_html=narrative_html,
        image_paths=rel_images,
        data_csv_path=rel_csv,
    )

    out_path = out_dir_p / f"{now.strftime('%Y-%m-%d')}_{model}.html"
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)