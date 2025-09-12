# report_agent/nlg/report_service.py
from pathlib import Path
from typing import Sequence

from report_agent.analysis.metrics_loader import MetricsLoader
from report_agent.analysis.metrics_registry import MetricsRegistry
from report_agent.nlg.html_report import render_html_report

def generate_html_report(
    model: str,
    connector,                   # CodeInterpreterConnector (duck-typed)
    out_dir: str = "reports",
    plots_subdir: str = "plots",
    data_subdir: str = "data",
) -> str:
    """
    Run the interpreter, download plots, save CSV, and write an HTML report.
    Returns the HTML file path.
    """
    # 1) Run interpreter to get narrative
    narrative = connector.run_report(model)

    # 2) Download any plots the model created
    plots_dir = str(Path(out_dir) / plots_subdir)
    saved = connector.download_artifacts(output_dir=plots_dir)
    image_paths: Sequence[str] = [
        p for p in saved
        if not str(p).startswith("ERROR:")
        and str(p).lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
    ]

    # 3) Save raw CSV for convenience (re-fetching avoids coupling to temp paths)
    reg = MetricsRegistry()
    hist = reg.get_history_days(model)
    df = MetricsLoader().fetch_time_series(model, lookback_days=hist)

    data_dir = Path(out_dir) / data_subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"{model}.csv"
    df.to_csv(data_path, index=False)

    # 4) Render HTML
    html_path = render_html_report(
        model=model,
        narrative_markdown=narrative,
        image_paths=[str(Path(p)) for p in image_paths],
        data_csv_path=str(data_path),
        out_dir=out_dir,
    )
    return html_path