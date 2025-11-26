import argparse
import os
from pathlib import Path

from report_agent.connectors.llm.openai import OpenAICodeInterpreterConnector
from report_agent.metrics.metrics_registry import MetricsRegistry
from report_agent.nlg.report_service import generate_html_report
from report_agent.nlg.summary_service import generate_portfolio_summary  # we'll add this
from report_agent.utils.config_loader import load_configs


def main():
    parser = argparse.ArgumentParser(description="Run metrics reports")
    parser.add_argument("--metric", help="Run report only for this metric/model")
    parser.add_argument("--out-dir", default="reports", help="Base output directory")
    parser.add_argument("--no-summary", action="store_true", help="Do not produce a cross-metric summary")
    args = parser.parse_args()

    cfg = load_configs()
    api_key = os.getenv("OPENAI_API_KEY") or cfg["llm"]["api_key"]
    model_name = cfg["llm"]["model"]

    connector = OpenAICodeInterpreterConnector(api_key=api_key, model_name=model_name)
    registry = MetricsRegistry()

    out_root = Path(args.out_dir)
    out_root.mkdir(exist_ok=True)

    if args.metric:
        models = [args.metric]
    else:
        models = registry.list_models()

    per_metric_html = []

    for model in models:
        if not registry.has(model):
            print(f"[WARN] Model '{model}' not found in metrics.yml, skipping.")
            continue

        print(f"Running report for {model}...")
        html_path = generate_html_report(
            model=model,
            connector=connector,
            out_dir=str(out_root),
        )
        per_metric_html.append((model, Path(html_path)))
        print(f"  HTML saved to: {html_path}")

    if not args.metric and not args.no_summary and per_metric_html:
        print("Generating portfolio summary...")
        summary_path = generate_portfolio_summary(
            metric_reports=per_metric_html,
            out_dir=str(out_root),
        )
        print("Summary HTML saved to:", summary_path)

    print("Done.")