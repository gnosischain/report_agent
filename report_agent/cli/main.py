import argparse
import os
import sys
from pathlib import Path

from report_agent.connectors.llm.openai import OpenAICodeInterpreterConnector
from report_agent.metrics.metrics_registry import MetricsRegistry
from report_agent.nlg.report_service import generate_html_report
from report_agent.nlg.summary_service import generate_portfolio_summary
from report_agent.utils.config_loader import load_configs, validate_config


def main():
    parser = argparse.ArgumentParser(description="Run metrics reports")
    parser.add_argument("--metric", help="Run report only for this metric/model")
    parser.add_argument("--out-dir", default="reports", help="Base output directory")
    parser.add_argument("--no-summary", action="store_true", help="Do not produce a cross-metric summary")
    args = parser.parse_args()

    # Load and validate configuration early
    try:
        cfg = load_configs()
        validate_config(cfg, require_llm=True, require_db=True)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY") or cfg["llm"]["api_key"]
    model_name = cfg["llm"]["model"]

    if not api_key:
        print("ERROR: OPENAI_API_KEY not found. Please set it in your .env file or environment.", file=sys.stderr)
        sys.exit(1)

    try:
        connector = OpenAICodeInterpreterConnector(api_key=api_key, model_name=model_name)
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI connector: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        registry = MetricsRegistry()
    except Exception as e:
        print(f"ERROR: Failed to load metrics registry: {e}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.out_dir)
    out_root.mkdir(exist_ok=True)

    if args.metric:
        models = [args.metric]
    else:
        models = registry.list_models()

    if not models:
        print("WARN: No metrics found in metrics.yml", file=sys.stderr)
        return

    per_metric_html = []
    failed_metrics = []

    # Process each metric with error handling
    for model in models:
        if not registry.has(model):
            print(f"[WARN] Model '{model}' not found in metrics.yml, skipping.")
            failed_metrics.append((model, "Not found in metrics.yml"))
            continue

        print(f"Running report for {model}...")
        try:
            html_path = generate_html_report(
                model=model,
                connector=connector,
                out_dir=str(out_root),
            )
            per_metric_html.append((model, Path(html_path)))
            print(f"  ✓ HTML saved to: {html_path}")
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Stopping report generation...")
            break
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Failed: {error_msg}", file=sys.stderr)
            failed_metrics.append((model, error_msg))
            continue

    # Generate portfolio summary if requested and we have successful reports
    if not args.metric and not args.no_summary and per_metric_html:
        print("\nGenerating portfolio summary...")
        try:
            summary_path = generate_portfolio_summary(
                metric_reports=per_metric_html,
                out_dir=str(out_root),
            )
            print(f"  ✓ Summary HTML saved to: {summary_path}")
        except Exception as e:
            print(f"  ✗ Failed to generate summary: {e}", file=sys.stderr)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Completed: {len(per_metric_html)} successful, {len(failed_metrics)} failed")
    if failed_metrics:
        print("\nFailed metrics:")
        for model, error in failed_metrics:
            print(f"  - {model}: {error}")
    print("=" * 60)