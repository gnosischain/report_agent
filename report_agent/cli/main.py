import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from report_agent.connectors.llm.openai import OpenAICodeInterpreterConnector
from report_agent.metrics.metrics_registry import MetricsRegistry
from report_agent.nlg.report_service import generate_html_report
from report_agent.nlg.summary_service import generate_weekly_report
from report_agent.utils.config_loader import load_configs, validate_config


def main():
    # Configure logging to show INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        stream=sys.stdout
    )
    
    parser = argparse.ArgumentParser(description="Run metrics reports")
    parser.add_argument("--metric", help="Run report only for this metric/model")
    parser.add_argument("--out-dir", default="reports", help="Base output directory")
    parser.add_argument("--no-summary", action="store_true", help="Do not produce a cross-metric summary")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum number of parallel workers (default: 3)")
    args = parser.parse_args()
    
    # Set debug level if verbose flag is used
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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
    openai_model_name = cfg["llm"]["model"]  # OpenAI model name (e.g., "gpt-4.1")

    if not api_key:
        print("ERROR: OPENAI_API_KEY not found. Please set it in your .env file or environment.", file=sys.stderr)
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

    # Filter out invalid models first
    valid_models = []
    invalid_models = []
    for model in models:
        if not registry.has(model):
            invalid_models.append((model, "Not found in metrics.yml"))
        else:
            valid_models.append(model)
    
    if invalid_models:
        for model, error in invalid_models:
            print(f"[WARN] Model '{model}' not found in metrics.yml, skipping.", file=sys.stderr)

    if not valid_models:
        print("ERROR: No valid metrics to process.", file=sys.stderr)
        return

    per_metric_html = []
    failed_metrics = invalid_models.copy()

    def process_single_metric(metric_name: str) -> Tuple[str, Optional[Path], Optional[str]]:
        """
        Process a single metric report.
        Returns: (metric_name, html_path_or_none, error_message_or_none)
        """
        # Create a new connector instance for this metric to avoid artifact conflicts
        # Use openai_model_name (not metric_name) to avoid confusion
        try:
            metric_connector = OpenAICodeInterpreterConnector(api_key=api_key, model_name=openai_model_name)
        except Exception as e:
            return (metric_name, None, f"Failed to initialize connector: {e}")
        
        try:
            html_path = generate_html_report(
                model=metric_name,
                connector=metric_connector,
                out_dir=str(out_root),
            )
            return (metric_name, Path(html_path), None)
        except Exception as e:
            return (metric_name, None, str(e))

    # Process metrics in parallel
    max_workers = min(args.max_workers, len(valid_models))
    if max_workers == 1:
        # Sequential processing (for single metric or --max-workers=1)
        print(f"Processing {len(valid_models)} metric(s) sequentially...")
        for metric in valid_models:
            print(f"Running report for {metric}...")
            metric_name, html_path, error = process_single_metric(metric)
            if error:
                print(f"  ✗ Failed: {error}", file=sys.stderr)
                failed_metrics.append((metric_name, error))
            else:
                per_metric_html.append((metric_name, html_path))
                print(f"  ✓ HTML saved to: {html_path}")
    else:
        # Parallel processing
        print(f"Processing {len(valid_models)} metric(s) with {max_workers} worker(s)...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_metric = {
                executor.submit(process_single_metric, metric): metric 
                for metric in valid_models
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_metric):
                metric = future_to_metric[future]
                try:
                    metric_name, html_path, error = future.result()
                    if error:
                        print(f"  ✗ {metric_name} failed: {error}", file=sys.stderr)
                        failed_metrics.append((metric_name, error))
                    else:
                        per_metric_html.append((metric_name, html_path))
                        print(f"  ✓ {metric_name} completed: {html_path}")
                except KeyboardInterrupt:
                    print("\n[INTERRUPTED] Stopping report generation...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                except Exception as e:
                    print(f"  ✗ {metric} failed with unexpected error: {e}", file=sys.stderr)
                    failed_metrics.append((metric, str(e)))

    # Generate weekly report (saved as index.html) if requested and we have successful reports
    if not args.metric and not args.no_summary and per_metric_html:
        print("\nGenerating weekly report...")
        try:
            summary_path = generate_weekly_report(
                metric_reports=per_metric_html,
                out_dir=str(out_root),
            )
            print(f"  ✓ Weekly report HTML saved to: {summary_path} (main entry point)")
        except Exception as e:
            print(f"  ✗ Failed to generate weekly report: {e}", file=sys.stderr)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Completed: {len(per_metric_html)} successful, {len(failed_metrics)} failed")
    if failed_metrics:
        print("\nFailed metrics:")
        for model, error in failed_metrics:
            print(f"  - {model}: {error}")
    print("=" * 60)