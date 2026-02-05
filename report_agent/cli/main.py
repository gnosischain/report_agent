import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import json

from report_agent.config import get_config, AppConfig
from report_agent.connectors.db.clickhouse_connector import ClickHouseConnector
from report_agent.metrics.metrics_loader import MetricsLoader
from report_agent.metrics.metrics_registry import MetricsRegistry
from report_agent.pipeline import ReportPipeline
from report_agent.pipeline.stages import OpenAIAnalyzer
from report_agent.pipeline.stages.data_fetcher import DataFetcher
from report_agent.pipeline.stages.context_builder import ContextBuilder
from report_agent.pipeline.exceptions import PipelineError
from report_agent.nlg.cross_metric_service import generate_cross_metric_analysis
from report_agent.nlg.report_service import generate_html_report
from report_agent.nlg.summary_service import generate_weekly_report


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
        config = get_config()
        config.validate(require_llm=True, require_db=True)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract LLM settings from typed config
    api_key = config.llm.api_key
    openai_model_name = config.llm.model

    # Initialize shared dependencies (composition root)
    # NOTE: Only registry is shared - it's read-only after initialization.
    # ClickHouseConnector and MetricsLoader are created per-thread because
    # clickhouse-connect doesn't support concurrent queries on the same client.
    try:
        registry = MetricsRegistry()
    except Exception as e:
        print(f"ERROR: Failed to initialize metrics registry: {e}", file=sys.stderr)
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
        Process a single metric report using the pipeline architecture.
        Returns: (metric_name, html_path_or_none, error_message_or_none)
        """
        # Create a new pipeline instance for this metric
        # - Shared: config (immutable), registry (read-only after init)
        # - Per-thread: db, loader (clickhouse-connect requires separate clients per thread)
        # - Per-thread: analyzer (each gets its own OpenAI client)
        try:
            # Create per-thread database connection
            # clickhouse-connect doesn't support concurrent queries on the same client
            db = ClickHouseConnector(config=config.clickhouse)
            loader = MetricsLoader(db=db, registry=registry)
            
            analyzer = OpenAIAnalyzer(api_key=api_key, model_name=openai_model_name)
            data_fetcher = DataFetcher(registry=registry, loader=loader)
            context_builder = ContextBuilder(config=config)
            pipeline = ReportPipeline(
                llm_analyzer=analyzer,
                data_fetcher=data_fetcher,
                context_builder=context_builder,
            )
        except ConnectionError as e:
            return (metric_name, None, f"Database connection failed: {e}")
        except Exception as e:
            return (metric_name, None, f"Failed to initialize pipeline: {e}")
        
        try:
            html_path = generate_html_report(
                model=metric_name,
                pipeline=pipeline,
                out_dir=str(out_root),
            )
            return (metric_name, Path(html_path), None)
        except PipelineError as e:
            return (metric_name, None, str(e))
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

    # Collect structured findings and data files for cross-metric analysis
    structured_findings = {}
    metric_data_files = {}
    
    for metric_name, html_path in per_metric_html:
        # Load structured findings
        structured_path = out_root / "structured" / f"{metric_name}.json"
        if structured_path.exists():
            try:
                structured_data = json.loads(structured_path.read_text(encoding="utf-8"))
                structured_findings[metric_name] = structured_data
            except Exception:
                pass
        
        # Track data files
        data_path = out_root / "data" / f"{metric_name}.csv"
        if data_path.exists():
            metric_data_files[metric_name] = str(data_path)
    
    # Perform cross-metric analysis (only if we have multiple metrics and not single-metric mode)
    if not args.metric and len(per_metric_html) > 1 and len(structured_findings) >= 2:
        print("\nPerforming cross-metric analysis...")
        try:
            cross_insights = generate_cross_metric_analysis(
                metric_findings=structured_findings,
                metric_data_files=metric_data_files,
                out_dir=str(out_root),
            )
            print(f"  ✓ Cross-metric insights saved")
        except Exception as e:
            print(f"  ✗ Cross-metric analysis failed: {e}", file=sys.stderr)
            logging.exception("Cross-metric analysis error")
    
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
