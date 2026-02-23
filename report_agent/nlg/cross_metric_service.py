"""
Cross-metric analysis service.

Discovers relationships and patterns across multiple metrics using LLM analysis.
Only analyzes metrics that were already processed (from metrics.yml).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

from report_agent.config import AppConfig
from report_agent.connectors.llm import create_code_execution_client
from report_agent.dbt_context.from_docs_json import (
    build_model_catalog,
    save_catalog_to_file,
)
from report_agent.utils.cost_tracker import get_cost_tracker

log = logging.getLogger(__name__)


def generate_cross_metric_analysis(
    metric_findings: Dict[str, dict],  # metric_name -> {narrative, structured, validation_status}
    metric_data_files: Dict[str, str],  # metric_name -> path to CSV
    config: AppConfig,
    out_dir: str = "reports",
) -> dict:
    """
    Perform cross-metric correlation analysis.
    
    Model receives:
    - All metrics' structured findings
    - All metrics' raw data (CSV files)
    - Full catalog (for discovery only, no data fetching)
    
    Model discovers relationships and patterns.
    
    Args:
        metric_findings: Dict mapping metric names to their analysis results
        metric_data_files: Dict mapping metric names to CSV file paths
        config: AppConfig instance
        out_dir: Output directory for saving results
        
    Returns:
        Dict with cross-metric insights
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    if not metric_findings or len(metric_findings) < 2:
        log.info("Skipping cross-metric analysis: need at least 2 metrics")
        return {}
    
    # Prepare all data files for LLM
    tmpdir = tempfile.mkdtemp(prefix="cross_metric_")
    
    try:
        # 1. Create findings summary file
        findings_path = os.path.join(tmpdir, "metric_findings.json")
        with open(findings_path, "w", encoding="utf-8") as f:
            json.dump(metric_findings, f, indent=2)
        
        # 2. Copy all metric CSVs
        data_files = {}
        for metric_name, csv_path in metric_data_files.items():
            if not Path(csv_path).exists():
                log.warning(f"CSV file not found for {metric_name}: {csv_path}")
                continue
            dest_path = os.path.join(tmpdir, f"{metric_name}.csv")
            shutil.copy2(csv_path, dest_path)
            data_files[metric_name] = os.path.basename(dest_path)
        
        if not data_files:
            log.warning("No data files available for cross-metric analysis")
            return {}
        
        # 3. Build catalog
        cfg_dict = config.to_dict()
        catalog = {}
        try:
            catalog = build_model_catalog(cfg_dict)
        except Exception as e:
            log.warning(f"Could not build model catalog: {e}")
        
        catalog_path = os.path.join(tmpdir, "model_catalog.json")
        if catalog:
            save_catalog_to_file(catalog, catalog_path)
        
        # 4. Build prompt
        analyzed_metrics = list(metric_findings.keys())
        prompt = _build_cross_metric_prompt(metric_findings, data_files, catalog, analyzed_metrics)
        
        # 5. Collect all file paths to upload
        upload_paths: List[str] = [findings_path]
        for metric_name, filename in data_files.items():
            upload_paths.append(os.path.join(tmpdir, filename))
        if catalog and Path(catalog_path).exists():
            upload_paths.append(catalog_path)
        
        # 6. Run analysis via provider-agnostic code execution
        client, model_name, provider = create_code_execution_client(config.llm)
        
        try:
            if provider == "anthropic":
                analysis_text, input_tokens, output_tokens = _run_anthropic_cross_metric(
                    client, model_name, prompt, upload_paths,
                )
            else:
                analysis_text, input_tokens, output_tokens = _run_openai_cross_metric(
                    client, model_name, prompt, upload_paths,
                )
            
            # Track API usage/cost
            tracker = get_cost_tracker()
            tracker.record_usage(
                category="cross_metric",
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            
            # Parse structured output
            cross_metric_insights = _parse_cross_metric_output(analysis_text)
            
            # Save to file
            insights_path = out_root / "cross_metric_insights.json"
            insights_path.write_text(json.dumps(cross_metric_insights, indent=2), encoding="utf-8")
            
            return cross_metric_insights
        except Exception as e:
            log.error(f"Cross-metric analysis API call failed: {e}")
            return {}
        
    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


def _run_openai_cross_metric(
    client, model_name: str, prompt: str, file_paths: List[str],
) -> tuple[str, int, int]:
    """Run cross-metric analysis via OpenAI Code Interpreter."""
    file_ids = []
    for path in file_paths:
        with open(path, "rb") as f:
            file_ids.append(client.files.create(file=f, purpose="assistants").id)

    resp = client.responses.create(
        model=model_name,
        tools=[{
            "type": "code_interpreter",
            "container": {"type": "auto", "file_ids": file_ids},
        }],
        tool_choice="required",
        max_tool_calls=10,
        instructions=(
            "You are a data analyst performing cross-metric correlation analysis. "
            "Use Python to analyze the data."
        ),
        input=prompt,
        temperature=0.2,
    )

    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) or 0 if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0 if usage else 0
    text = getattr(resp, "output_text", None) or str(resp)
    return text, input_tokens, output_tokens


def _run_anthropic_cross_metric(
    client, model_name: str, prompt: str, file_paths: List[str],
) -> tuple[str, int, int]:
    """Run cross-metric analysis via Anthropic Code Execution."""
    _FILES_BETA = "files-api-2025-04-14"
    _CODE_EXEC_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}

    # Upload files via Files API
    file_ids = []
    for path in file_paths:
        with open(path, "rb") as f:
            file_obj = client.beta.files.upload(file=f, betas=[_FILES_BETA])
        file_ids.append(file_obj.id)

    # Build message content
    content: list = [{"type": "text", "text": prompt}]
    for fid in file_ids:
        content.append({"type": "container_upload", "file_id": fid})

    resp = client.beta.messages.create(
        model=model_name,
        betas=[_FILES_BETA],
        max_tokens=16384,
        messages=[{"role": "user", "content": content}],
        tools=[_CODE_EXEC_TOOL],
    )

    # Handle pause_turn
    messages = [{"role": "user", "content": content}]
    for _ in range(8):
        if getattr(resp, "stop_reason", None) != "pause_turn":
            break
        messages.append({"role": "assistant", "content": resp.content})
        container_id = resp.container.id if hasattr(resp, "container") else None
        kwargs = dict(
            model=model_name, betas=[_FILES_BETA], max_tokens=16384,
            messages=messages, tools=[_CODE_EXEC_TOOL],
        )
        if container_id:
            kwargs["container"] = container_id
        resp = client.beta.messages.create(**kwargs)

    # Extract only final text (after last tool-result block)
    content = resp.content or []
    last_tool_idx = -1
    for i, item in enumerate(content):
        itype = getattr(item, "type", None)
        if itype in ("bash_code_execution_tool_result",
                      "text_editor_code_execution_tool_result",
                      "server_tool_use"):
            last_tool_idx = i

    parts = []
    for i, item in enumerate(content):
        if getattr(item, "type", None) == "text" and i > last_tool_idx:
            parts.append(getattr(item, "text", ""))
    text = "\n".join(p for p in parts if p) or str(resp)

    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) or 0 if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0 if usage else 0
    return text, input_tokens, output_tokens


def _build_cross_metric_prompt(
    metric_findings: Dict[str, dict],
    data_files: Dict[str, str],
    catalog: dict,
    analyzed_metrics: List[str],
) -> str:
    """Build prompt for cross-metric analysis."""
    
    prompt = f"""You are analyzing relationships and patterns across multiple metrics.

## Analyzed Metrics (Data Available)

You have access to data for these metrics only:
{', '.join(analyzed_metrics)}

For each, you have:
- Structured findings (metric_findings.json) - includes significance, key numbers, statistical evidence
- Raw data files: {', '.join(data_files.keys())}

## Model Catalog (For Discovery Only)

The model_catalog.json shows ALL available models in the database. You can use this to:
- Understand what other metrics exist
- Discover potential relationships
- Suggest which metrics should be analyzed together in future reports

**IMPORTANT:** You can only analyze the metrics listed above. The catalog is for discovery and context only - you cannot fetch data for models not in the analyzed list.

## Your Task

Analyze the relationships between metrics and identify ecosystem-level patterns:

1. **Correlation Analysis** (using analyzed metrics only)
   - Which of the analyzed metrics moved together this week?
   - Are there positive or negative correlations?
   - Calculate correlation coefficients if relevant
   - **CRITICAL: Be Conservative** - Only report correlations if they are STRONG (correlation coefficient >0.7 or <-0.7)
   - Weak correlations (<0.5) are likely noise - do NOT report them

2. **Pattern Detection** (within analyzed metrics)
   - Are there ecosystem-wide trends among analyzed metrics?
   - Do certain metric groups behave similarly?
   - Are there contradictory signals?
   - **CRITICAL: Only mark patterns as "high significance" if:**
     * Multiple metrics show strong, unusual changes together
     * The pattern is clearly unusual (not normal variation)
     * There's a plausible explanation

3. **Root Cause Hypotheses**
   - If multiple metrics changed, what might explain it?
   - Are changes consistent with known ecosystem dynamics?

4. **Discovery** (using catalog)
   - Based on the catalog, identify models that might be related to analyzed metrics
   - Suggest which catalog models should be added to metrics.yml for future analysis
   - Explain why they would be valuable to analyze together

**CRITICAL: Avoid False Patterns**
- Don't create narratives from coincidental movements
- Don't infer ecosystem-wide trends from 2-3 metrics
- When in doubt, don't report the correlation
- Multiple metrics moving in same direction doesn't always mean correlation

## Output Format

Provide your analysis as JSON:

```json
{{
  "correlations": [
    {{
      "metric1": "<name from analyzed list>",
      "metric2": "<name from analyzed list>",
      "correlation_type": "positive" | "negative" | "none",
      "strength": "strong" | "moderate" | "weak",
      "correlation_coefficient": <number> | null,
      "evidence": "<description>"
    }}
  ],
  "ecosystem_patterns": [
    {{
      "pattern": "<description>",
      "affected_metrics": ["<metric1>", "<metric2>"],
      "significance": "high" | "medium" | "low"
    }}
  ],
  "contradictions": [
    {{
      "description": "<what contradicts>",
      "metrics_involved": ["<metric1>", "<metric2>"],
      "possible_explanation": "<hypothesis>"
    }}
  ],
  "discovered_relationships": [
    {{
      "analyzed_metric": "<from analyzed list>",
      "catalog_metric": "<from catalog, not analyzed>",
      "reason": "<why they might be related>",
      "suggested_for_future": true,
      "expected_correlation": "positive" | "negative" | "unknown"
    }}
  ],
  "summary": "<overall ecosystem narrative>"
}}
```

Then provide a detailed narrative analysis after the JSON block.
"""
    return prompt


def _parse_cross_metric_output(text: str) -> dict:
    """Parse structured output from cross-metric analysis."""
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to find JSON without code block markers
    json_match = re.search(r'\{[^{}]*"correlations"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # If no structured output found, return raw text
    return {"raw_text": text}

