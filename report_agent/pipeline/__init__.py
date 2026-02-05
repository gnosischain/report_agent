# report_agent/pipeline/__init__.py
"""
Pipeline architecture for report generation.

This module provides a clean, testable pipeline for generating metric reports:
- DataFetcher: Fetches raw data from ClickHouse
- ContextBuilder: Prepares files and prompts for LLM analysis
- LLMAnalyzer: Runs LLM analysis (abstract base with OpenAI implementation)
- ResultValidator: Validates LLM output against actual data
- ReportPipeline: Orchestrates the full pipeline

Usage:
    from report_agent.pipeline import ReportPipeline
    from report_agent.pipeline.stages import OpenAIAnalyzer

    analyzer = OpenAIAnalyzer(api_key="...", model_name="gpt-4.1")
    pipeline = ReportPipeline(llm_analyzer=analyzer)
    result = pipeline.run("api_p2p_discv4_clients_daily")
"""

from report_agent.pipeline.orchestrator import ReportPipeline
from report_agent.pipeline.models import (
    MetricData,
    AnalysisContext,
    RawAnalysis,
    ValidatedResult,
)
from report_agent.pipeline.exceptions import (
    PipelineError,
    DataFetchError,
    ContextBuildError,
    AnalysisError,
    ValidationError,
)

__all__ = [
    "ReportPipeline",
    "MetricData",
    "AnalysisContext",
    "RawAnalysis",
    "ValidatedResult",
    "PipelineError",
    "DataFetchError",
    "ContextBuildError",
    "AnalysisError",
    "ValidationError",
]
