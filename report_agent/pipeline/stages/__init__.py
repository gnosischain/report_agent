# report_agent/pipeline/stages/__init__.py
"""
Pipeline stages for report generation.

Each stage is a single-responsibility class that transforms data
from one format to another.
"""

from report_agent.pipeline.stages.data_fetcher import DataFetcher
from report_agent.pipeline.stages.context_builder import ContextBuilder
from report_agent.pipeline.stages.llm_analyzer import LLMAnalyzer
from report_agent.pipeline.stages.openai_analyzer import OpenAIAnalyzer
from report_agent.pipeline.stages.validator import ResultValidator

__all__ = [
    "DataFetcher",
    "ContextBuilder",
    "LLMAnalyzer",
    "OpenAIAnalyzer",
    "ResultValidator",
]
