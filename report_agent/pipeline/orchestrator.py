# report_agent/pipeline/orchestrator.py
"""
ReportPipeline: Orchestrates the full analysis pipeline.

This is the main entry point for the pipeline architecture.
It coordinates the flow between stages:
    DataFetcher -> ContextBuilder -> LLMAnalyzer -> ResultValidator
"""
from __future__ import annotations

import logging
import shutil
from typing import Optional

from report_agent.pipeline.models import MetricData, AnalysisContext, ValidatedResult
from report_agent.pipeline.stages.data_fetcher import DataFetcher
from report_agent.pipeline.stages.context_builder import ContextBuilder
from report_agent.pipeline.stages.llm_analyzer import LLMAnalyzer
from report_agent.pipeline.stages.validator import ResultValidator
from report_agent.pipeline.exceptions import PipelineError

log = logging.getLogger(__name__)


class ReportPipeline:
    """
    Orchestrates the full analysis pipeline.
    
    This class coordinates the flow of data through all pipeline stages:
    1. DataFetcher: Fetch raw data from ClickHouse
    2. ContextBuilder: Prepare files and prompts for LLM
    3. LLMAnalyzer: Run LLM analysis
    4. ResultValidator: Validate the results
    
    Usage:
        from report_agent.pipeline import ReportPipeline
        from report_agent.pipeline.stages import OpenAIAnalyzer
        
        analyzer = OpenAIAnalyzer(api_key="...", model_name="gpt-4.1")
        pipeline = ReportPipeline(llm_analyzer=analyzer)
        result = pipeline.run("api_p2p_discv4_clients_daily")
    """
    
    def __init__(self, llm_analyzer: LLMAnalyzer):
        """
        Initialize the pipeline with an LLM analyzer.
        
        Args:
            llm_analyzer: An instance of LLMAnalyzer (e.g., OpenAIAnalyzer)
        """
        self.data_fetcher = DataFetcher()
        self.context_builder = ContextBuilder()
        self.llm_analyzer = llm_analyzer
        self.validator = ResultValidator()
        
        # Keep reference to last context for artifact downloads
        self._last_context: Optional[AnalysisContext] = None
    
    def run(self, model_name: str, lookback_days: Optional[int] = None) -> ValidatedResult:
        """
        Run the full analysis pipeline for a metric.
        
        Args:
            model_name: The metric/model name from metrics.yml
            lookback_days: Override the default history window (optional)
            
        Returns:
            ValidatedResult containing the analysis and validation status
            
        Raises:
            PipelineError: If any stage fails
        """
        context = None
        
        try:
            # Stage 1: Fetch data
            log.info(f"[1/4] Fetching data for '{model_name}'...")
            metric_data = self.data_fetcher.fetch(model_name, lookback_days)
            
            # Stage 2: Build context
            log.info(f"[2/4] Building context for '{model_name}'...")
            context = self.context_builder.build(metric_data)
            self._last_context = context
            
            # Stage 3: LLM analysis
            log.info(f"[3/4] Running LLM analysis for '{model_name}'...")
            raw_analysis = self.llm_analyzer.analyze(context)
            
            # Stage 4: Validate
            log.info(f"[4/4] Validating results for '{model_name}'...")
            result = self.validator.validate(raw_analysis, context.df)
            
            log.info(f"Pipeline complete for '{model_name}' - status: {result.validation_status}")
            return result
            
        except PipelineError:
            # Re-raise pipeline errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise PipelineError(f"Unexpected error in pipeline: {e}", model_name)
        finally:
            # Cleanup temp files
            if context is not None:
                self._cleanup(context)
    
    def _cleanup(self, context: AnalysisContext):
        """Clean up temporary files created during pipeline execution."""
        if context.temp_dir and context.temp_dir.exists():
            try:
                shutil.rmtree(context.temp_dir)
                log.debug(f"Cleaned up temp dir: {context.temp_dir}")
            except Exception as e:
                log.warning(f"Failed to clean up temp dir {context.temp_dir}: {e}")
    
    def download_artifacts(self, output_dir: str = "reports/plots") -> list[str]:
        """
        Download artifacts from the last analysis.
        
        Convenience method that delegates to the LLM analyzer.
        
        Args:
            output_dir: Directory to save artifacts to
            
        Returns:
            List of saved file paths
        """
        return self.llm_analyzer.download_artifacts(output_dir)
    
    def get_last_dataframe(self, model_name: str):
        """
        Get the DataFrame from the last analysis if it matches the model.
        
        This is for backward compatibility with code that expects
        to retrieve the dataframe from the connector.
        """
        if self._last_context and self._last_context.model_name == model_name:
            return self._last_context.df.copy()
        return None
