# report_agent/pipeline/stages/llm_analyzer.py
"""
LLMAnalyzer abstract base class.

This defines the interface that all LLM analyzer implementations must follow.
Currently supports OpenAI Code Interpreter, but can be extended to support
other providers (e.g., Gemini, Claude) by implementing this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Iterable

from report_agent.pipeline.models import AnalysisContext, RawAnalysis


class LLMAnalyzer(ABC):
    """
    Abstract base class for LLM analysis providers.
    
    Implementations must provide:
    - analyze(): Run analysis on the context and return raw results
    - download_artifacts(): Download any generated artifacts (plots, etc.)
    
    Example implementation:
        class OpenAIAnalyzer(LLMAnalyzer):
            def analyze(self, context: AnalysisContext) -> RawAnalysis:
                # Upload files, call API, parse response
                ...
            
            def download_artifacts(self, output_dir: str) -> List[str]:
                # Download plots from container
                ...
    """
    
    @abstractmethod
    def analyze(self, context: AnalysisContext) -> RawAnalysis:
        """
        Run LLM analysis on the provided context.
        
        Args:
            context: AnalysisContext from the ContextBuilder stage
            
        Returns:
            RawAnalysis containing the narrative and structured output
            
        Raises:
            AnalysisError: If the analysis fails
        """
        ...
    
    @abstractmethod
    def download_artifacts(
        self,
        output_dir: str = "reports/plots",
        include_extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".csv", ".json", ".md"),
    ) -> List[str]:
        """
        Download any artifacts generated during analysis.
        
        Args:
            output_dir: Directory to save artifacts to
            include_extensions: File extensions to include
            
        Returns:
            List of saved file paths (or 'ERROR:<filename>:<exc>' strings on failure)
        """
        ...
    
    def get_last_artifacts(self) -> dict | None:
        """
        Return artifacts metadata from the most recent analysis.
        
        Default implementation returns None; override in subclasses that track artifacts.
        """
        return None
