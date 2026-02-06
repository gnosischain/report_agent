# report_agent/pipeline/exceptions.py
"""
Custom exceptions for the report generation pipeline.

Each stage has its own exception type for clear error handling.
"""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    
    def __init__(self, message: str, model_name: str = None):
        self.model_name = model_name
        super().__init__(message)


class DataFetchError(PipelineError):
    """Raised when data fetching fails."""
    
    def __init__(self, message: str, model_name: str = None):
        super().__init__(f"Data fetch error: {message}", model_name)


class ContextBuildError(PipelineError):
    """Raised when context building fails."""
    
    def __init__(self, message: str, model_name: str = None):
        super().__init__(f"Context build error: {message}", model_name)


class AnalysisError(PipelineError):
    """Raised when LLM analysis fails."""
    
    def __init__(self, message: str, model_name: str = None):
        super().__init__(f"Analysis error: {message}", model_name)


class ValidationError(PipelineError):
    """Raised when validation fails critically."""
    
    def __init__(self, message: str, model_name: str = None):
        super().__init__(f"Validation error: {message}", model_name)
