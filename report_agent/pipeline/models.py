# report_agent/pipeline/models.py
"""
Data models for the report generation pipeline.

These dataclasses define the contracts between pipeline stages,
making the data flow explicit and testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class MetricData:
    """
    Output of DataFetcher stage.
    
    Contains the raw data fetched from ClickHouse for a single metric.
    """
    model_name: str
    kind: str  # "time_series" or "snapshot"
    df: pd.DataFrame
    history_days: Optional[int] = None
    
    def __post_init__(self):
        # Validate required fields
        if self.df is None:
            raise ValueError(f"DataFrame cannot be None for model '{self.model_name}'")
        if self.kind not in ("time_series", "snapshot"):
            raise ValueError(f"Invalid kind '{self.kind}' for model '{self.model_name}'")


@dataclass
class AnalysisContext:
    """
    Output of ContextBuilder stage.
    
    Contains everything the LLM needs to perform analysis:
    - File paths (temp files ready for upload)
    - The prompt to send
    - Reference to original data for validation
    """
    model_name: str
    kind: str
    history_days: int
    
    # File paths (temp files ready for upload)
    csv_path: Path
    schema_path: Path
    meta_path: Path
    docs_path: Optional[Path] = None
    catalog_path: Optional[Path] = None
    
    # Prompt ready to send
    prompt: str = ""
    
    # Keep reference to original data for validation
    df: pd.DataFrame = field(default=None, repr=False)
    
    # Temp directory to clean up later
    temp_dir: Path = field(default=None, repr=False)
    
    def get_all_file_paths(self) -> List[Path]:
        """Return all file paths that should be uploaded to the LLM."""
        paths = [self.csv_path, self.schema_path, self.meta_path]
        if self.docs_path:
            paths.append(self.docs_path)
        if self.catalog_path:
            paths.append(self.catalog_path)
        return paths


@dataclass
class RawAnalysis:
    """
    Output of LLMAnalyzer stage - before validation.
    
    Contains the raw output from the LLM, parsed into narrative and structured data.
    """
    model_name: str
    narrative: str
    structured: Dict
    artifacts: Optional[Dict] = None  # Container file citations for downloading plots
    
    def has_structured_output(self) -> bool:
        """Check if structured output was successfully parsed."""
        return bool(self.structured)
    
    def get_significance(self) -> str:
        """Get the significance level from structured output."""
        return self.structured.get("significance", "").upper()


@dataclass
class ValidatedResult:
    """
    Output of Validator stage - ready for reporting.
    
    Contains the validated analysis result with any warnings or errors.
    """
    model_name: str
    narrative: str
    structured: Dict
    validation_status: str  # "valid" | "warnings" | "errors"
    validation_warnings: List[str] = field(default_factory=list)
    artifacts: Optional[Dict] = None
    df: pd.DataFrame = field(default=None, repr=False)  # Keep for CSV export
    
    def is_valid(self) -> bool:
        """Check if the result passed validation without errors."""
        return self.validation_status != "errors"
    
    def has_warnings(self) -> bool:
        """Check if the result has validation warnings."""
        return self.validation_status == "warnings" or bool(self.validation_warnings)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format expected by report_service."""
        return {
            "narrative": self.narrative,
            "structured": self.structured,
            "validation_status": self.validation_status,
            "validation_warnings": self.validation_warnings,
        }
