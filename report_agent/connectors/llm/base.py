# report_agent/connectors/llm_connector/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional


class LLMConnector(ABC):
    """
    Minimal interface every LLM connector must implement.

    For now we focus on:
      - run_report: given a dbt/ClickHouse model name, return a text report
      - download_artifacts: optionally fetch plots or other files from the run
    """

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    # Keep this hook so future function-calling connectors can still use it.
    # Code-interpreter based connectors can ignore it.
    def register_tools(self, functions: List[callable]) -> None:
        """Optional: register JSON tools for function-calling style connectors."""
        return

    @abstractmethod
    def run_report(self, model_name: str, lookback_days: Optional[int] = None) -> str:
        """
        Generate a report for the given model and lookback window.
        Implementations decide how to:
          - fetch data
          - prepare files
          - call the LLM
          - return final text
        """
        ...

    def download_artifacts(
        self,
        output_dir: str = "reports/plots",
        include_extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".csv", ".json", ".md"),
    ) -> List[str]:
        """
        Optional: download artifacts (plots, extra files) from the last run.
        Connectors that don't support this can just raise or return [].
        """
        raise NotImplementedError("This connector does not support artifact downloads.")