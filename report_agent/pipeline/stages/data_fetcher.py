# report_agent/pipeline/stages/data_fetcher.py
"""
DataFetcher stage: Fetches raw data from ClickHouse.

This is the first stage of the pipeline, responsible for:
- Looking up metric configuration from the registry
- Fetching the appropriate data slice (time series or snapshot)
- Validating that the data meets requirements
"""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Optional

from report_agent.pipeline.models import MetricData
from report_agent.pipeline.exceptions import DataFetchError

if TYPE_CHECKING:
    from report_agent.metrics.metrics_loader import MetricsLoader
    from report_agent.metrics.metrics_registry import MetricsRegistry

log = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches raw data for a metric from ClickHouse.
    
    Args:
        registry: MetricsRegistry instance for metric metadata.
        loader: MetricsLoader instance for data fetching.
        
    If not provided, creates instances internally (deprecated behavior).
    
    Usage:
        fetcher = DataFetcher(registry=registry, loader=loader)
        data = fetcher.fetch("api_p2p_discv4_clients_daily")
    """
    
    def __init__(
        self,
        registry: Optional[MetricsRegistry] = None,
        loader: Optional[MetricsLoader] = None,
    ):
        # Support legacy usage without injected dependencies
        if registry is None or loader is None:
            warnings.warn(
                "DataFetcher() without registry/loader is deprecated. "
                "Pass dependencies explicitly: DataFetcher(registry=registry, loader=loader)",
                DeprecationWarning,
                stacklevel=2,
            )
            # Lazy import to avoid circular imports
            from report_agent.metrics.metrics_loader import MetricsLoader
            from report_agent.metrics.metrics_registry import MetricsRegistry
            
            if registry is None:
                registry = MetricsRegistry()
            if loader is None:
                loader = MetricsLoader()
        
        self.registry = registry
        self.loader = loader
    
    def fetch(self, model_name: str, lookback_days: Optional[int] = None) -> MetricData:
        """
        Fetch data for a metric.
        
        Args:
            model_name: The metric/model name from metrics.yml
            lookback_days: Override the default history window (optional)
            
        Returns:
            MetricData containing the fetched DataFrame and metadata
            
        Raises:
            DataFetchError: If the model is not found or data fetch fails
        """
        # Validate model exists
        if not self.registry.has(model_name):
            raise DataFetchError(f"Unknown metric '{model_name}' - not found in metrics.yml", model_name)
        
        kind = self.registry.get_kind(model_name)
        log.debug(f"Fetching {kind} data for model '{model_name}'")
        
        try:
            if kind == "time_series":
                history = int(lookback_days or self.registry.get_history_days(model_name))
                df = self.loader.fetch_time_series(model_name, lookback_days=history)
            else:
                # Snapshots: no lookback_days
                history = None
                df = self.loader.fetch_snapshot(model_name)
        except Exception as e:
            raise DataFetchError(f"Failed to fetch data for '{model_name}': {e}", model_name)
        
        # Validate data
        if df is None or df.empty:
            raise DataFetchError(f"No data returned for model '{model_name}'", model_name)
        
        if kind == "time_series" and "date" not in df.columns:
            raise DataFetchError(
                f"Model '{model_name}' has no 'date' column; time series metrics require a date column",
                model_name
            )
        
        log.info(f"Fetched {len(df)} rows for model '{model_name}'")
        
        return MetricData(
            model_name=model_name,
            kind=kind,
            df=df,
            history_days=history,
        )
