from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from report_agent.connectors.db.clickhouse_connector import ClickHouseConnector
    from report_agent.metrics.metrics_registry import MetricsRegistry


class MetricsLoader:
    """
    Loads metric data from ClickHouse.
    
    Args:
        db: ClickHouseConnector instance for database access.
        registry: MetricsRegistry instance for metric metadata.
        
    If not provided, creates instances internally (deprecated behavior).
    """
    
    def __init__(
        self,
        db: Optional[ClickHouseConnector] = None,
        registry: Optional[MetricsRegistry] = None,
    ):
        # Support legacy usage without injected dependencies
        if db is None or registry is None:
            warnings.warn(
                "MetricsLoader() without db/registry is deprecated. "
                "Pass dependencies explicitly: MetricsLoader(db=db, registry=registry)",
                DeprecationWarning,
                stacklevel=2,
            )
            # Lazy import to avoid circular imports
            from report_agent.connectors.db.clickhouse_connector import ClickHouseConnector
            from report_agent.metrics.metrics_registry import MetricsRegistry
            
            if db is None:
                db = ClickHouseConnector()
            if registry is None:
                registry = MetricsRegistry()
        
        self.db = db
        self.registry = registry

    def fetch_time_series(self, model: str, lookback_days: int = None):
        """
        Pull raw rows for `model` for the last N days.
        Always returns all original columns (including 'date'), sorted ascending by `date`.
        NOTE: There may be multiple rows per day (long format). We do not aggregate.
        """
        if not self.registry.has(model):
            raise KeyError(f"Unknown metric {model}")

        # Always use 'date' as time column (you confirmed this invariant).
        time_col = "date"
        days = int(lookback_days or self.registry.get_history_days(model))

        sql = f"""
            SELECT *
            FROM {self.db.read.database}.{model}
            WHERE `{time_col}` >= today() - INTERVAL {days} DAY
            ORDER BY `{time_col}` ASC
        """
        return self.db.fetch_df(sql)

    def fetch_snapshot(self, model: str):
        """
        Fetch a snapshot metric table (no date filter).
        Used for metrics with kind: snapshot in metrics.yml.
        """
        if not self.registry.has(model):
            raise KeyError(f"Unknown metric {model}")

        sql = f"""
            SELECT *
            FROM {self.db.read.database}.{model}
        """
        return self.db.fetch_df(sql)
