from report_agent.connectors.db.clickhouse_connector import ClickHouseConnector
from report_agent.metrics.metrics_registry import MetricsRegistry

class MetricsLoader:
    def __init__(self):
        self.db       = ClickHouseConnector()
        self.registry = MetricsRegistry()

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
