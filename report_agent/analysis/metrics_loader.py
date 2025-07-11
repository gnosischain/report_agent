from report_agent.connectors.clickhouse_connector import ClickHouseConnector
from report_agent.analysis.metrics_registry import MetricsRegistry

class MetricsLoader:
    def __init__(self):
        self.db       = ClickHouseConnector()
        self.registry = MetricsRegistry()

    def fetch_time_series(self, model: str, lookback_days: int = 14):
        """
        Pulls every column from `model` for the last `lookback_days`.
        Returns a DataFrame with all original columns (including 'date').
        """
        if not self.registry.has(model):
            raise KeyError(f"Unknown metric {model}")

        # assume time column is always named 'date'
        time_col = "date"

        sql = f"""
            SELECT *
            FROM {self.db.read.database}.{model}
            WHERE `{time_col}` >= today() - INTERVAL {lookback_days} DAY
            ORDER BY `{time_col}` ASC
        """
        return self.db.fetch_df(sql)
