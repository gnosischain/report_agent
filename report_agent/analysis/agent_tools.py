from report_agent.analysis.metrics_loader import MetricsLoader
from report_agent.analysis.analyzer        import (
    compute_weekly_delta,
    detect_anomalies,
    moving_average,
    summary_statistics,
    compare_periods,
)
from report_agent.analysis.thresholds      import check_thresholds
from report_agent.analysis.metrics_registry import MetricsRegistry


def _get_val_col(df):
    """Return the last non-date column, or raise if none."""
    val_cols = [c for c in df.columns if c != "date"]
    if not val_cols:
        raise ValueError(f"No numeric column found; columns={list(df.columns)}")
    return val_cols[-1]


def delta_for_metric(
    metric: str,
    lookback_days: int = 14,
    window: int = 7
) -> dict:
    """
    Fetches the time series for `metric`, then returns the weekly delta.
    If there’s no data or no numeric column, returns null‐filled structure.
    """
    df = MetricsLoader().fetch_time_series(metric, lookback_days)
    try:
        val_col = _get_val_col(df)
    except ValueError:
        return {"previous": None, "current": None, "pct_change": None}

    return compute_weekly_delta(
        df, date_col="date", value_col=val_col, window=window
    )


def anomalies_for_metric(
    metric: str,
    lookback_days: int = 14,
    z_thresh: float = 2.0
) -> list:
    """
    Fetches the time series for `metric`, then returns anomaly dates.
    If there’s no data or no numeric column, returns an empty list.
    """
    df = MetricsLoader().fetch_time_series(metric, lookback_days)
    try:
        val_col = _get_val_col(df)
    except ValueError:
        return []

    return detect_anomalies(
        df, date_col="date", value_col=val_col, z_thresh=z_thresh
    )


def moving_average_for_metric(
    metric: str,
    lookback_days: int = 14,
    window: int = 7
) -> dict:
    """
    Fetches the time series for `metric`, computes a rolling average,
    and returns the most‐recent row (including the MA).
    If there’s no data, returns an empty dict.
    """
    df = MetricsLoader().fetch_time_series(metric, lookback_days)
    try:
        val_col = _get_val_col(df)
    except ValueError:
        return {}

    df_ma = moving_average(df, date_col="date", value_col=val_col, window=window)
    row   = df_ma.iloc[-1]
    result = {c: row[c] for c in df.columns}
    result[f"{val_col}_ma"] = row[f"{val_col}_ma"]
    return result


def summary_for_metric(
    metric: str,
    lookback_days: int = 14
) -> dict:
    """
    Fetches the time series for `metric`, then returns summary statistics.
    If there’s no data, returns an empty dict.
    """
    df = MetricsLoader().fetch_time_series(metric, lookback_days)
    try:
        val_col = _get_val_col(df)
    except ValueError:
        return {}

    return summary_statistics(df, value_col=val_col)


def compare_periods_for_metric(
    metric: str,
    lookback_days: int = 14,
    period: int = 7
) -> dict:
    """
    Fetches the time series for `metric`, then compares two consecutive periods.
    If there’s no data, returns an empty dict.
    """
    df = MetricsLoader().fetch_time_series(metric, lookback_days)
    try:
        val_col = _get_val_col(df)
    except ValueError:
        return {}

    return compare_periods(
        df, date_col="date", value_col=val_col, period=period
    )


def check_alerts_for_metric(
    metric: str,
    lookback_days: int = 14,
    window: int = 7,
    z_thresh: float = 2.0
) -> dict:
    """
    Runs delta_for_metric and anomalies_for_metric, then checks thresholds.
    Returns a dict with keys: delta, anomalies, alerts (alerts may be empty).
    """
    delta     = delta_for_metric(metric, lookback_days, window)
    anomalies = anomalies_for_metric(metric, lookback_days, z_thresh)
    thresholds = MetricsRegistry().get_thresholds(metric)
    alerts    = check_thresholds(delta, anomalies, thresholds)
    return {
        "delta":     delta,
        "anomalies": anomalies,
        "alerts":    alerts
    }