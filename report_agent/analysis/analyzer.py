import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

def compute_weekly_delta(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    window: int = 7
) -> Dict[str, Any]:
    """
    Compute the previous- and current-week sums and percent change for any time series.

    Args:
      df: DataFrame containing at least `2 * window` rows.
      date_col: name of the datetime column (will be sorted ascending).
      value_col: name of the numeric column to aggregate.
      window: number of days in each period (default 7).

    Returns:
      {
        'previous': float,      # sum of values in the first `window` days
        'current':  float,      # sum of values in the last  `window` days
        'pct_change': float     # ((current - previous)/previous*100) rounded to 2 decimals, or None
      }

    Raises:
      ValueError: if df has fewer than 2*window rows.
    """
    df2 = df.sort_values(date_col)
    vals = df2[value_col].astype(float).values
    if len(vals) < window * 2:
        raise ValueError(f"Need at least {2*window} rows to compute two periods of length {window}.")
    prev = float(vals[:window].sum())
    curr = float(vals[-window:].sum())
    pct  = (curr - prev) / prev * 100 if prev != 0 else None
    return {
        "previous": prev,
        "current":  curr,
        "pct_change": round(pct, 2) if pct is not None else None,
    }

def detect_anomalies(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    z_thresh: float = 2.0
) -> List[str]:
    """
    Flag dates where the `value_col` deviates more than `z_thresh` σ from its mean.

    Args:
      df: DataFrame with at least one numeric `value_col`.
      date_col: name of the datetime column.
      value_col: name of the numeric column to test.
      z_thresh: z-score threshold.

    Returns:
      A list of ISO date strings for all anomaly points.
    """
    df2 = df.sort_values(date_col)
    vals = df2[value_col].astype(float)
    mean, std = float(vals.mean()), float(vals.std())
    if std == 0 or np.isnan(std):
        return []
    mask = np.abs((vals - mean) / std) > z_thresh
    return df2.loc[mask, date_col].dt.strftime("%Y-%m-%d").tolist()

def moving_average(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    window: int = 7
) -> pd.DataFrame:
    """
    Compute a rolling moving average of any series.

    Args:
      df: DataFrame with a datetime `date_col` and numeric `value_col`.
      window: window size for the rolling average.

    Returns:
      A DataFrame with original columns plus:
        - `{value_col}_ma`: the moving average.
    """
    df2 = df.sort_values(date_col).copy()
    df2[f"{value_col}_ma"] = df2[value_col].astype(float).rolling(window=window, min_periods=1).mean()
    return df2

def summary_statistics(
    df: pd.DataFrame,
    value_col: str = "value"
) -> Dict[str, float]:
    """
    Compute basic summary stats for any numeric column.

    Args:
      df: DataFrame with numeric `value_col`.

    Returns:
      {
        'min':    float,
        'max':    float,
        'mean':   float,
        'median': float,
        'std':    float
      }
    """
    vals = df[value_col].astype(float)
    return {
        "min":    float(vals.min()),
        "max":    float(vals.max()),
        "mean":   float(vals.mean()),
        "median": float(vals.median()),
        "std":    float(vals.std()),
    }

def compare_periods(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    period: int = 7
) -> Dict[str, Any]:
    """
    Compare sums of two consecutive periods of any length.

    Args:
      df: DataFrame with `date_col` and `value_col`, sorted by date.
      period: number of days in each period.

    Returns:
      {
        'prev_sum':       float,
        'curr_sum':       float,
        'percent_change': float
      }

    Raises:
      ValueError: if df has fewer than 2*period rows.
    """
    df2 = df.sort_values(date_col)
    vals = df2[value_col].astype(float).values
    if len(vals) < period * 2:
        raise ValueError(f"Need at least {2*period} rows for compare_periods.")
    prev = float(vals[-2*period:-period].sum())
    curr = float(vals[-period:].sum())
    pct  = (curr - prev) / prev * 100 if prev else None
    return {
        "prev_sum":       prev,
        "curr_sum":       curr,
        "percent_change": round(pct, 2) if pct is not None else None
    }
