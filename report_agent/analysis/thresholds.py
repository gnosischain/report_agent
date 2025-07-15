from typing import Dict, Any

def check_thresholds(
    delta: Dict[str, Any],
    anomalies: list,
    thresholds: Dict[str, Any]
) -> Dict[str, bool]:
    """
    Given your computed weekly delta and list of anomalies, plus
    the thresholds dict from metrics.yml, return which alerts fire.

    Args:
      delta: {"previous":…, "current":…, "pct_change":…}
      anomalies: List of anomalous date strings.
      thresholds: e.g. {"pct_change":5, "anomaly_count":1}

    Returns:
      {
        "pct_change_alert": True/False,
        "anomaly_alert":   True/False
      }
    """
    alerts = {}
    pct_thr = thresholds.get("pct_change")
    if pct_thr is not None and delta.get("pct_change") is not None:
        alerts["pct_change_alert"] = abs(delta["pct_change"]) >= pct_thr
    else:
        alerts["pct_change_alert"] = False

    anom_thr = thresholds.get("anomaly_count")
    if anom_thr is not None:
        alerts["anomaly_alert"] = len(anomalies) >= anom_thr
    else:
        alerts["anomaly_alert"] = False

    return alerts