from report_agent.utils.config_loader import load_configs
from typing import List, Dict, Any, Optional

class MetricsRegistry:
    def __init__(self):
        cfg = load_configs()
        self._metrics: List[Dict[str, Any]] = cfg["metrics"]

    def list_models(self) -> List[str]:
        return [m["model"] for m in self._metrics]

    def has(self, model_name: str) -> bool:
        return any(m["model"] == model_name for m in self._metrics)

    def get(self, model_name: str) -> Dict[str, Any]:
        """
        Return the full metric config (including thresholds) for a given model.
        """
        for m in self._metrics:
            if m["model"] == model_name:
                return m
        raise KeyError(f"Unknown metric {model_name}")

    def get_thresholds(self, model_name: str) -> Dict[str, Any]:
        """
        Return the thresholds dict (or empty dict if not set).
        """
        metric = self.get(model_name)
        return metric.get("thresholds", {})
    
    def get_history_days(self, model_name: str) -> int:
        """
        Return configured history window in days, defaulting to 180 if not set.
        """
        metric = self.get(model_name)
        return int(metric.get("history_days", 180))