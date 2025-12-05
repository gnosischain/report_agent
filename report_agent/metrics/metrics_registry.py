from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_HISTORY_DAYS = 30


class MetricsRegistry:
    """
    Loads metric definitions from metrics/metrics.yml and provides helpers
    to query per-model configuration (thresholds, history window, etc.).
    """

    def __init__(self, metrics_cfg: Optional[List[Dict[str, Any]]] = None):
        if metrics_cfg is None:
            pkg_root = Path(__file__).resolve().parent.parent
            path = pkg_root / "metrics" / "metrics.yml"
            if not path.exists():
                raise FileNotFoundError(f"Missing metrics YAML at {path}")
            raw = yaml.safe_load(path.read_text()) or {}
            metrics_cfg = raw.get("metrics", [])

        self._by_model: Dict[str, Dict[str, Any]] = {}
        for m in metrics_cfg:
            name = m.get("model")
            if not name:
                continue
            self._by_model[name] = m

    def list_models(self) -> List[str]:
        return sorted(self._by_model.keys())

    def has(self, model: str) -> bool:
        return model in self._by_model

    def get(self, model: str) -> Dict[str, Any]:
        return self._by_model.get(model, {})

    def get_history_days(self, model: str) -> int:
        cfg = self.get(model)
        return int(cfg.get("history_days", DEFAULT_HISTORY_DAYS))
    
    def get_kind(self, model: str) -> str:
        """
        Return the metric kind, defaulting to 'time_series' if not specified.
        Valid kinds currently: 'time_series', 'snapshot'.
        """
        cfg = self.get(model)
        return cfg.get("kind", "time_series")
    
    def get_display_name(self, model: str) -> str:
        """
        Return the human-readable display name for a metric.
        Falls back to the model name if no display name is specified.
        """
        cfg = self.get(model)
        return cfg.get("name") or cfg.get("display_name") or model