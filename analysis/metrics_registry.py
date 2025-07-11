from utils.config_loader import load_configs

class MetricsRegistry:
    def __init__(self):
        cfg = load_configs()
        # Each entry is just {"model": ...}
        self._models = [m["model"] for m in cfg["metrics"]]

    def list_models(self):
        return list(self._models)

    def has(self, model_name: str) -> bool:
        return model_name in self._models
