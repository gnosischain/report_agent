import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMConnector(ABC):
    def __init__(self, api_key: str, model_name: str):
        self.api_key    = api_key
        self.model_name = model_name
        self.tools      = []   # JSON specs for function‐calling
        self.fn_map     = {}   # name -> Python callable

    @abstractmethod
    def register_tools(self, functions: List[callable]) -> None:
        """
        Introspect each Python function and:
          • populate self.tools with its JSON schema
          • populate self.fn_map[name] = function
        """
        ...

    @abstractmethod
    def _generate(
        self,
        messages: List[Dict[str, Any]],
        functions: List[Dict[str, Any]],
        function_call: str
    ) -> Dict[str, Any]:
        """
        Send a request to the LLM and return the full response as a dict.
        """
        ...

    def run_report(self, model_name: str, lookback_days: int = 14) -> str:
        # deferred imports to avoid circular dependencies
        from report_agent.analysis.metrics_loader    import MetricsLoader
        from report_agent.analysis.metrics_registry  import MetricsRegistry
        from report_agent.analysis.analyzer           import (
            compute_weekly_delta, detect_anomalies,
            moving_average, summary_statistics, compare_periods
        )
        from report_agent.analysis.thresholds         import check_thresholds
        from report_agent.nlg.prompt_builder          import build_prompt

        loader   = MetricsLoader()
        registry = MetricsRegistry()
        df       = loader.fetch_time_series(model_name, lookback_days)

        val_cols = [c for c in df.columns if c != "date"]
        if not val_cols:
            raise ValueError(f"No numeric column found in {model_name}")
        value_col = val_cols[-1]

        delta     = compute_weekly_delta(df, date_col="date", value_col=value_col)
        anomalies = detect_anomalies(df, date_col="date", value_col=value_col)
        thresholds = registry.get_thresholds(model_name)
        alerts    = check_thresholds(delta, anomalies, thresholds)

        prompt = build_prompt(model_name, df.head(7), delta, anomalies, alerts)

        messages      = [{"role": "user", "content": prompt}]
        functions     = self.tools
        function_call = "auto"

        response = self._generate(messages, functions, function_call)

        # function calls
        while True:
            msg = response["choices"][0]["message"]

            # If the model wants to call a function
            if msg.get("function_call"):
                call    = msg["function_call"]
                fn_name = call["name"]
                args    = json.loads(call.get("arguments") or "{}")

                # Look up real function
                fn = self.fn_map.get(fn_name)
                if fn is None:
                    raise KeyError(f"No tool registered under name '{fn_name}'")

                # Execute and capture result
                result = fn(**args)

                # Append the function call and its result to the conversation
                messages.append(msg)
                messages.append({
                    "role":    "function",
                    "name":    fn_name,
                    "content": json.dumps(result)
                })

                # Ask the model to continue reasoning
                response = self._generate(messages, functions, "auto")
                continue

            # Otherwise we've got final text—return it
            return msg.get("content", "")
