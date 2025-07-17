from openai import OpenAI
import inspect
from typing import List, Callable, Dict, Any
from report_agent.connectors.llm_connector.base import LLMConnector

# Core JSON‑friendly wrapper tools
from report_agent.analysis.agent_tools import (
    delta_for_metric,
    anomalies_for_metric,
    check_alerts_for_metric,
)

class OpenAIConnector(LLMConnector):
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        super().__init__(api_key, model_name)
        self.client = OpenAI(api_key=api_key)

        # only register tools for now that we know will have data
        self.register_tools([
            delta_for_metric,
            anomalies_for_metric,
            check_alerts_for_metric,
        ])

    def register_tools(self, functions: List[Callable]) -> None:
        tools: List[Dict[str, Any]] = []
        fn_map: Dict[str, Callable] = {}

        for fn in functions:
            sig   = inspect.signature(fn)
            props = {}
            for name, param in sig.parameters.items():
                ann = param.annotation
                t   = "number" if ann in (int, float) else "string"
                props[name] = {"type": t}

            doc  = inspect.getdoc(fn) or ""
            desc = doc.split("\n", 1)[0] if doc else fn.__name__

            tools.append({
                "name":        fn.__name__,
                "description": desc,
                "parameters": {
                    "type":       "object",
                    "properties": props,
                    "required":   list(sig.parameters.keys())
                }
            })
            fn_map[fn.__name__] = fn

        self.tools  = tools
        self.fn_map = fn_map

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        functions: List[Dict[str, Any]],
        function_call: str
    ) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            functions=functions,
            function_call=function_call
        )
        return resp.to_dict()

