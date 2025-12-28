from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from ludic.agents.base_agent import Agent
from ludic.inference.request import InferenceSpec, ToolRequest

logger = logging.getLogger(__name__)


class ToolAgent(Agent):
    """
    Base Agent with OpenAI/vLLM-style tool calling support.

    Provides:
      - Tool schema generation from python callables.
      - SamplingArgs augmentation to advertise tools to the model.
      - Execution + recording of tool calls into the ContextStrategy.

    Tool errors:
      - Missing tools, invalid JSON arguments, and tool exceptions are
        caught and recorded as tool messages in the ContextStrategy.
    """

    def __init__(self, tools: Sequence[Callable], **kwargs):
        super().__init__(**kwargs)
        self.tool_map: Dict[str, Callable] = {t.__name__: t for t in tools}
        self.tool_schemas: List[Dict[str, Any]] = [self._func_to_schema(t) for t in tools]
        if self.tool_schemas and not self._chat_template.supports_tools():
            raise ValueError(
                "ToolAgent requires a chat_template with tool parsing support "
                "(configure a ToolParser, e.g., HermesToolParser)."
            )

    def _tool_request(self) -> ToolRequest:
        return ToolRequest(tools=list(self.tool_schemas))

    def _with_tools(self, inference: Optional[InferenceSpec]) -> InferenceSpec:
        """Return an InferenceSpec suitable for tool calling."""
        return inference or InferenceSpec()

    def _run_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Execute tool calls and record results in context."""
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            args_json = tc["function"]["arguments"]
            call_id = tc["id"]

            result_str = ""
            if fn_name in self.tool_map:
                try:
                    fn_args = json.loads(args_json)
                    obs = self.tool_map[fn_name](**fn_args)
                    result_str = str(obs)
                except Exception as e:
                    result_str = f"Error executing tool {fn_name}: {e}"
                    logger.warning(result_str)
            else:
                result_str = f"Error: Tool {fn_name} not found."

            self._ctx.add_tool_result(call_id, fn_name, result_str)

    def _func_to_schema(self, f: Callable) -> Dict[str, Any]:
        """
        Minimal schema generator.
        For production, use Pydantic to inspect signature types accurately.
        """
        sig = inspect.signature(f)
        params: Dict[str, Any] = {}
        required_params: List[str] = []

        for name, param in sig.parameters.items():
            p_type = "string"
            # Handle both actual types and PEP 563 string annotations
            ann = param.annotation
            if ann in (int, "int"):
                p_type = "integer"
            elif ann in (float, "float"):
                p_type = "number"
            elif ann in (bool, "bool"):
                p_type = "boolean"

            params[name] = {"type": p_type}

            if param.default == inspect.Parameter.empty:
                required_params.append(name)

        return {
            "type": "function",
            "function": {
                "name": f.__name__,
                "description": f.__doc__ or "No description provided.",
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required_params,
                },
            },
        }
