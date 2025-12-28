from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set

from ludic.agents.base_agent import Agent
from ludic.inference.request import InferenceSpec, ToolRequest

logger = logging.getLogger(__name__)

ToolScope = Literal["internal", "external"]


class ToolAgent(Agent):
    """
    Base Agent with OpenAI/vLLM-style tool calling support.

    Provides:
      - Tool schema generation from python callables.
      - SamplingArgs augmentation to advertise tools to the model.
      - Execution + recording of tool calls into the ContextStrategy.

    Tool Scopes:
      - tools: Tools executed by the agent (calculator, code interpreter).
        Results are added to context and the agent continues reasoning.
      - external_tools: Tools the agent can call but doesn't execute.
        These are returned to the protocol, which decides how to handle them
        (e.g., delegation, environment interaction, etc.).

    Tool errors:
      - Missing tools, invalid JSON arguments, and tool exceptions are
        caught and recorded as tool messages in the ContextStrategy.
    """

    def __init__(
        self,
        *,
        tools: Optional[Sequence[Callable]] = None,
        external_tools: Optional[Sequence[Callable]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Normalize tool inputs
        internal = list(tools or [])
        external = list(external_tools or [])

        # Build tool maps and track scopes
        self._internal_tools: Dict[str, Callable] = {t.__name__: t for t in internal}
        self._external_tools: Dict[str, Callable] = {t.__name__: t for t in external}
        self._external_tool_names: Set[str] = set(self._external_tools.keys())

        # Combined map for schema generation (all tools are advertised)
        self.tool_map: Dict[str, Callable] = {**self._internal_tools, **self._external_tools}

        # Check for name collisions
        overlap = set(self._internal_tools.keys()) & set(self._external_tools.keys())
        if overlap:
            raise ValueError(f"Tool names must be unique across internal and external tools: {overlap}")

        # Generate schemas for all tools
        all_tools = internal + external
        self.tool_schemas: List[Dict[str, Any]] = [self._func_to_schema(t) for t in all_tools]

        if self.tool_schemas and not self._chat_template.supports_tools():
            raise ValueError(
                "ToolAgent requires a chat_template with tool parsing support "
                "(configure a ToolParser, e.g., HermesToolParser)."
            )

    def get_tool_scope(self, tool_name: str) -> Optional[ToolScope]:
        """Return the scope of a tool by name, or None if not found."""
        if tool_name in self._internal_tools:
            return "internal"
        if tool_name in self._external_tools:
            return "external"
        return None

    def is_external_tool(self, tool_name: str) -> bool:
        """Check if a tool is an external tool (not executed by agent)."""
        return tool_name in self._external_tool_names

    def has_external_tool_call(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """Check if any of the tool calls target an external tool."""
        return any(self.is_external_tool(tc["function"]["name"]) for tc in tool_calls)

    def _tool_request(self) -> ToolRequest:
        return ToolRequest(tools=list(self.tool_schemas))

    def _with_tools(self, inference: Optional[InferenceSpec]) -> InferenceSpec:
        """Return an InferenceSpec suitable for tool calling."""
        return inference or InferenceSpec()

    def _run_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        internal_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls and record results in context.

        Args:
            tool_calls: List of tool call dicts from the model.
            internal_only: If True (default), only execute internal tools.
                          External tools are skipped (handled by protocol).

        Returns:
            List of result dicts for executed tools.
        """
        results: List[Dict[str, Any]] = []
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            args_json = tc["function"]["arguments"]
            call_id = tc["id"]

            # Skip external tools if internal_only
            if internal_only and self.is_external_tool(fn_name):
                continue

            result_str = ""
            tool_fn = self._internal_tools.get(fn_name) if internal_only else self.tool_map.get(fn_name)

            if tool_fn is not None:
                try:
                    fn_args = json.loads(args_json)
                    obs = tool_fn(**fn_args)
                    result_str = str(obs)
                except Exception as e:
                    result_str = f"Error executing tool {fn_name}: {e}"
                    logger.warning(result_str)
            else:
                result_str = f"Error: Tool {fn_name} not found."

            self._ctx.add_tool_result(call_id, fn_name, result_str)
            results.append(
                {
                    "tool_call_id": call_id,
                    "tool_name": fn_name,
                    "arguments_json": args_json,
                    "content": result_str,
                }
            )
        return results

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
