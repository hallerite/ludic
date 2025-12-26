"""
Tool call parsers for token-in API.

These parsers extract tool calls from raw model output text.
Tool *formatting* (injecting tool schemas into prompts) is handled by
HuggingFace's apply_chat_template(tools=...) - we don't need to do that ourselves.

Different models output tool calls differently:
- Hermes format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
- Llama format: {"name": ..., "parameters": ...}
- etc.

This module provides parsers for extracting structured tool calls from these formats.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolParseResult:
    """
    Result of parsing tool calls from a completion.

    tool_calls is None when no tool calls are present or none could be parsed.
    parse_error is True when tool-call tags exist but parsing failed.
    """

    tool_calls: Optional[List[Dict[str, Any]]]
    parse_error: bool = False


class ToolParser(ABC):
    """
    Base class for parsing tool calls from model output.

    Different models emit tool calls in different formats. Implementations
    parse the raw completion text and extract structured tool call data.
    """

    @abstractmethod
    def parse(self, completion_text: str) -> ToolParseResult:
        """
        Extract tool calls from completion text.

        Args:
            completion_text: Raw model output text.

        Returns:
            ToolParseResult with tool_calls in OpenAI format:
            [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
            tool_calls is None if no tool calls were parsed.
        """
        ...


class HermesToolParser(ToolParser):
    """
    Parser for the Hermes tool calling format.

    This format is used by many models including Qwen, NousResearch Hermes,
    and others. Tool calls are wrapped in <tool_call> tags:

        <tool_call>
        {"name": "function_name", "arguments": {"arg1": "value1"}}
        </tool_call>
    """

    def parse(self, completion_text: str) -> ToolParseResult:
        # Look for <tool_call>...</tool_call> blocks
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, completion_text, re.DOTALL)

        if not matches:
            return ToolParseResult(tool_calls=None, parse_error=False)

        tool_calls = []
        parse_error = False
        for i, match in enumerate(matches):
            try:
                call_data = json.loads(match)
                tool_calls.append({
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": call_data.get("name", ""),
                        "arguments": json.dumps(call_data.get("arguments", {})),
                    },
                })
            except json.JSONDecodeError:
                parse_error = True
                continue

        return ToolParseResult(
            tool_calls=tool_calls if tool_calls else None,
            parse_error=parse_error,
        )
