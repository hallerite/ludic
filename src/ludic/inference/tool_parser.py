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
from typing import Any, Dict, List, Optional


class ToolParser(ABC):
    """
    Base class for parsing tool calls from model output.

    Different models emit tool calls in different formats. Implementations
    parse the raw completion text and extract structured tool call data.
    """

    @abstractmethod
    def parse(self, completion_text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract tool calls from completion text.

        Args:
            completion_text: Raw model output text.

        Returns:
            List of tool calls in OpenAI format:
            [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
            Returns None if no tool calls found.
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

    def parse(self, completion_text: str) -> Optional[List[Dict[str, Any]]]:
        # Look for <tool_call>...</tool_call> blocks
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, completion_text, re.DOTALL)

        if not matches:
            return None

        tool_calls = []
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
                continue

        return tool_calls if tool_calls else None
