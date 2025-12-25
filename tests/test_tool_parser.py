"""Tests for tool parsers (token-in API)."""

from __future__ import annotations

import pytest

from ludic.inference.tool_parser import HermesToolParser


class TestHermesToolParser:
    """Tests for HermesToolParser."""

    def test_parse_single_tool_call(self):
        """Parse a single tool call in Hermes format."""
        parser = HermesToolParser()
        text = (
            "Let me calculate that.\n"
            "<tool_call>\n"
            '{"name": "calculator", "arguments": {"a": 5, "b": 3}}\n'
            "</tool_call>"
        )

        result = parser.parse(text)

        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "call_0"
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "calculator"
        assert result[0]["function"]["arguments"] == '{"a": 5, "b": 3}'

    def test_parse_multiple_tool_calls(self):
        """Parse multiple tool calls in a single response."""
        parser = HermesToolParser()
        text = (
            "I'll use both tools.\n"
            "<tool_call>\n"
            '{"name": "calculator", "arguments": {"a": 1, "b": 2}}\n'
            "</tool_call>\n"
            "And also:\n"
            "<tool_call>\n"
            '{"name": "weather", "arguments": {"location": "NYC"}}\n'
            "</tool_call>"
        )

        result = parser.parse(text)

        assert result is not None
        assert len(result) == 2
        assert result[0]["id"] == "call_0"
        assert result[0]["function"]["name"] == "calculator"
        assert result[1]["id"] == "call_1"
        assert result[1]["function"]["name"] == "weather"
        assert result[1]["function"]["arguments"] == '{"location": "NYC"}'

    def test_parse_no_tool_calls_returns_none(self):
        """Return None when no tool_call tags are present."""
        parser = HermesToolParser()
        text = "The answer is 42. No tools needed."

        result = parser.parse(text)

        assert result is None

    def test_parse_invalid_json_skipped(self):
        """Invalid JSON inside tool_call tags is skipped."""
        parser = HermesToolParser()
        text = (
            "<tool_call>\n"
            "not valid json\n"
            "</tool_call>"
        )

        result = parser.parse(text)

        # Invalid JSON results in empty list -> None
        assert result is None

    def test_parse_mixed_valid_invalid_json(self):
        """Valid tool calls are kept, invalid ones are skipped."""
        parser = HermesToolParser()
        text = (
            "<tool_call>\n"
            "invalid json here\n"
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "valid_tool", "arguments": {"x": 1}}\n'
            "</tool_call>"
        )

        result = parser.parse(text)

        assert result is not None
        assert len(result) == 1
        # The valid one gets id call_1 because it's second in the match order
        # but after filtering invalid, it's the only one
        assert result[0]["function"]["name"] == "valid_tool"

    def test_parse_empty_arguments(self):
        """Handle tool calls with no arguments."""
        parser = HermesToolParser()
        text = (
            "<tool_call>\n"
            '{"name": "no_args_tool", "arguments": {}}\n'
            "</tool_call>"
        )

        result = parser.parse(text)

        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "no_args_tool"
        assert result[0]["function"]["arguments"] == "{}"

    def test_parse_missing_arguments_key(self):
        """Handle tool calls without arguments key (defaults to empty)."""
        parser = HermesToolParser()
        text = (
            "<tool_call>\n"
            '{"name": "simple_tool"}\n'
            "</tool_call>"
        )

        result = parser.parse(text)

        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "simple_tool"
        assert result[0]["function"]["arguments"] == "{}"

    def test_parse_whitespace_tolerance(self):
        """Handle various whitespace in tool_call tags."""
        parser = HermesToolParser()
        # Compact format
        text_compact = '<tool_call>{"name": "tool1", "arguments": {}}</tool_call>'
        # Lots of whitespace
        text_spaced = (
            "<tool_call>  \n\n  "
            '{"name": "tool2", "arguments": {"a": 1}}'
            "  \n  </tool_call>"
        )

        result1 = parser.parse(text_compact)
        result2 = parser.parse(text_spaced)

        assert result1 is not None
        assert result1[0]["function"]["name"] == "tool1"
        assert result2 is not None
        assert result2[0]["function"]["name"] == "tool2"

    def test_parse_nested_json_arguments(self):
        """Handle complex nested JSON in arguments."""
        parser = HermesToolParser()
        text = (
            "<tool_call>\n"
            '{"name": "complex_tool", "arguments": {"nested": {"a": [1, 2, 3], "b": {"c": "d"}}}}\n'
            "</tool_call>"
        )

        result = parser.parse(text)

        assert result is not None
        assert result[0]["function"]["name"] == "complex_tool"
        # Arguments should be JSON string
        import json
        args = json.loads(result[0]["function"]["arguments"])
        assert args["nested"]["a"] == [1, 2, 3]
        assert args["nested"]["b"]["c"] == "d"

    def test_parse_tool_call_with_surrounding_text(self):
        """Tool calls embedded in other text are extracted."""
        parser = HermesToolParser()
        text = (
            "I'm thinking about this problem...\n"
            "Let me use a tool:\n"
            "<tool_call>\n"
            '{"name": "my_tool", "arguments": {"query": "test"}}\n'
            "</tool_call>\n"
            "Now I'll wait for the result."
        )

        result = parser.parse(text)

        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "my_tool"

    def test_parse_empty_string(self):
        """Empty input returns None."""
        parser = HermesToolParser()

        result = parser.parse("")

        assert result is None

    def test_parse_missing_name_key(self):
        """Handle tool call without name (defaults to empty string)."""
        parser = HermesToolParser()
        text = (
            "<tool_call>\n"
            '{"arguments": {"x": 1}}\n'
            "</tool_call>"
        )

        result = parser.parse(text)

        assert result is not None
        assert result[0]["function"]["name"] == ""
