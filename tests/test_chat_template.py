"""Tests for chat template (token-in API)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from ludic.inference.chat_template import HFChatTemplate, TemplateResult
from ludic.inference.tool_parser import HermesToolParser


class MockTokenizer:
    """
    Mock HuggingFace tokenizer for testing.

    Provides predictable token IDs and text output without requiring
    a real model or transformers installation.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.apply_chat_template_calls: List[Dict[str, Any]] = []
        self.decode_calls: List[List[int]] = []

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
    ) -> List[int]:
        """Mock apply_chat_template that returns predictable token IDs."""
        # Record the call for assertions
        self.apply_chat_template_calls.append({
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": add_generation_prompt,
            "tokenize": tokenize,
        })

        # Generate deterministic token IDs based on content
        token_ids = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            # Simple hash to create token IDs
            token_ids.extend([ord(c) % self.vocab_size for c in f"[{role}]{content}"])

        # Add tool tokens if present
        if tools:
            token_ids.extend([900, 901, 902])  # Mock tool tokens

        # Add generation prompt marker if requested
        if add_generation_prompt:
            token_ids.extend([998, 999])  # Mock assistant prefix

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Mock decode that creates readable text."""
        self.decode_calls.append(token_ids)
        return f"<decoded:{len(token_ids)}_tokens>"


class TestTemplateResult:
    """Tests for TemplateResult dataclass."""

    def test_template_result_fields(self):
        """TemplateResult holds prompt_token_ids and prompt_text."""
        result = TemplateResult(
            prompt_token_ids=[1, 2, 3],
            prompt_text="Hello world",
        )

        assert result.prompt_token_ids == [1, 2, 3]
        assert result.prompt_text == "Hello world"

    def test_template_result_is_frozen(self):
        """TemplateResult is immutable."""
        result = TemplateResult(prompt_token_ids=[1, 2], prompt_text="test")

        with pytest.raises(AttributeError):
            result.prompt_token_ids = [3, 4]  # type: ignore


class TestHFChatTemplate:
    """Tests for HFChatTemplate."""

    def test_apply_basic_messages(self):
        """Apply template to basic messages."""
        tokenizer = MockTokenizer()
        template = HFChatTemplate(tokenizer)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        result = template.apply(messages)

        assert isinstance(result, TemplateResult)
        assert len(result.prompt_token_ids) > 0
        assert "<decoded:" in result.prompt_text

        # Verify tokenizer was called correctly
        assert len(tokenizer.apply_chat_template_calls) == 1
        call = tokenizer.apply_chat_template_calls[0]
        assert call["messages"] == messages
        assert call["tools"] is None
        assert call["add_generation_prompt"] is True
        assert call["tokenize"] is True

    def test_apply_with_tools(self):
        """Apply template with tool schemas."""
        tokenizer = MockTokenizer()
        template = HFChatTemplate(tokenizer)

        messages = [{"role": "user", "content": "What's 2+2?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Adds numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    },
                },
            }
        ]

        result = template.apply(messages, tools=tools)

        # Verify tools were passed to tokenizer
        call = tokenizer.apply_chat_template_calls[0]
        assert call["tools"] == tools

        # Mock tokenizer adds tool tokens (900, 901, 902)
        assert 900 in result.prompt_token_ids

    def test_apply_without_generation_prompt(self):
        """Apply template without assistant prefix."""
        tokenizer = MockTokenizer()
        template = HFChatTemplate(tokenizer)

        messages = [{"role": "user", "content": "Test"}]

        result = template.apply(messages, add_generation_prompt=False)

        call = tokenizer.apply_chat_template_calls[0]
        assert call["add_generation_prompt"] is False

        # Mock tokenizer doesn't add 998, 999 when add_generation_prompt=False
        assert 998 not in result.prompt_token_ids

    def test_parse_tool_calls_with_parser(self):
        """parse_tool_calls delegates to tool_parser."""
        tokenizer = MockTokenizer()
        parser = HermesToolParser()
        template = HFChatTemplate(tokenizer, tool_parser=parser)

        text = (
            '<tool_call>\n'
            '{"name": "my_tool", "arguments": {"x": 1}}\n'
            '</tool_call>'
        )

        result = template.parse_tool_calls(text)

        assert result.tool_calls is not None
        assert result.parse_error is False
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "my_tool"

    def test_parse_tool_calls_without_parser(self):
        """parse_tool_calls returns None when no parser configured."""
        tokenizer = MockTokenizer()
        template = HFChatTemplate(tokenizer)  # No tool_parser

        text = '<tool_call>{"name": "tool", "arguments": {}}</tool_call>'

        result = template.parse_tool_calls(text)

        assert result.tool_calls is None
        assert result.parse_error is False

    def test_supports_tools_false_without_parser(self):
        """supports_tools is False when no tool parser is configured."""
        tokenizer = MockTokenizer()
        template = HFChatTemplate(tokenizer)

        assert template.supports_tools() is False

    def test_parse_tool_calls_no_tool_calls_in_text(self):
        """parse_tool_calls returns None when text has no tool calls."""
        tokenizer = MockTokenizer()
        parser = HermesToolParser()
        template = HFChatTemplate(tokenizer, tool_parser=parser)

        text = "Just a regular response with no tools."

        result = template.parse_tool_calls(text)

        assert result.tool_calls is None
        assert result.parse_error is False

    def test_supports_tools_true_with_parser(self):
        """supports_tools is True when tool parser is configured."""
        tokenizer = MockTokenizer()
        parser = HermesToolParser()
        template = HFChatTemplate(tokenizer, tool_parser=parser)

        assert template.supports_tools() is True

    def test_apply_returns_list_of_ints(self):
        """Token IDs are a list of integers, not other types."""
        tokenizer = MockTokenizer()
        template = HFChatTemplate(tokenizer)

        messages = [{"role": "user", "content": "Test"}]
        result = template.apply(messages)

        assert isinstance(result.prompt_token_ids, list)
        assert all(isinstance(t, int) for t in result.prompt_token_ids)

    def test_apply_empty_messages(self):
        """Apply template to empty message list."""
        tokenizer = MockTokenizer()
        template = HFChatTemplate(tokenizer)

        result = template.apply([])

        # Should still work, just with generation prompt tokens
        assert len(result.prompt_token_ids) == 2  # Just [998, 999] from mock

    def test_decode_is_called(self):
        """Verify decode is called to produce prompt_text."""
        tokenizer = MockTokenizer()
        template = HFChatTemplate(tokenizer)

        messages = [{"role": "user", "content": "Hello"}]
        result = template.apply(messages)

        # Check decode was called with the token IDs
        assert len(tokenizer.decode_calls) == 1
        assert tokenizer.decode_calls[0] == result.prompt_token_ids
