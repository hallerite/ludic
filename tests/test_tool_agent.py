from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import pytest

from ludic.agents.tool_agent import ToolAgent
from ludic.context.full_dialog import FullDialog
from ludic.inference.client import ChatClient, ChatResponse
from ludic.inference.request import ChatCompletionRequest, TokenCompletionRequest
from ludic.inference.tool_parser import HermesToolParser

from tests._mocks import _mock_parser, calculator_tool, MockChatTemplate


class DummyClient(ChatClient):
    """Dummy client supporting both complete() and complete_tokens()."""

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        return ChatResponse(text="ok"), {"raw_response": {"choices": [{"message": {"content": "ok"}}]}}

    async def complete_tokens(
        self,
        request: TokenCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        return ChatResponse(
            text="ok",
            prompt_token_ids=request.prompt_token_ids,
            completion_token_ids=[100, 101],
        ), {"mode": "token_in"}

    def sync_weights(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError


@pytest.mark.asyncio
async def test_tool_agent_helpers_and_execution():
    """Test ToolAgent helper methods work correctly."""
    ctx = FullDialog()
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ToolAgent(
        client=DummyClient(),
        model="mock",
        ctx=ctx,
        parser=_mock_parser,
        tools=[calculator_tool],
        chat_template=chat_template,
    )

    # _with_tools enforces return_token_ids for drift-free training.
    inf = agent._with_tools(None)
    assert inf.return_.return_token_ids is True

    # _tool_request advertises schemas and tool choice.
    tool_req = agent._tool_request()
    assert tool_req.tool_choice == "auto"
    assert len(tool_req.tools) == 1

    # _run_tool_calls executes and records tool results
    tool_calls_payload = [
        {
            "id": "call_1",
            "function": {
                "name": "calculator_tool",
                "arguments": json.dumps({"a": 2, "b": 3}),
            },
        }
    ]
    agent._run_tool_calls(tool_calls_payload)

    assert ctx.messages[-1]["role"] == "tool"
    assert ctx.messages[-1]["content"] == "5"
    assert ctx.messages[-1]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_tool_agent_schema_generation():
    """Test that tool schemas are correctly generated from Python functions."""
    ctx = FullDialog()
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    def my_tool(name: str, count: int, enabled: bool = True) -> str:
        """Does something useful."""
        return f"{name}: {count}"

    agent = ToolAgent(
        client=DummyClient(),
        model="mock",
        ctx=ctx,
        parser=_mock_parser,
        tools=[my_tool],
        chat_template=chat_template,
    )

    schemas = agent.tool_schemas
    assert len(schemas) == 1

    schema = schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "my_tool"
    assert schema["function"]["description"] == "Does something useful."

    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "name" in params["properties"]
    assert "count" in params["properties"]
    assert "enabled" in params["properties"]
    assert params["properties"]["name"]["type"] == "string"
    assert params["properties"]["count"]["type"] == "integer"
    assert params["properties"]["enabled"]["type"] == "boolean"
    assert "name" in params["required"]
    assert "count" in params["required"]
    assert "enabled" not in params["required"]  # Has default


@pytest.mark.asyncio
async def test_tool_agent_uses_token_in_when_template_provided():
    """Test that ToolAgent uses token-in API when chat_template is set."""

    class TrackingClient(ChatClient):
        def __init__(self):
            self.complete_called = False
            self.complete_tokens_called = False

        async def complete(self, request: ChatCompletionRequest):
            self.complete_called = True
            return ChatResponse(text="ok"), {}

        async def complete_tokens(self, request: TokenCompletionRequest):
            self.complete_tokens_called = True
            return ChatResponse(
                text="ok",
                prompt_token_ids=request.prompt_token_ids,
            ), {"mode": "token_in"}

        def sync_weights(self, *args, **kwargs):
            pass

    client = TrackingClient()
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ToolAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=_mock_parser,
        tools=[calculator_tool],
        chat_template=chat_template,
    )

    await agent.act()

    assert client.complete_tokens_called is True
    assert client.complete_called is False


@pytest.mark.asyncio
async def test_tool_agent_uses_legacy_when_no_template():
    """Test that ToolAgent uses legacy complete() when no chat_template."""

    class TrackingClient(ChatClient):
        def __init__(self):
            self.complete_called = False
            self.complete_tokens_called = False

        async def complete(self, request: ChatCompletionRequest):
            self.complete_called = True
            return ChatResponse(text="ok"), {"raw_response": {"choices": [{"message": {"content": "ok"}}]}}

        async def complete_tokens(self, request: TokenCompletionRequest):
            self.complete_tokens_called = True
            return ChatResponse(text="ok"), {}

        def sync_weights(self, *args, **kwargs):
            pass

    client = TrackingClient()

    agent = ToolAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=_mock_parser,
        tools=[calculator_tool],
        # No chat_template - uses legacy mode
    )

    await agent.act()

    assert client.complete_called is True
    assert client.complete_tokens_called is False
