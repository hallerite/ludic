from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import pytest

from ludic.agents.tool_agent import ToolAgent
from ludic.context.full_dialog import FullDialog
from ludic.inference.client import ChatClient, ChatResponse
from ludic.inference.request import ChatCompletionRequest

from tests._mocks import _mock_parser, calculator_tool


class DummyClient(ChatClient):
    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        return ChatResponse(text="ok"), {"raw_response": {"choices": [{"message": {"content": "ok"}}]}}

    def sync_weights(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError

@pytest.mark.asyncio
async def test_tool_agent_helpers_and_execution():
    ctx = FullDialog()
    agent = ToolAgent(
        client=DummyClient(),
        model="mock",
        ctx=ctx,
        parser=_mock_parser,
        tools=[calculator_tool],
    )

    # _with_tools enforces return_token_ids for drift-free training.
    inf = agent._with_tools(None)
    assert inf.return_.return_token_ids is True

    # _tool_request advertises schemas and tool choice.
    tool_req = agent._tool_request()
    assert tool_req.tool_choice == "auto"
    assert len(tool_req.tools) == 1

    # _extract_openai_message reads content/tool_calls
    info = {
        "raw_response": {
            "choices": [
                {"message": {"content": "hi", "tool_calls": [{"id": "c1"}]}}
            ]
        }
    }
    content, tool_calls = agent._extract_openai_message(info)
    assert content == "hi"
    assert tool_calls is not None and tool_calls[0]["id"] == "c1"

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
