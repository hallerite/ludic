from __future__ import annotations
import json
import pytest
from typing import Any, List, Dict, Optional, Tuple

from ludic.agents.react_agent import ReActAgent
from ludic.context.full_dialog import FullDialog
from ludic.inference.client import ChatClient, ChatResponse
from ludic.inference.request import ChatCompletionRequest
from ludic.types import Message
from ludic.parsers import xml_tag_parser
from tests._mocks import calculator_tool

# ---------------------------------------------------------------------
# Mocks & Helpers
# ---------------------------------------------------------------------

class ReplayMockClient(ChatClient):
    """
    A mock client that replays a sequence of pre-defined responses.
    It simulates the vLLM raw_response structure required by ReActAgent.
    """
    def __init__(self, steps: List[Dict[str, Any]]):
        # Steps is a list of dicts representing the "message" content
        # e.g. [{"content": "...", "tool_calls": [...]}, ...]
        self.steps = steps
        self.call_count = 0
        self.last_request: Optional[ChatCompletionRequest] = None
        self.last_messages: Optional[List[Message]] = None

    async def complete(
        self,
        request: ChatCompletionRequest,
        **kwargs,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        
        if self.call_count >= len(self.steps):
            raise RuntimeError("ReplayMockClient exhausted all steps")

        step_data = self.steps[self.call_count]
        self.call_count += 1
        
        self.last_request = request
        self.last_messages = request.messages

        # Construct ChatResponse (High level)
        text = step_data.get("content") or ""
        
        # Construct the raw vLLM/OpenAI structure (Low level)
        # ReActAgent inspects info["raw_response"]["choices"][0]["message"]
        raw_response = {
            "choices": [
                {
                    "message": step_data,
                    "finish_reason": "tool_calls" if step_data.get("tool_calls") else "stop"
                }
            ]
        }

        resp = ChatResponse(text=text)
        return resp, {"raw_response": raw_response}

    def sync_weights(self, *args, **kwargs):
        pass

def weather_tool(location: str) -> str:
    """Gets weather."""
    return f"Sunny in {location}"

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_react_agent_happy_path_single_tool():
    """
    Scenario:
    1. Agent Thinks -> Calls Calculator
    2. Tool executes -> Returns result
    3. Agent Thinks -> Returns Final Answer
    """
    
    # Step 1: Model wants to call tool
    step_1 = {
        "content": "I need to calculate 2+2.",
        "tool_calls": [
            {
                "id": "call_123",
                "function": {
                    "name": "calculator_tool",
                    "arguments": json.dumps({"a": 2, "b": 2})
                }
            }
        ]
    }
    
    # Step 2: Model sees result "4" and answers
    step_2 = {
        "content": "The answer is <move>4</move>",
        "tool_calls": None
    }

    client = ReplayMockClient([step_1, step_2])
    
    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool]
    )

    parse_result, raw_text, _, _ = await agent.act()

    # Assert Final Output
    assert parse_result.action == "4"
    assert raw_text == "The answer is <move>4</move>"
    
    # Assert Context State
    # 0: User (Env reset - assumed handled by protocol, but we called act directly so context had prompts)
    # Actually, let's check the messages sent on the FINAL call
    final_history = client.last_messages
    
    # We expect: 
    # ... System/User prompt ...
    # Assistant: "I need to calculate..." + tool_calls
    # Tool: "4"
    
    assert final_history[-2]["role"] == "assistant"
    assert final_history[-2]["tool_calls"][0]["function"]["name"] == "calculator_tool"
    
    assert final_history[-1]["role"] == "tool"
    assert final_history[-1]["content"] == "4"
    assert final_history[-1]["tool_call_id"] == "call_123"


@pytest.mark.asyncio
async def test_react_agent_shot_clock_fallback():
    """
    Scenario:
    - max_react_steps = 2
    - Step 1: Tool Call
    - Step 2: (Limit reached) -> Agent should FORCE final text, strip tools.
    """
    
    # Step 1: Tool Call
    step_1 = {
        "content": "Thinking...",
        "tool_calls": [
            {
                "id": "call_A",
                "function": {"name": "weather_tool", "arguments": '{"location": "NYC"}'}
            }
        ]
    }
    
    # Step 2: Model tries to call tool again? 
    # NO, the Agent logic should prevent the model from seeing tools in args.
    # The mock will simulate the model reacting to the "You must output now" prompt.
    step_2 = {
        "content": "Okay, fine. <move>Sunny</move>",
        "tool_calls": None
    }

    client = ReplayMockClient([step_1, step_2])
    
    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[weather_tool],
        max_react_steps=2 # Strict limit
    )

    result, _, _, _ = await agent.act()

    # 1. Did it finish?
    assert result.action == "Sunny"

    # 2. Was the "Shot Clock" logic triggered?
    # On the last call, "tools" should NOT be in extras
    last_req = client.last_request
    assert last_req is not None
    assert last_req.tools is None

    # 3. Was the forced message injected?
    last_msgs = client.last_messages
    assert last_msgs[-1]["role"] == "user"
    assert "exhausted your reasoning steps" in last_msgs[-1]["content"]


@pytest.mark.asyncio
async def test_react_agent_handles_bad_tool_calls():
    """
    Scenario: Model tries to call a non-existent tool or passes bad JSON.
    Agent should record the error in context and let model try again.
    """
    
    # Step 1: Call missing tool
    step_1 = {
        "content": "Trying weird tool",
        "tool_calls": [{
            "id": "call_bad",
            "function": {"name": "ghost_tool", "arguments": "{}"}
        }]
    }
    
    # Step 2: Model sees error, fixes it
    step_2 = {
        "content": "Oops. <move>Fixed</move>",
        "tool_calls": None
    }

    client = ReplayMockClient([step_1, step_2])
    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool] # ghost_tool not here
    )

    await agent.act()

    # Check that the error was added to context
    # Messages sent during Step 2 should include the Tool output from Step 1
    msgs = client.last_messages
    tool_msg = msgs[-1]
    
    assert tool_msg["role"] == "tool"
    assert "Error: Tool ghost_tool not found" in tool_msg["content"]


@pytest.mark.asyncio
async def test_react_agent_requires_tool_supported_context():
    """
    Ensures the agent raises TypeError if init'd with a context 
    that doesn't support tools.
    """
    
    class DumbDialog(FullDialog):
        @property
        def supports_tools(self) -> bool:
            return False

    with pytest.raises(TypeError, match="requires a context with supports_tools=True"):
        ReActAgent(
            client=ReplayMockClient([]),
            model="mock",
            ctx=DumbDialog(), # <-- The culprit
            parser=xml_tag_parser("move"),
            tools=[]
        )


@pytest.mark.asyncio
async def test_react_agent_records_bad_json_tool_arguments():
    """
    If the model emits invalid JSON arguments for a tool call, the agent
    should catch the decode error and record it as a tool result.
    """
    step_1 = {
        "content": "Trying tool with bad args",
        "tool_calls": [
            {
                "id": "call_badjson",
                "function": {
                    "name": "calculator_tool",
                    "arguments": "not json",
                },
            }
        ],
    }
    step_2 = {"content": "Done <move>ok</move>", "tool_calls": None}

    client = ReplayMockClient([step_1, step_2])
    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool],
        max_react_steps=3,
    )

    result, _, _, _ = await agent.act()
    assert result.action == "ok"

    final_history = client.last_messages
    tool_msg = final_history[-1]
    assert tool_msg["role"] == "tool"
    assert "Error executing tool calculator_tool" in tool_msg["content"]


@pytest.mark.asyncio
async def test_react_agent_records_tool_execution_exception():
    """
    If a tool raises during execution, the agent should record the exception
    in a tool message and continue the loop.
    """

    def explode_tool(x: int) -> int:
        """Always raises."""
        raise ValueError("boom")

    step_1 = {
        "content": "Use explode",
        "tool_calls": [
            {
                "id": "call_explode",
                "function": {
                    "name": "explode_tool",
                    "arguments": json.dumps({"x": 1}),
                },
            }
        ],
    }
    step_2 = {"content": "Recovered <move>ok</move>", "tool_calls": None}

    client = ReplayMockClient([step_1, step_2])
    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[explode_tool],
        max_react_steps=3,
    )

    result, _, _, _ = await agent.act()
    assert result.action == "ok"

    final_history = client.last_messages
    tool_msg = final_history[-1]
    assert tool_msg["role"] == "tool"
    assert "Error executing tool explode_tool" in tool_msg["content"]
    assert "boom" in tool_msg["content"]


@pytest.mark.asyncio
async def test_react_agent_handles_multiple_tool_calls_in_one_turn():
    """
    If the model emits multiple tool calls in a single assistant message,
    the agent should execute all of them and record one tool message per call.
    """
    step_1 = {
        "content": "Need both tools.",
        "tool_calls": [
            {
                "id": "call_calc",
                "function": {
                    "name": "calculator_tool",
                    "arguments": json.dumps({"a": 2, "b": 2}),
                },
            },
            {
                "id": "call_weather",
                "function": {
                    "name": "weather_tool",
                    "arguments": json.dumps({"location": "NYC"}),
                },
            },
        ],
    }
    step_2 = {"content": "<move>4 and Sunny in NYC</move>", "tool_calls": None}

    client = ReplayMockClient([step_1, step_2])
    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool, weather_tool],
        max_react_steps=3,
    )

    result, _, _, _ = await agent.act()
    assert result.action == "4 and Sunny in NYC"

    final_history = client.last_messages
    assert final_history[-3]["role"] == "assistant"
    assert len(final_history[-3]["tool_calls"]) == 2
    assert final_history[-3]["tool_calls"][0]["function"]["name"] == "calculator_tool"
    assert final_history[-3]["tool_calls"][1]["function"]["name"] == "weather_tool"

    assert final_history[-2]["role"] == "tool"
    assert final_history[-2]["tool_call_id"] == "call_calc"
    assert final_history[-2]["name"] == "calculator_tool"
    assert final_history[-2]["content"] == "4"

    assert final_history[-1]["role"] == "tool"
    assert final_history[-1]["tool_call_id"] == "call_weather"
    assert final_history[-1]["name"] == "weather_tool"
    assert final_history[-1]["content"] == "Sunny in NYC"


@pytest.mark.asyncio
async def test_react_agent_multi_tool_calls_continue_on_error():
    """
    If one tool call fails (missing tool / bad args / exception), the agent
    should record the error and still execute remaining tool calls.
    """
    step_1 = {
        "content": "Call missing tool then real one.",
        "tool_calls": [
            {
                "id": "call_missing",
                "function": {"name": "ghost_tool", "arguments": "{}"},
            },
            {
                "id": "call_calc",
                "function": {
                    "name": "calculator_tool",
                    "arguments": json.dumps({"a": 1, "b": 3}),
                },
            },
        ],
    }
    step_2 = {"content": "<move>ok</move>", "tool_calls": None}

    client = ReplayMockClient([step_1, step_2])
    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool],
        max_react_steps=3,
    )

    result, _, _, _ = await agent.act()
    assert result.action == "ok"

    final_history = client.last_messages
    assert final_history[-2]["role"] == "tool"
    assert final_history[-2]["tool_call_id"] == "call_missing"
    assert final_history[-2]["name"] == "ghost_tool"
    assert "Error: Tool ghost_tool not found." in final_history[-2]["content"]

    assert final_history[-1]["role"] == "tool"
    assert final_history[-1]["tool_call_id"] == "call_calc"
    assert final_history[-1]["name"] == "calculator_tool"
    assert final_history[-1]["content"] == "4"
