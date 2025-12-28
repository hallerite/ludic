from __future__ import annotations
import json
import pytest
from typing import Any, List, Dict, Optional, Tuple

from ludic.agents.react_agent import ReActAgent
from ludic.context.full_dialog import FullDialog
from ludic.inference.client import ChatClient, ChatResponse
from ludic.inference.request import TokenCompletionRequest
from ludic.inference.tool_parser import HermesToolParser
from ludic.parsers import xml_tag_parser
from tests._mocks import calculator_tool, MockChatTemplate

# ---------------------------------------------------------------------
# Mocks & Helpers
# ---------------------------------------------------------------------


class ReplayMockClient(ChatClient):
    """
    A mock client that replays a sequence of pre-defined responses.
    Supports token-in API via complete_tokens().
    """

    def __init__(self, steps: List[str]):
        """
        Args:
            steps: List of raw text responses the model will output.
                   Tool calls should be in Hermes format: <tool_call>...</tool_call>
        """
        self.steps = steps
        self.call_count = 0
        self.last_request: Optional[TokenCompletionRequest] = None
        self.last_prompt_text: Optional[str] = None

    async def complete_tokens(
        self,
        request: TokenCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        if self.call_count >= len(self.steps):
            raise RuntimeError("ReplayMockClient exhausted all steps")

        text = self.steps[self.call_count]
        self.call_count += 1

        self.last_request = request
        self.last_prompt_text = request.prompt_text

        # Determine finish reason based on content
        has_tool_call = "<tool_call>" in text
        finish_reason = "tool_calls" if has_tool_call else "stop"

        resp = ChatResponse(
            text=text,
            finish_reason=finish_reason,
            prompt_token_ids=request.prompt_token_ids,
            completion_token_ids=[100 + i for i in range(len(text) // 10 + 1)],
        )
        return resp, {"mode": "token_in"}

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
    1. Agent Thinks -> Calls Calculator via Hermes format
    2. Tool executes -> Returns result
    3. Agent Thinks -> Returns Final Answer
    """

    # Step 1: Model outputs a tool call in Hermes format
    step_1 = (
        "I need to calculate 2+2.\n"
        "<tool_call>\n"
        '{"name": "calculator_tool", "arguments": {"a": 2, "b": 2}}\n'
        "</tool_call>"
    )

    # Step 2: Model sees result "4" and answers
    step_2 = "The answer is <move>4</move>"

    client = ReplayMockClient([step_1, step_2])
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool],
        chat_template=chat_template,
    )

    act_result = await agent.act()
    final_step = act_result.final_step
    parse_result = final_step.parse_result
    raw_text = final_step.action

    # Assert Final Output
    assert parse_result.action == "4"
    assert raw_text == "The answer is <move>4</move>"

    # Assert the tool was called correctly
    assert client.call_count == 2

    # Check context has tool result
    ctx_messages = agent._ctx.messages
    tool_msg = [m for m in ctx_messages if m.get("role") == "tool"]
    assert len(tool_msg) == 1
    assert tool_msg[0]["content"] == "4"
    assert tool_msg[0]["tool_call_id"] == "call_0"


@pytest.mark.asyncio
async def test_react_agent_records_internal_steps():
    """Test that ReActAgent records internal tool call steps separately from final answer."""
    # Step 1: Tool call in Hermes format
    step_1 = (
        "Calling tool.\n"
        "<tool_call>\n"
        '{"name": "calculator_tool", "arguments": {"a": 2, "b": 3}}\n'
        "</tool_call>"
    )
    # Step 2: Final answer
    step_2 = "Final <move>5</move>"

    client = ReplayMockClient([step_1, step_2])
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool],
        chat_template=chat_template,
        max_react_steps=2,
    )

    act_result = await agent.act()
    assert len(act_result.steps) == 2

    internal_step = act_result.steps[0]
    final_step = act_result.steps[1]

    assert internal_step.action_target == "internal"
    assert internal_step.tool_calls is not None
    assert internal_step.tool_results is not None
    assert internal_step.tool_results[0]["content"] == "5"
    assert final_step.action_target == "env"
    assert final_step.parse_result.action == "5"


@pytest.mark.asyncio
async def test_react_agent_shot_clock_fallback():
    """
    Scenario:
    - max_react_steps = 2
    - Step 1: Tool Call
    - Step 2: (Limit reached) -> Agent should FORCE final text, strip tools.
    """

    # Step 1: Tool Call in Hermes format
    step_1 = (
        "Thinking...\n"
        "<tool_call>\n"
        '{"name": "weather_tool", "arguments": {"location": "NYC"}}\n'
        "</tool_call>"
    )

    # Step 2: Model is forced to output final answer (no tools)
    step_2 = "Okay, fine. <move>Sunny</move>"

    client = ReplayMockClient([step_1, step_2])
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[weather_tool],
        chat_template=chat_template,
        max_react_steps=2,  # Strict limit
    )

    act_result = await agent.act()
    result = act_result.final_step.parse_result

    # 1. Did it finish?
    assert result.action == "Sunny"

    # 2. Was the "Shot Clock" logic triggered?
    # On the last call, tools should NOT be passed
    last_prompt = client.last_prompt_text
    assert last_prompt is not None
    assert "exhausted your reasoning steps" in last_prompt


@pytest.mark.asyncio
async def test_react_agent_handles_bad_tool_calls():
    """
    Scenario: Model tries to call a non-existent tool.
    Agent should record the error in context and let model try again.
    """

    # Step 1: Call missing tool
    step_1 = (
        "Trying weird tool\n"
        "<tool_call>\n"
        '{"name": "ghost_tool", "arguments": {}}\n'
        "</tool_call>"
    )

    # Step 2: Model sees error, fixes it
    step_2 = "Oops. <move>Fixed</move>"

    client = ReplayMockClient([step_1, step_2])
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool],  # ghost_tool not here
        chat_template=chat_template,
    )

    await agent.act()

    # Check that the error was added to context
    ctx_messages = agent._ctx.messages
    tool_msgs = [m for m in ctx_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "Error: Tool ghost_tool not found" in tool_msgs[0]["content"]


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

    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    with pytest.raises(TypeError, match="requires a context with supports_tools=True"):
        ReActAgent(
            client=ReplayMockClient([]),
            model="mock",
            ctx=DumbDialog(),  # <-- The culprit
            parser=xml_tag_parser("move"),
            tools=[],
            chat_template=chat_template,
        )


@pytest.mark.asyncio
async def test_react_agent_records_bad_json_tool_arguments():
    """
    If the model emits invalid JSON in <tool_call>, the agent returns a parse error.
    """
    # Step 1: Bad JSON in tool call - parser will return None
    step_1 = (
        "Trying tool with bad args\n"
        "<tool_call>\n"
        "not valid json\n"
        "</tool_call>"
    )

    client = ReplayMockClient([step_1])
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool],
        chat_template=chat_template,
        max_react_steps=3,
    )

    act_result = await agent.act()
    final_step = act_result.final_step
    assert final_step.parse_result.action is None
    assert final_step.info.get("tool_parse_error") is True
    assert final_step.action == step_1


@pytest.mark.asyncio
async def test_react_agent_records_tool_execution_exception():
    """
    If a tool raises during execution, the agent should record the exception
    in a tool message and continue the loop.
    """

    def explode_tool(x: int) -> int:
        """Always raises."""
        raise ValueError("boom")

    step_1 = (
        "Use explode\n"
        "<tool_call>\n"
        '{"name": "explode_tool", "arguments": {"x": 1}}\n'
        "</tool_call>"
    )
    step_2 = "Recovered <move>ok</move>"

    client = ReplayMockClient([step_1, step_2])
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[explode_tool],
        chat_template=chat_template,
        max_react_steps=3,
    )

    act_result = await agent.act()
    result = act_result.final_step.parse_result
    assert result.action == "ok"

    ctx_messages = agent._ctx.messages
    tool_msgs = [m for m in ctx_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "Error executing tool explode_tool" in tool_msgs[0]["content"]
    assert "boom" in tool_msgs[0]["content"]


@pytest.mark.asyncio
async def test_react_agent_handles_multiple_tool_calls_in_one_turn():
    """
    If the model emits multiple <tool_call> blocks in a single response,
    the agent should execute all of them.
    """
    step_1 = (
        "Need both tools.\n"
        "<tool_call>\n"
        '{"name": "calculator_tool", "arguments": {"a": 2, "b": 2}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "weather_tool", "arguments": {"location": "NYC"}}\n'
        "</tool_call>"
    )
    step_2 = "<move>4 and Sunny in NYC</move>"

    client = ReplayMockClient([step_1, step_2])
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool, weather_tool],
        chat_template=chat_template,
        max_react_steps=3,
    )

    act_result = await agent.act()
    result = act_result.final_step.parse_result
    assert result.action == "4 and Sunny in NYC"

    ctx_messages = agent._ctx.messages
    tool_msgs = [m for m in ctx_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["content"] == "4"
    assert tool_msgs[0]["name"] == "calculator_tool"
    assert tool_msgs[1]["content"] == "Sunny in NYC"
    assert tool_msgs[1]["name"] == "weather_tool"


@pytest.mark.asyncio
async def test_react_agent_multi_tool_calls_continue_on_error():
    """
    If one tool call fails (missing tool), the agent should record the error
    and still execute remaining tool calls.
    """
    step_1 = (
        "Call missing tool then real one.\n"
        "<tool_call>\n"
        '{"name": "ghost_tool", "arguments": {}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "calculator_tool", "arguments": {"a": 1, "b": 3}}\n'
        "</tool_call>"
    )
    step_2 = "<move>ok</move>"

    client = ReplayMockClient([step_1, step_2])
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool],
        chat_template=chat_template,
        max_react_steps=3,
    )

    act_result = await agent.act()
    result = act_result.final_step.parse_result
    assert result.action == "ok"

    ctx_messages = agent._ctx.messages
    tool_msgs = [m for m in ctx_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2

    assert tool_msgs[0]["name"] == "ghost_tool"
    assert "Error: Tool ghost_tool not found." in tool_msgs[0]["content"]

    assert tool_msgs[1]["name"] == "calculator_tool"
    assert tool_msgs[1]["content"] == "4"


@pytest.mark.asyncio
async def test_react_agent_no_tool_call_returns_immediately():
    """
    If the model's first response has no <tool_call>, agent returns immediately.
    """
    step_1 = "I know the answer: <move>42</move>"

    client = ReplayMockClient([step_1])
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool],
        chat_template=chat_template,
    )

    act_result = await agent.act()
    assert act_result.final_step.parse_result.action == "42"
    assert client.call_count == 1  # Only one call needed


# ---------------------------------------------------------------------
# Env Tools Tests
# ---------------------------------------------------------------------


def move_tool(direction: str) -> str:
    """Move in a direction (env tool - not executed by agent)."""
    return f"moved {direction}"


@pytest.mark.asyncio
async def test_react_agent_env_tool_returns_immediately():
    """
    When an env tool is called, the agent should return immediately
    with action_target='env' without executing the tool locally.
    """
    # Agent calls an env tool
    step_1 = (
        "I will move north.\n"
        "<tool_call>\n"
        '{"name": "move_tool", "arguments": {"direction": "north"}}\n'
        "</tool_call>"
    )

    client = ReplayMockClient([step_1])
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool],           # calculator is internal
        external_tools=[move_tool],        # move is external (handled by protocol)
        chat_template=chat_template,
        max_react_steps=3,
    )

    act_result = await agent.act()

    # Should return immediately with one step
    assert len(act_result.steps) == 1
    step = act_result.steps[0]

    # Step should target external (protocol will handle)
    assert step.action_target == "external"
    assert step.tool_calls is not None
    assert len(step.tool_calls) == 1
    assert step.tool_calls[0]["function"]["name"] == "move_tool"

    # Tool should NOT be executed (no tool_results for external tools)
    assert step.tool_results is None

    # External tool calls have parse_result=None - they're not final actions.
    # The protocol handles them and feeds results back to the agent.
    assert step.parse_result is None

    # Only one inference call (no loop continuation)
    assert client.call_count == 1


@pytest.mark.asyncio
async def test_react_agent_internal_tool_continues_loop():
    """
    Internal tools should be executed and the loop should continue.
    """
    # Step 1: Internal tool call (calculator)
    step_1 = (
        "Let me calculate.\n"
        "<tool_call>\n"
        '{"name": "calculator_tool", "arguments": {"a": 2, "b": 3}}\n'
        "</tool_call>"
    )
    # Step 2: Final answer
    step_2 = "The answer is <move>5</move>"

    client = ReplayMockClient([step_1, step_2])
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool],
        external_tools=[move_tool],
        chat_template=chat_template,
        max_react_steps=3,
    )

    act_result = await agent.act()

    # Should have two steps: internal tool + final answer
    assert len(act_result.steps) == 2

    # First step: internal tool (executed)
    internal_step = act_result.steps[0]
    assert internal_step.action_target == "internal"
    assert internal_step.tool_calls is not None
    assert internal_step.tool_results is not None
    assert internal_step.tool_results[0]["content"] == "5"

    # Second step: final answer to environment
    final_step = act_result.steps[1]
    assert final_step.action_target == "env"
    assert final_step.parse_result is not None
    assert final_step.parse_result.action == "5"

    # Two inference calls
    assert client.call_count == 2


@pytest.mark.asyncio
async def test_react_agent_mixed_tool_calls_executes_internal_first():
    """
    When both internal and env tools are called together,
    internal tools should be executed but env tools should not.
    """
    # Agent calls both calculator (internal) and move (env) in one turn
    step_1 = (
        "Calculating and moving.\n"
        "<tool_call>\n"
        '{"name": "calculator_tool", "arguments": {"a": 1, "b": 1}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "move_tool", "arguments": {"direction": "east"}}\n'
        "</tool_call>"
    )

    client = ReplayMockClient([step_1])
    chat_template = MockChatTemplate(tool_parser=HermesToolParser())

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=xml_tag_parser("move"),
        tools=[calculator_tool],
        external_tools=[move_tool],
        chat_template=chat_template,
        max_react_steps=3,
    )

    act_result = await agent.act()

    # Should return immediately (external tool present)
    assert len(act_result.steps) == 1
    step = act_result.steps[0]
    assert step.action_target == "external"

    # Both tool calls should be recorded
    assert step.tool_calls is not None
    assert len(step.tool_calls) == 2

    # Only internal tool should have been executed
    assert step.tool_results is not None
    assert len(step.tool_results) == 1
    assert step.tool_results[0]["tool_name"] == "calculator_tool"
    assert step.tool_results[0]["content"] == "2"
