from __future__ import annotations

import pytest

from ludic.agents import ReToolAgent, ReToolParser, CodeExecutionResult
from ludic.context.full_dialog import FullDialog
from ludic.parsers import boxed_parser
from ludic.inference.client import ChatClient, ChatResponse
from ludic.inference.request import ChatCompletionRequest
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------
# ReToolParser tests
# ---------------------------------------------------------------------


def test_retool_parser_extracts_code_block():
    parser = ReToolParser()
    text = "Let me calculate: <code>\n```python\nprint(2+2)\n```\n</code>"
    result = parser.parse(text)

    assert result.has_code
    assert result.code_block == "print(2+2)"
    assert result.final_answer is None


def test_retool_parser_extracts_plain_fenced_code_block():
    parser = ReToolParser()
    text = "Compute:\n```python\nprint(2+2)\n```\n"
    result = parser.parse(text)

    assert result.has_code
    assert result.code_block == "print(2+2)"


def test_retool_parser_extracts_boxed_answer():
    parser = ReToolParser()
    text = "The answer is \\boxed{42}."
    result = parser.parse(text)

    assert not result.has_code
    assert result.code_block is None
    assert result.final_answer == "42"


def test_retool_parser_supports_nested_braces_in_boxed():
    parser = ReToolParser()
    text = "Answer: \\boxed{\\frac{1}{2}}"
    result = parser.parse(text)

    assert not result.has_code
    assert result.final_answer == "\\frac{1}{2}"


def test_retool_parser_uses_last_box():
    parser = ReToolParser()
    text = "Intermediate: \\boxed{0} Final: \\boxed{1}"
    result = parser.parse(text)

    assert not result.has_code
    assert result.final_answer == "1"


def test_retool_parser_prioritizes_code_over_answer():
    parser = ReToolParser()
    text = "<code>\n```python\nx = 10\n```\n</code> Answer is \\boxed{20}."
    result = parser.parse(text)

    # When code is present, return code even if answer also exists
    assert result.has_code
    assert result.code_block == "x = 10"


def test_retool_parser_no_code_no_answer():
    parser = ReToolParser()
    text = "Just regular text here."
    result = parser.parse(text)

    assert not result.has_code
    assert result.code_block is None
    assert result.final_answer is None


# ---------------------------------------------------------------------
# ReToolAgent tests
# ---------------------------------------------------------------------


class ReToolMockClient(ChatClient):
    """Mock client that returns predetermined responses for ReTool tests."""

    def __init__(
        self,
        responses: list[str],
    ) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        if self.call_count >= len(self.responses):
            raise RuntimeError("Unexpected call to mock client")

        text = self.responses[self.call_count]
        self.call_count += 1

        resp = ChatResponse(
            text=text,
            finish_reason="stop",
            prompt_token_ids=list(range(len(request.messages))),
            completion_token_ids=list(range(10, 10 + len(text))),
        )
        return resp, {}

    def sync_weights(self, *args, **kwargs) -> str:
        """Mock implementation of sync_weights."""
        return "mock-version"


async def mock_code_sandbox(code: str) -> CodeExecutionResult:
    """Simple mock sandbox that evaluates simple Python code."""
    try:
        exec_globals = {}
        exec(code, exec_globals)
        if "result" in exec_globals:
            output = str(exec_globals["result"])
        else:
            output = "(no output)"
        return CodeExecutionResult(output=output, success=True)
    except Exception as e:
        return CodeExecutionResult(
            output="",
            success=False,
            error=str(e),
        )


@pytest.mark.asyncio
async def test_retool_agent_single_code_block_then_answer():
    responses = [
        "Let me calculate: <code>\n```python\nresult = 2 + 2\n```\n</code>",
        "So the answer is \\boxed{4}.",
    ]
    client = ReToolMockClient(responses)

    agent = ReToolAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=boxed_parser,
        code_sandbox=mock_code_sandbox,
        max_code_blocks=10,
    )

    obs = "What is 2 + 2?"
    info = {}
    agent.reset()
    agent.on_env_reset(obs, info)

    result = await agent.act()

    assert len(result.steps) == 2

    # First step: code block (internal)
    step0 = result.steps[0]
    assert step0.action_target == "internal"
    assert "<code>" in step0.action
    assert step0.info.get("interpreter_output") == "4"
    assert step0.info.get("code_success") is True

    # Second step: final answer (env)
    step1 = result.steps[1]
    assert step1.action_target == "env"
    assert step1.parse_result.action == "4"


@pytest.mark.asyncio
async def test_retool_agent_direct_answer_no_code():
    responses = [
        "The answer is \\boxed{42}.",
    ]
    client = ReToolMockClient(responses)

    agent = ReToolAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=boxed_parser,
        code_sandbox=mock_code_sandbox,
        max_code_blocks=10,
    )

    obs = "What is 6 * 7?"
    info = {}
    agent.reset()
    agent.on_env_reset(obs, info)

    result = await agent.act()

    assert len(result.steps) == 1
    assert result.steps[0].action_target == "env"
    assert result.steps[0].parse_result.action == "42"


@pytest.mark.asyncio
async def test_retool_agent_respects_max_code_blocks():
    responses = [
        "<code>\n```python\nresult = 1\n```\n</code>",
        "<code>\n```python\nresult = 2\n```\n</code>",
        "<code>\n```python\nresult = 3\n```\n</code>",
        "\\boxed{3}",
    ]
    client = ReToolMockClient(responses)

    agent = ReToolAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=boxed_parser,
        code_sandbox=mock_code_sandbox,
        max_code_blocks=2,  # Only allow 2 code blocks
    )

    obs = "Calculate..."
    info = {}
    agent.reset()
    agent.on_env_reset(obs, info)

    result = await agent.act()

    # Should execute 2 code blocks, then force final answer
    assert len(result.steps) == 3
    assert result.steps[0].action_target == "internal"
    assert result.steps[1].action_target == "internal"
    assert result.steps[2].action_target == "env"  # Forced final answer


@pytest.mark.asyncio
async def test_retool_agent_code_execution_error():
    responses = [
        "<code>\n```python\n1 / 0\n```\n</code>",
        "Let me fix that: \\boxed{1}.",
    ]
    client = ReToolMockClient(responses)

    async def error_sandbox(code: str) -> CodeExecutionResult:
        return CodeExecutionResult(
            output="",
            success=False,
            error="division by zero",
        )

    agent = ReToolAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=boxed_parser,
        code_sandbox=error_sandbox,
        max_code_blocks=10,
    )

    obs = "Calculate..."
    info = {}
    agent.reset()
    agent.on_env_reset(obs, info)

    result = await agent.act()

    assert len(result.steps) == 2
    assert result.steps[0].action_target == "internal"
    assert result.steps[0].info.get("code_success") is False
    assert "division by zero" in result.steps[0].info.get("code_error", "")


@pytest.mark.asyncio
async def test_retool_agent_parse_failure_no_code_no_answer():
    responses = [
        "I don't know how to answer this.",
    ]
    client = ReToolMockClient(responses)

    agent = ReToolAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=boxed_parser,
        code_sandbox=mock_code_sandbox,
        max_code_blocks=10,
    )

    obs = "Calculate..."
    info = {}
    agent.reset()
    agent.on_env_reset(obs, info)

    result = await agent.act()

    assert len(result.steps) == 1
    assert result.steps[0].action_target == "env"
    assert result.steps[0].parse_result.action is None  # Parse failed
    assert result.steps[0].parse_result.reward < 0
    assert result.steps[0].info.get("parse_error") is True
