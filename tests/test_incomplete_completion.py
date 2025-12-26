from __future__ import annotations

import pytest

from ludic.agents.base_agent import Agent
from ludic.context.full_dialog import FullDialog
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.parsers import ParseResult
from tests._mocks import MockClient, MockEnv, MockChatTemplate


def pass_through_parser(raw: str) -> ParseResult:
    return ParseResult(action=raw, reward=0.0, obs=None)


@pytest.mark.asyncio
async def test_agent_rejects_incomplete_completion_by_default() -> None:
    ctx = FullDialog()
    agent = Agent(
        client=MockClient(text="RAW", finish_reason="length"),
        model="mock",
        ctx=ctx,
        parser=pass_through_parser,
        incomplete_completion_penalty=-0.2,
        incomplete_completion_feedback="too long",
        chat_template=MockChatTemplate(),
    )
    agent.reset(system_prompt=None)
    agent.on_env_reset("obs", {})

    parse_result, raw, info, _ = await agent.act()

    assert raw == "RAW"
    assert parse_result.action is None
    assert parse_result.reward == pytest.approx(-0.2)
    assert parse_result.obs == "too long"
    assert info.get("finish_reason") == "length"
    assert info.get("incomplete_completion") is True
    assert {"role": "assistant", "content": "RAW"} in ctx.messages


@pytest.mark.asyncio
async def test_agent_can_allow_incomplete_completion() -> None:
    agent = Agent(
        client=MockClient(text="1", finish_reason="length"),
        model="mock",
        ctx=FullDialog(),
        parser=pass_through_parser,
        reject_incomplete_completions=False,
        chat_template=MockChatTemplate(),
    )
    agent.reset(system_prompt=None)
    agent.on_env_reset("obs", {})

    parse_result, raw, info, _ = await agent.act()

    assert raw == "1"
    assert parse_result.action == "1"
    assert info.get("finish_reason") == "length"
    assert "incomplete_completion" not in info


@pytest.mark.asyncio
async def test_single_agent_protocol_marks_incomplete_completion_as_parse_error() -> None:
    agent = Agent(
        client=MockClient(text="RAW", finish_reason="length"),
        model="mock",
        ctx=FullDialog(),
        parser=pass_through_parser,
        chat_template=MockChatTemplate(),
    )
    protocol = SingleAgentSyncProtocol(agent=agent)

    env = MockEnv(max_steps=10, target="1")
    rollouts = await protocol.run(env=env, max_steps=1)

    assert len(rollouts) == 1
    r = rollouts[0]
    assert len(r.steps) == 1

    step = r.steps[0]
    assert step.info.get("finish_reason") == "length"
    assert step.info.get("incomplete_completion") is True
    assert step.info.get("parse_error") is True

    # Time-limit truncation is recorded, but synthetic next_obs is preserved
    assert step.truncated is True
    assert step.info.get("truncation_reason") == "max_steps"
    assert step.next_obs is not None
