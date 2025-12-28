from typing import Dict, List, Tuple, Optional
import pytest

from ludic.context.full_dialog import FullDialog
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.interaction.multi_agent import MultiAgentProtocol
from ludic.agents.base_agent import Agent
from ludic.inference.client import ChatResponse
from ludic.parsers import (
    ParseResult,
    xml_tag_parser,
    compose_parsers,
    Parser,
    think_prefix_parser,
)
from ludic.types import StepOutcome, Info
from ludic.envs.env import LudicEnv
from tests._mocks import MockEnv, MockClient, MockAgent, MockChatTemplate


# ---------------------------------------------------------------------
# Basic env/agent termination cases (Single Agent)
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_terminates_immediately():
    env = MockEnv(max_steps=3, target="1")
    # MockAgent provides a default ctx and a pass-through parser
    agent = MockAgent(client=MockClient(text="1"))
    protocol = SingleAgentSyncProtocol(agent=agent)

    # run() now returns List[Rollout]
    rollouts = await protocol.run(
        env=env,
        max_steps=5,
    )

    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.steps[-1].terminated is True
    assert rollout.total_reward == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_truncation_when_agent_is_wrong():
    class WrongClient(MockClient):
        async def complete_tokens(self, request, **kwargs):
            return ChatResponse(
                text="nope",
                prompt_token_ids=request.prompt_token_ids,
                completion_token_ids=[1, 2, 3],
            ), {"mode": "token_in"}

    env = MockEnv(max_steps=2, target="1")
    agent = MockAgent(client=WrongClient())
    protocol = SingleAgentSyncProtocol(agent=agent)

    rollouts = await protocol.run(
        env=env,
        max_steps=10,
    )

    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.steps[-1].truncated is True
    assert rollout.total_reward < 0.0


# ---------------------------------------------------------------------
# Parser integration test
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_episode_uses_action_parser_and_logs_parsed_action():
    """
    Ensure that:
      - protocol.run() uses the agent's configured parser
      - Step.action keeps the raw LLM text
      - Step.info['parsed_action'] is the parsed action
      - parser reward is added to env reward
      - env.step receives the parsed action
    """

    env = MockEnv(max_steps=3, target="A1")

    # LLM emits a valid CoT-prefixed XML move
    raw_llm_output = "<think>some reasoning</think>\n<move>  A1  </move>"

    action_parser: Parser = compose_parsers(
        think_prefix_parser,
        xml_tag_parser("move"),
    )

    agent = Agent(
        client=MockClient(text=raw_llm_output),
        model="mock",
        ctx=FullDialog(),
        parser=action_parser,
        chat_template=MockChatTemplate(),
    )
    
    protocol = SingleAgentSyncProtocol(agent=agent)

    rollouts = await protocol.run(
        env=env,
        max_steps=5,
    )

    assert len(rollouts) == 1
    rollout = rollouts[0]

    assert rollout.length >= 1
    step = rollout.steps[-1]

    # Raw LLM text must be preserved in the main action field
    assert "<think>" in step.action
    assert "<move>" in step.action

    # Parsed action must be logged in info
    assert step.info["parsed_action"] == "A1"

    # Env should terminate because parsed action == target "A1"
    assert step.terminated is True

    # Parser reward: +0.2 (two successful parsers); Env reward: 1.0
    assert rollout.total_reward == pytest.approx(1.2)


# ---------------------------------------------------------------------
# Multi-Agent Tests
# ---------------------------------------------------------------------

class MockTurnBasedEnv(LudicEnv[str, str, str]):
    """
    A simple 2-agent turn-based env for testing MultiAgentProtocol.
    Agents alternate turns. Each terminates after acting twice.
    Properly tracks terminated agents and excludes them from active_agents.
    """
    def __init__(self):
        self._turn = 0
        self._agents = ["agent_A", "agent_B"]
        self._agent_steps = {"agent_A": 0, "agent_B": 0}
        self._terminated_agents: set = set()

    @property
    def agent_ids(self) -> List[str]:
        return self._agents

    @property
    def active_agents(self) -> List[str]:
        # Return agents who haven't terminated yet, in turn order
        available = [a for a in self._agents if a not in self._terminated_agents]
        if not available:
            return []
        # Alternate between available agents
        return [available[self._turn % len(available)]]

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, Tuple[str, Info]]:
        self._turn = 0
        self._agent_steps = {"agent_A": 0, "agent_B": 0}
        self._terminated_agents = set()
        return {
            "agent_A": ("Start A", {}),
            "agent_B": ("Start B", {}),
        }

    def step(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
        if not actions:
            return {}

        active = list(actions.keys())[0]
        self._agent_steps[active] += 1

        # Each agent terminates after 2 actions
        terminated = self._agent_steps[active] >= 2
        if terminated:
            self._terminated_agents.add(active)

        reward = 1.0 if active == "agent_A" else 2.0

        outcome = StepOutcome(
            obs=f"Obs for {active} after step {self._agent_steps[active]}",
            reward=reward,
            truncated=False,
            terminated=terminated,
            info={"turn": self._turn, "agent_step": self._agent_steps[active]},
        )

        self._turn += 1
        return {active: outcome}


@pytest.mark.asyncio
async def test_multi_agent_returns_multiple_rollouts():
    """
    Test that MultiAgentProtocol correctly generates distinct rollouts
    for different agents in the same episode.

    With independent continuation, agents keep playing until they each
    terminate individually.
    """
    env = MockTurnBasedEnv()

    agent_A = MockAgent(client=MockClient(text="Move A"))
    agent_B = MockAgent(client=MockClient(text="Move B"))

    agents = {"agent_A": agent_A, "agent_B": agent_B}
    protocol = MultiAgentProtocol(agents)

    # Run for enough steps to let both act multiple times
    rollouts = await protocol.run(env=env, max_steps=10)

    # Expecting 2 rollouts: one for A, one for B
    assert len(rollouts) == 2

    # Identify which is which
    r_a = next(r for r in rollouts if r.meta["agent_id"] == "agent_A")
    r_b = next(r for r in rollouts if r.meta["agent_id"] == "agent_B")

    # Each agent terminates after 2 actions
    assert len(r_a.steps) == 2
    assert len(r_b.steps) == 2

    # Verify Agent A's trace
    assert r_a.steps[0].action == "Move A"
    assert r_a.steps[0].reward == 1.0
    assert r_a.steps[0].prev_obs == "Start A"
    assert r_a.steps[0].terminated is False
    assert r_a.steps[1].terminated is True

    # Verify Agent B's trace
    assert r_b.steps[0].action == "Move B"
    assert r_b.steps[0].reward == 2.0
    assert r_b.steps[0].prev_obs == "Start B"
    assert r_b.steps[0].terminated is False
    assert r_b.steps[1].terminated is True

    # Both rollouts should NOT be marked as truncated (they terminated naturally)
    assert r_a.meta["episode_truncated"] is False
    assert r_b.meta["episode_truncated"] is False

class MockMixedBotEnv(LudicEnv[str, str, str]):
    """
    Turn 0: 'bot' (unmanaged) acts.
    Turn 1: 'hero' (managed) acts.
    """
    def __init__(self):
        self._turn = 0
    
    @property
    def agent_ids(self) -> List[str]:
        return ["bot", "hero"]

    @property
    def active_agents(self) -> List[str]:
        return ["bot"] if self._turn == 0 else ["hero"]

    def reset(self, *, seed=None) -> Dict[str, Tuple[str, Info]]:
        self._turn = 0
        return {"hero": ("Start", {}), "bot": ("Start", {})}

    def step(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
        # If it's the bot's turn, actions should be empty coming from the protocol
        if self._turn == 0:
            assert actions == {}, "Protocol should not send actions for unmanaged bot"
            outcome = StepOutcome(obs="Bot attacked!", reward=0.0, truncated=False, terminated=False, info={})
            self._turn = 1
            # Both see the bot's move
            return {"bot": outcome, "hero": outcome}
        
        # Hero's turn
        if self._turn == 1:
            assert "hero" in actions
            outcome = StepOutcome(obs="Hero won", reward=10.0, truncated=False, terminated=True, info={})
            self._turn = 2
            return {"hero": outcome}
            
        return {}

@pytest.mark.asyncio
async def test_multi_agent_handles_unmanaged_bot_turns():
    """
    Ensure the protocol steps the env with empty actions when only
    unmanaged agents (bots) are active, but still updates managed agents.
    """
    env = MockMixedBotEnv()
    hero_agent = MockAgent(client=MockClient(text="Hero Attack"))
    
    # We only manage the 'hero', not the 'bot'
    protocol = MultiAgentProtocol(agents={"hero": hero_agent})

    rollouts = await protocol.run(env=env, max_steps=5)

    assert len(rollouts) == 1
    r = rollouts[0]
    
    assert r.meta["agent_id"] == "hero"
    # Hero should have 1 step (Turn 1), but the prev_obs should reflect Turn 0
    assert len(r.steps) == 1
    
    step = r.steps[0]
    # The hero's observation *before* acting should be the result of the bot's turn
    assert step.prev_obs == "Bot attacked!"
    assert step.action == "Hero Attack"
    assert step.reward == 10.0
    assert step.terminated is True


# ---------------------------------------------------------------------
# Parser failure handling (protocols)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_agent_protocol_logs_parser_failure_without_env_step():
    """
    If the agent parser fails, SingleAgentSyncProtocol should:
      - NOT call env.step()
      - log a synthetic step with parse_error info
      - feed the synthetic observation back to the agent context
    """

    class CountingEnv(MockEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_calls = 0

        def env_step(self, action: str) -> StepOutcome:  # type: ignore[override]
            self.step_calls += 1
            return super().env_step(action)

    def always_fail_parser(_: str) -> ParseResult:
        return ParseResult(action=None, reward=-0.5, obs="bad action")

    env = CountingEnv(max_steps=3, target="1")
    agent = Agent(
        client=MockClient(text="BADRAW"),
        model="mock",
        ctx=FullDialog(),
        parser=always_fail_parser,
        chat_template=MockChatTemplate(),
    )
    protocol = SingleAgentSyncProtocol(agent=agent)

    rollouts = await protocol.run(env=env, max_steps=1)

    assert env.step_calls == 0
    assert len(rollouts) == 1
    r = rollouts[0]
    assert len(r.steps) == 1

    step = r.steps[0]
    assert step.action == "BADRAW"
    assert step.next_obs == "bad action"  # Parser failure synthetic obs preserved
    assert step.reward == pytest.approx(-0.5)
    assert step.info.get("parse_error") is True
    assert step.terminated is False
    # With max_steps=1, the episode is truncated due to time limit
    assert step.truncated is True
    assert step.info.get("truncation_reason") == "max_steps"

    # Rollout metadata should reflect truncation
    assert r.meta.get("episode_truncated") is True
    assert r.meta.get("truncation_reason") == "max_steps"

    # Synthetic obs should be appended to agent context as user message.
    assert agent._ctx.messages[-1]["role"] == "user"
    assert agent._ctx.messages[-1]["content"] == "bad action"


@pytest.mark.asyncio
async def test_multi_agent_protocol_excludes_parse_fail_actions_and_logs_synthetic_step():
    """
    MultiAgentProtocol should omit invalid parsed actions from env.step(),
    but still log a synthetic step and update context for the failing agent.
    """

    class SimulEnv(LudicEnv[str, str, str]):
        def __init__(self) -> None:
            self._agents = ["A", "B"]
            self.last_actions: Dict[str, str] | None = None
            self._done = False

        @property
        def agent_ids(self) -> List[str]:
            return list(self._agents)

        @property
        def active_agents(self) -> List[str]:
            return [] if self._done else list(self._agents)

        def reset(self, *, seed: Optional[int] = None) -> Dict[str, Tuple[str, Info]]:
            self._done = False
            return {"A": ("obsA", {}), "B": ("obsB", {})}

        def step(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
            self.last_actions = dict(actions)
            assert "B" not in actions, "Parse-fail agent should not be sent to env.step"
            assert actions.get("A") == "GOOD"
            self._done = True
            return {
                "A": StepOutcome(obs="doneA", reward=1.0, truncated=False, terminated=True, info={}),
                "B": StepOutcome(obs="doneB", reward=0.0, truncated=False, terminated=True, info={}),
            }

    def always_fail_parser(_: str) -> ParseResult:
        return ParseResult(action=None, reward=-0.7, obs="bad action")

    env = SimulEnv()
    good_agent = MockAgent(client=MockClient(text="GOOD"))
    bad_agent = MockAgent(client=MockClient(text="BADRAW"), parser=always_fail_parser)

    protocol = MultiAgentProtocol(agents={"A": good_agent, "B": bad_agent})
    rollouts = await protocol.run(env=env, max_steps=3)

    assert env.last_actions == {"A": "GOOD"}
    assert len(rollouts) == 2

    r_a = next(r for r in rollouts if r.meta["agent_id"] == "A")
    r_b = next(r for r in rollouts if r.meta["agent_id"] == "B")

    assert len(r_a.steps) == 1
    assert r_a.steps[0].reward == pytest.approx(1.0)

    assert len(r_b.steps) == 1
    step_b = r_b.steps[0]
    assert step_b.action == "BADRAW"
    assert step_b.next_obs == "bad action"
    assert step_b.reward == pytest.approx(-0.7)
    assert step_b.info.get("parse_error") is True
    assert step_b.terminated is False


@pytest.mark.asyncio
async def test_multi_agent_protocol_all_parse_fail_does_not_step_env():
    """
    Regression test: if *all* active managed agents fail parsing in a turn,
    MultiAgentProtocol should NOT call env.step({}).
    """

    class StrictSimulEnv(LudicEnv[str, str, str]):
        def __init__(self) -> None:
            self._agents = ["A", "B"]
            self.step_calls = 0

        @property
        def agent_ids(self) -> List[str]:
            return list(self._agents)

        @property
        def active_agents(self) -> List[str]:
            return list(self._agents)

        def reset(self, *, seed: Optional[int] = None) -> Dict[str, Tuple[str, Info]]:
            return {"A": ("obsA", {}), "B": ("obsB", {})}

        def step(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
            self.step_calls += 1
            raise AssertionError("env.step should not be called when all actions are invalid")

    def always_fail_parser(_: str) -> ParseResult:
        return ParseResult(action=None, reward=-0.3, obs="bad action")

    env = StrictSimulEnv()
    agent_a = MockAgent(client=MockClient(text="RAW_A"), parser=always_fail_parser)
    agent_b = MockAgent(client=MockClient(text="RAW_B"), parser=always_fail_parser)

    protocol = MultiAgentProtocol(agents={"A": agent_a, "B": agent_b})
    rollouts = await protocol.run(env=env, max_steps=1)

    assert env.step_calls == 0
    assert len(rollouts) == 2

    r_a = next(r for r in rollouts if r.meta["agent_id"] == "A")
    r_b = next(r for r in rollouts if r.meta["agent_id"] == "B")

    assert len(r_a.steps) == 1
    assert r_a.steps[0].action == "RAW_A"
    assert r_a.steps[0].next_obs == "bad action"
    assert r_a.steps[0].reward == pytest.approx(-0.3)
    assert r_a.steps[0].info.get("parse_error") is True

    assert len(r_b.steps) == 1
    assert r_b.steps[0].action == "RAW_B"
    assert r_b.steps[0].next_obs == "bad action"
    assert r_b.steps[0].reward == pytest.approx(-0.3)
    assert r_b.steps[0].info.get("parse_error") is True


# ---------------------------------------------------------------------
# Truncation handling tests
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_agent_max_steps_truncation():
    """
    When the agent doesn't terminate the env within max_steps,
    the episode should be marked as truncated with truncation_reason='max_steps'.
    """
    # Agent always says "wrong", env wants "correct"
    env = MockEnv(max_steps=10, target="correct")  # env allows many steps
    agent = MockAgent(client=MockClient(text="wrong"))
    protocol = SingleAgentSyncProtocol(agent=agent)

    # Protocol max_steps=3, so we'll hit that before env's max_steps
    rollouts = await protocol.run(env=env, max_steps=3)

    assert len(rollouts) == 1
    r = rollouts[0]
    assert len(r.steps) == 3

    # Last step should be marked as truncated
    last_step = r.steps[-1]
    assert last_step.truncated is True
    assert last_step.terminated is False
    assert last_step.info.get("truncation_reason") == "max_steps"
    assert last_step.next_obs is None  # Truncated non-parser-failure step

    # Earlier steps should NOT be truncated
    for step in r.steps[:-1]:
        assert step.truncated is False
        assert step.terminated is False
        assert step.next_obs is not None

    # Rollout metadata
    assert r.meta.get("episode_truncated") is True
    assert r.meta.get("truncation_reason") == "max_steps"


@pytest.mark.asyncio
async def test_single_agent_env_truncation_preserved():
    """
    When the env itself truncates (e.g., env's max_steps), that should be
    recorded as truncation_reason='env'.
    """
    # Env will truncate after 2 wrong answers
    env = MockEnv(max_steps=2, target="correct")
    agent = MockAgent(client=MockClient(text="wrong"))
    protocol = SingleAgentSyncProtocol(agent=agent)

    # Protocol allows many steps, but env will truncate at 2
    rollouts = await protocol.run(env=env, max_steps=100)

    assert len(rollouts) == 1
    r = rollouts[0]
    assert len(r.steps) == 2

    last_step = r.steps[-1]
    assert last_step.truncated is True
    assert last_step.terminated is False

    # Rollout metadata should reflect env-initiated truncation
    assert r.meta.get("episode_truncated") is True
    assert r.meta.get("truncation_reason") == "env"


@pytest.mark.asyncio
async def test_single_agent_normal_termination_not_truncated():
    """
    When the agent correctly terminates the env, the episode should NOT
    be marked as truncated.
    """
    env = MockEnv(max_steps=10, target="win")
    agent = MockAgent(client=MockClient(text="win"))
    protocol = SingleAgentSyncProtocol(agent=agent)

    rollouts = await protocol.run(env=env, max_steps=100)

    assert len(rollouts) == 1
    r = rollouts[0]
    assert len(r.steps) == 1

    last_step = r.steps[-1]
    assert last_step.terminated is True
    assert last_step.truncated is False
    assert last_step.next_obs is None  # Terminal step

    # Rollout metadata
    assert r.meta.get("episode_truncated") is False
    assert r.meta.get("truncation_reason") is None


@pytest.mark.asyncio
async def test_multi_agent_max_steps_truncation():
    """
    MultiAgentProtocol should mark all agents' last steps as truncated
    when max_steps is hit.
    """

    class ContinuingEnv(LudicEnv[str, str, str]):
        """An env that never terminates on its own."""
        def __init__(self) -> None:
            self._agents = ["A", "B"]
            self._step = 0

        @property
        def agent_ids(self) -> List[str]:
            return list(self._agents)

        @property
        def active_agents(self) -> List[str]:
            return list(self._agents)

        def reset(self, *, seed: Optional[int] = None) -> Dict[str, Tuple[str, Info]]:
            self._step = 0
            return {"A": ("obsA", {}), "B": ("obsB", {})}

        def step(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
            self._step += 1
            return {
                aid: StepOutcome(
                    obs=f"obs_{self._step}",
                    reward=0.0,
                    truncated=False,
                    terminated=False,
                    info={},
                )
                for aid in actions
            }

    env = ContinuingEnv()
    agent_a = MockAgent(client=MockClient(text="action_a"))
    agent_b = MockAgent(client=MockClient(text="action_b"))

    protocol = MultiAgentProtocol(agents={"A": agent_a, "B": agent_b})
    rollouts = await protocol.run(env=env, max_steps=3)

    assert len(rollouts) == 2

    for r in rollouts:
        assert len(r.steps) == 3
        assert r.meta.get("episode_truncated") is True
        assert r.meta.get("truncation_reason") == "max_steps"

        # Last step truncated
        last_step = r.steps[-1]
        assert last_step.truncated is True
        assert last_step.info.get("truncation_reason") == "max_steps"
        assert last_step.next_obs is None

        # Earlier steps not truncated
        for step in r.steps[:-1]:
            assert step.truncated is False
            assert step.next_obs is not None


@pytest.mark.asyncio
async def test_multi_agent_independent_termination():
    """
    Test that when one agent terminates, the other continues playing
    until it also terminates or hits max_steps.
    """

    class AsymmetricEnv(LudicEnv[str, str, str]):
        """
        Agent A terminates on first action.
        Agent B never terminates (must hit max_steps).
        """
        def __init__(self) -> None:
            self._agents = ["A", "B"]
            self._terminated: set = set()
            self._step = 0

        @property
        def agent_ids(self) -> List[str]:
            return list(self._agents)

        @property
        def active_agents(self) -> List[str]:
            # Both agents act simultaneously, but skip terminated ones
            return [a for a in self._agents if a not in self._terminated]

        def reset(self, *, seed: Optional[int] = None) -> Dict[str, Tuple[str, Info]]:
            self._terminated = set()
            self._step = 0
            return {"A": ("obsA", {}), "B": ("obsB", {})}

        def step(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
            self._step += 1
            results = {}
            for aid in actions:
                # Agent A terminates immediately
                terminated = (aid == "A")
                if terminated:
                    self._terminated.add(aid)
                results[aid] = StepOutcome(
                    obs=f"obs_{self._step}",
                    reward=1.0 if aid == "A" else 0.5,
                    truncated=False,
                    terminated=terminated,
                    info={},
                )
            return results

    env = AsymmetricEnv()
    agent_a = MockAgent(client=MockClient(text="action_a"))
    agent_b = MockAgent(client=MockClient(text="action_b"))

    protocol = MultiAgentProtocol(agents={"A": agent_a, "B": agent_b})
    rollouts = await protocol.run(env=env, max_steps=5)

    assert len(rollouts) == 2

    r_a = next(r for r in rollouts if r.meta["agent_id"] == "A")
    r_b = next(r for r in rollouts if r.meta["agent_id"] == "B")

    # Agent A: 1 step, terminated naturally
    assert len(r_a.steps) == 1
    assert r_a.steps[0].terminated is True
    assert r_a.steps[0].truncated is False
    assert r_a.meta["episode_truncated"] is False
    assert r_a.meta["truncation_reason"] is None

    # Agent B: 5 steps, hit max_steps (truncated)
    assert len(r_b.steps) == 5
    assert r_b.steps[-1].terminated is False
    assert r_b.steps[-1].truncated is True
    assert r_b.steps[-1].info.get("truncation_reason") == "max_steps"
    assert r_b.meta["episode_truncated"] is True
    assert r_b.meta["truncation_reason"] == "max_steps"

    # Earlier steps of B should not be truncated
    for step in r_b.steps[:-1]:
        assert step.truncated is False
        assert step.terminated is False
