from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, Optional

import pytest

from ludic.agents.base_agent import Agent
from ludic.inference.client import ChatResponse
from ludic.interaction.base import InteractionProtocol
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.context.full_dialog import FullDialog
from ludic.envs.env import LudicEnv
from ludic.inference.request import TokenCompletionRequest, InferenceSpec, ReturnSpec
from ludic.inference.sampling import SamplingParams

from ludic.training.batching import (
    RolloutEngine,
    RolloutBatchSource,
)
from ludic.training.types import (
    EnvSpec,
    ProtocolSpec,
    RolloutRequest,
)
from ludic.training.credit_assignment import MonteCarloReturn
from ludic.training.filters import drop_truncated
from ludic.types import Rollout, Step

from tests._mocks import MockClient, _mock_parser, MockAgent, MockChatTemplate

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

# ---------------------------------------------------------------------
# Helper / local mocks
# ---------------------------------------------------------------------

class TokenClient(MockClient):
    """
    Mock client that returns both prompt and completion token IDs so that
    RolloutEngine.generate_batch can consume them without retokenization.
    """

    async def complete_tokens(
        self,
        request: TokenCompletionRequest,
        **kwargs,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        # Prompt is "some prompt", completion is "1".
        # Tokens are arbitrary but deterministic.
        resp = ChatResponse(
            text="1",
            prompt_token_ids=[10, 11],
            completion_token_ids=[12, 13, 14],
        )
        return resp, {"mode": "token_in"}


class ConstantCreditAssigner:
    """
    Simple CreditAssigner that gives a constant weight to every step,
    just to exercise the plumbing in generate_batch.
    """

    def __init__(self, value: float = 1.23) -> None:
        self.value = value

    def compute(self, rollouts: List[Rollout]) -> Dict[Tuple[str, int], float]:
        out: Dict[Tuple[str, int], float] = {}
        for r in rollouts:
            for s in r.steps:
                out[(r.id, s.index)] = self.value
        return out


DEFAULT_INFERENCE = InferenceSpec(
    sampling=SamplingParams(temperature=0.0, max_tokens=16),
    return_=ReturnSpec.for_eval(return_token_ids=True),
)

# ---------------------------------------------------------------------
# Mock Protocol that produces MULTIPLE rollouts per run()
# ---------------------------------------------------------------------

class MultiTraceMockProtocol(InteractionProtocol):
    """A mock protocol that simulates 2 agents generating traces in 1 episode."""
    async def run(
        self,
        *,
        env: LudicEnv,
        max_steps: int,
        env_seed: Optional[int] = None,
        sampling_seed: Optional[int] = None,
        inference: Optional[InferenceSpec] = None,
        timeout_s: Optional[float] = None,
    ) -> List[Rollout]:
        # Simulate Agent A
        r1 = Rollout(steps=[Step(0, "obsA", "actA", "nextA", 1.0, False, True)])
        # Simulate Agent B
        r2 = Rollout(steps=[Step(0, "obsB", "actB", "nextB", -0.5, False, True)])
        return [r1, r2]

# ---------------------------------------------------------------------
# generate_rollouts: env/ctx registry + metadata + concurrency
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_rollouts_basic_metadata_and_termination(
    env_registry,
    mock_agent,
) -> None:
    protocol_registry: ProtocolRegistry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=mock_agent)
    }
    
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={"max_steps": 3, "target": "1"}),
        protocol=ProtocolSpec(kind="mock_protocol"),
        num_episodes=2,
        meta={"tag": "test"},
    )

    # Calling with single-agent protocol, should get 2 rollouts (1 per episode)
    rollouts = await engine.generate_rollouts(
        requests=[request],
        max_steps=5,
        timeout_s=None,
        concurrency=4,
    )

    assert len(rollouts) == 2

    seen_episode_idx = set()

    for r in rollouts:
        # should terminate successfully on correct action "1"
        assert r.steps[-1].terminated is True
        assert r.total_reward == pytest.approx(1.0)

        # engine metadata
        assert r.meta["engine"]["max_steps"] == 5
        assert r.meta["engine"]["timeout_s"] is None
        assert r.meta["engine"]["env_kind"] == "mock"
        assert r.meta["engine"]["protocol_kind"] == "mock_protocol"

        # request-level metadata propagated
        assert r.meta["request_meta"]["tag"] == "test"

        # unique episode indices
        episode_idx = r.meta["episode_idx"]
        assert isinstance(episode_idx, int)
        seen_episode_idx.add(episode_idx)

    assert seen_episode_idx == {0, 1}


@pytest.mark.asyncio
async def test_generate_rollouts_flattens_multi_trace_protocols(
    env_registry,
) -> None:
    """
    Verifies that if a protocol returns [RolloutA, RolloutB] for one episode,
    the engine correctly flattens them into the result list.
    """
    protocol_registry = {
        "multi_trace": lambda: MultiTraceMockProtocol()
    }
    
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={}),
        protocol=ProtocolSpec(kind="multi_trace"),
        num_episodes=2, # 2 Global Episodes
    )

    # 2 Episodes * 2 Traces per Episode = 4 Total Rollouts
    rollouts = await engine.generate_rollouts(
        requests=[request],
        max_steps=5,
    )

    assert len(rollouts) == 4
    
    # Check that we have 2 from episode 0 and 2 from episode 1
    ep_indices = [r.meta["episode_idx"] for r in rollouts]
    assert ep_indices.count(0) == 2
    assert ep_indices.count(1) == 2


@pytest.mark.asyncio
async def test_generate_rollouts_unknown_env_raises(
    env_registry,
    mock_agent,
) -> None:
    protocol_registry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=mock_agent)
    }
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    request = RolloutRequest(
        env=EnvSpec(kind="does_not_exist", kwargs={}),
        protocol=ProtocolSpec(kind="mock_protocol"),
        num_episodes=1,
    )

    with pytest.raises(KeyError, match="Unknown env kind"):
        await engine.generate_rollouts(
            requests=[request],
            max_steps=3,
        )


@pytest.mark.asyncio
async def test_generate_rollouts_unknown_protocol_raises(
    env_registry,
) -> None:
    """Verifies that an unknown protocol kind raises a KeyError."""
    protocol_registry = {
        # No protocols registered
    }
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={}),
        protocol=ProtocolSpec(kind="does_not_exist"),
        num_episodes=1,
    )

    with pytest.raises(KeyError, match="Unknown protocol kind"):
        await engine.generate_rollouts(
            requests=[request],
            max_steps=3,
        )


@pytest.mark.asyncio
async def test_generate_rollouts_heterogeneous_protocols(
    env_registry,
) -> None:
    """
    Tests that the RolloutEngine can run a batch containing
    requests for two different protocols.
    """
    # Define two different agent/protocol setups
    agent_A = MockAgent(client=MockClient(text="Agent A says hi"))
    protocol_A = SingleAgentSyncProtocol(agent=agent_A)

    agent_B = MockAgent(client=MockClient(text="Agent B says hi"))
    protocol_B = SingleAgentSyncProtocol(agent=agent_B)

    protocol_registry = {
        "protocol_A": lambda: protocol_A,
        "protocol_B": lambda: protocol_B,
    }
    
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    # Request 1 for Protocol A
    req_A = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={}),
        protocol=ProtocolSpec(kind="protocol_A"),
        num_episodes=1,
    )
    
    # Request 2 for Protocol B
    req_B = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={}),
        protocol=ProtocolSpec(kind="protocol_B"),
        num_episodes=1,
    )

    rollouts = await engine.generate_rollouts(
        requests=[req_A, req_B],
        max_steps=2,
        concurrency=2,
    )

    assert len(rollouts) == 2

    # Check that the rollouts were generated by the correct protocols
    protocol_kinds = sorted([r.meta["engine"]["protocol_kind"] for r in rollouts])
    assert protocol_kinds == ["protocol_A", "protocol_B"]

    # Check that the correct agent was used for each
    for r in rollouts:
        if r.meta["engine"]["protocol_kind"] == "protocol_A":
            assert r.steps[0].action == "Agent A says hi"
        elif r.meta["engine"]["protocol_kind"] == "protocol_B":
            assert r.steps[0].action == "Agent B says hi"


@pytest.mark.asyncio
async def test_generate_rollouts_writes_jsonl(
    tmp_path,
    env_registry,
    mock_agent,
) -> None:
    jsonl_path = tmp_path / "rollouts.jsonl"
    
    protocol_registry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=mock_agent)
    }

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=str(jsonl_path),
    )

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={"max_steps": 2, "target": "1"}),
        protocol=ProtocolSpec(kind="mock_protocol"),
        num_episodes=2,
        meta={"foo": "bar"},
    )

    rollouts = await engine.generate_rollouts(
        requests=[request],
        max_steps=4,
    )

    assert len(rollouts) == 2
    assert jsonl_path.exists()

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    for line in lines:
        payload = json.loads(line)
        assert "id" in payload
        assert "meta" in payload
        assert "steps" in payload
        assert payload["meta"]["request_meta"]["foo"] == "bar"
        assert payload["meta"]["engine"]["env_kind"] == "mock"
        assert payload["meta"]["engine"]["protocol_kind"] == "mock_protocol"
        assert payload["length"] == len(payload["steps"])


# ---------------------------------------------------------------------
# generate_batch: with model token IDs (no retokenize)
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_batch_uses_model_token_ids_when_available(
    env_registry,
) -> None:
    # Agent whose client returns prompt_token_ids + token_ids
    agent = Agent(
        client=TokenClient(),
        model="mock",
        ctx=FullDialog(),
        parser=_mock_parser,
        chat_template=MockChatTemplate(),
    )
    
    protocol_registry = {
        "token_protocol": lambda: SingleAgentSyncProtocol(agent=agent)
    }

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    credit_assigner = ConstantCreditAssigner(value=1.23)

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={"max_steps": 1, "target": "1"}),
        protocol=ProtocolSpec(kind="token_protocol"),
        num_episodes=1,
    )

    batch = await engine.generate_batch(
        requests=[request],
        max_steps=3,
        credit_assigner=credit_assigner,
        timeout_s=None,
        concurrency=1,
    )

    # Single rollout, single step
    assert batch.meta["target_rollouts"] == 1
    assert batch.meta["num_samples"] == 1
    # Env gives +1 reward when guess is correct
    assert batch.meta["avg_total_reward"] == pytest.approx(1.0)

    assert len(batch.items) == 1
    item = batch.items[0]

    # IDs must be prompt_ids + completion_ids
    assert item.input_ids == [10, 11, 12, 13, 14]
    assert item.attention_mask == [1, 1, 1, 1, 1]
    assert item.action_mask == [0, 0, 1, 1, 1]
    assert item.weight == pytest.approx(1.23)

    # meta carries rollout/step info
    assert "rollout_id" in item.meta
    assert "step_index" in item.meta
    assert item.meta["total_reward"] == pytest.approx(1.0)


# ---------------------------------------------------------------------
# generate_batch: missing token IDs
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# RolloutBatchSource integration: delegates to RolloutEngine.generate_batch
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rollout_batch_source_next_batch_integration(
    env_registry,
) -> None:
    agent = Agent(
        client=TokenClient(),
        model="mock",
        ctx=FullDialog(),
        parser=_mock_parser,
        chat_template=MockChatTemplate(),
    )
    protocol_registry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=agent)
    }
    
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    credit_assigner = MonteCarloReturn(gamma=0.9)

    def requests_fn() -> List[RolloutRequest]:
        # Two episodes per call
        return [
            RolloutRequest(
                env=EnvSpec(kind="mock", kwargs={"max_steps": 2, "target": "1"}),
                protocol=ProtocolSpec(kind="mock_protocol"),
                num_episodes=2,
                inference=DEFAULT_INFERENCE,
                meta={"batch_source": True},
            )
        ]

    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=credit_assigner,
        requests_fn=requests_fn,
        max_steps=3,
        timeout_s=None,
        concurrency=2,
    )

    saw_batch = await batch_source.next_batch()

    # We asked for num_episodes=2 => 2 rollouts; each should have at least 1 step
    assert saw_batch.meta["target_rollouts"] == 2
    assert saw_batch.meta["num_samples"] >= 2

    # All items should carry rollout metadata including request_meta
    assert len(saw_batch.items) == saw_batch.meta["num_samples"]
    for item in saw_batch.items:
        assert item.meta.get("request_meta", {}).get("batch_source") is True
        assert "rollout_id" in item.meta
        assert "step_index" in item.meta


@pytest.mark.asyncio
async def test_rollout_batch_source_passes_sample_filter(
    env_registry,
) -> None:
    agent = Agent(
        client=TokenClient(),
        model="mock",
        ctx=FullDialog(),
        parser=_mock_parser,
        chat_template=MockChatTemplate(),
    )
    protocol_registry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=agent)
    }

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    credit_assigner = MonteCarloReturn()

    def requests_fn() -> List[RolloutRequest]:
        return [
            RolloutRequest(
                env=EnvSpec(kind="mock", kwargs={"max_steps": 10, "target": "win"}),
                protocol=ProtocolSpec(kind="mock_protocol"),
                num_episodes=1,
                inference=DEFAULT_INFERENCE,
            )
        ]

    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=credit_assigner,
        requests_fn=requests_fn,
        max_steps=3,
        sample_filter=drop_truncated,
    )

    batch = await batch_source.next_batch()

    assert batch.meta["num_samples_before_filter"] == 3
    assert batch.meta["num_samples"] == 2
    assert all(item.meta.get("truncated") is False for item in batch.items)


@pytest.mark.asyncio
async def test_saw_item_contains_truncation_flags(
    env_registry,
) -> None:
    """
    SAWItem.meta should contain 'truncated' and 'terminated' flags
    propagated from the Step, as well as episode-level truncation info
    from Rollout.meta.
    """

    agent = Agent(
        client=TokenClient(),
        model="mock",
        ctx=FullDialog(),
        parser=_mock_parser,
        chat_template=MockChatTemplate(),
    )  # Never terminates the env since it never outputs target="win"
    protocol_registry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=agent),
    }

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    credit_assigner = MonteCarloReturn()

    requests = [
        RolloutRequest(
            env=EnvSpec(kind="mock", kwargs={"max_steps": 10, "target": "win"}),
            protocol=ProtocolSpec(kind="mock_protocol"),
            num_episodes=1,
            inference=DEFAULT_INFERENCE,
        )
    ]

    # max_steps=3 means we'll truncate at step 3
    batch = await engine.generate_batch(
        requests=requests,
        max_steps=3,
        credit_assigner=credit_assigner,
    )

    assert len(batch.items) == 3

    # First two steps: not truncated, not terminated
    for i in range(2):
        item = batch.items[i]
        assert item.meta.get("truncated") is False
        assert item.meta.get("terminated") is False

    # Last step: truncated due to max_steps
    last_item = batch.items[-1]
    assert last_item.meta.get("truncated") is True
    assert last_item.meta.get("terminated") is False
    assert last_item.meta.get("truncation_reason") == "max_steps"

    # Episode-level truncation info
    assert last_item.meta.get("episode_truncated") is True


@pytest.mark.asyncio
async def test_generate_batch_applies_sample_filter_and_updates_counts(
    env_registry,
) -> None:
    agent = Agent(
        client=TokenClient(),
        model="mock",
        ctx=FullDialog(),
        parser=_mock_parser,
        chat_template=MockChatTemplate(),
    )  # Never terminates the env since it never outputs target="win"
    protocol_registry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=agent),
    }

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    credit_assigner = MonteCarloReturn()

    requests = [
        RolloutRequest(
            env=EnvSpec(kind="mock", kwargs={"max_steps": 10, "target": "win"}),
            protocol=ProtocolSpec(kind="mock_protocol"),
            num_episodes=1,
            inference=DEFAULT_INFERENCE,
        )
    ]

    batch = await engine.generate_batch(
        requests=requests,
        max_steps=3,
        credit_assigner=credit_assigner,
        sample_filter=drop_truncated,
    )

    assert batch.meta["num_samples_before_filter"] == 3
    assert batch.meta["num_samples"] == 2
    assert batch.meta["num_samples_filtered"] == 1
    assert len(batch.items) == 2
    assert all(item.meta.get("truncated") is False for item in batch.items)


@pytest.mark.asyncio
async def test_avg_completion_length_respects_filtered_items(
    env_registry,
) -> None:
    class SeqClient(MockClient):
        def __init__(self, texts: List[str]) -> None:
            super().__init__(text=texts[0])
            self._texts = list(texts)
            self._i = 0

        async def complete_tokens(  # type: ignore[override]
            self,
            request: TokenCompletionRequest,
            **kwargs,
        ) -> Tuple[ChatResponse, Dict[str, Any]]:
            text = self._texts[min(self._i, len(self._texts) - 1)]
            self._i += 1
            completion_ids = list(range(100, 100 + len(text)))
            return (
                ChatResponse(
                    text=text,
                    prompt_token_ids=[1, 2],
                    completion_token_ids=completion_ids,
                ),
                {"mode": "token_in"},
            )

    agent = Agent(
        client=SeqClient(["a", "bb", "ccc"]),
        model="mock",
        ctx=FullDialog(),
        parser=_mock_parser,
        chat_template=MockChatTemplate(),
    )
    protocol_registry = {
        "mock_protocol": lambda: SingleAgentSyncProtocol(agent=agent),
    }

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    credit_assigner = MonteCarloReturn()
    requests = [
        RolloutRequest(
            env=EnvSpec(kind="mock", kwargs={"max_steps": 10, "target": "win"}),
            protocol=ProtocolSpec(kind="mock_protocol"),
            num_episodes=1,
            inference=DEFAULT_INFERENCE,
        )
    ]

    # Keep only the first step so avg_completion_length should match len("a") == 1.
    batch = await engine.generate_batch(
        requests=requests,
        max_steps=3,
        credit_assigner=credit_assigner,
        sample_filter=lambda item: item.meta.get("step_index") == 0,
    )

    assert batch.meta["num_samples"] == 1
    assert batch.meta["avg_completion_length"] == pytest.approx(1.0)
