from __future__ import annotations

from collections import defaultdict
from typing import Optional, Tuple, List, Dict, Any

import pytest

from ludic.envs.single_agent_env import SingleAgentEnv
from ludic.types import Observation, Info, StepOutcome
from ludic.inference import InferenceSpec, SamplingParams, ReturnSpec
from ludic.training.types import (
    RolloutRequest,
    ProtocolSpec,
    EnvSpec,
)
from ludic.training.batching import (
    RolloutEngine,
    RolloutBatchSource,
    GRPORequestStrategy,
)
from ludic.training.credit_assignment import (
    GroupNormalizedReturn,
)
from ludic.interaction.base import InteractionProtocol
from ludic.interaction.single_agent import SingleAgentSyncProtocol

from tests._mocks import SeedableMockAgent

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# 1. A mock environment that is deterministic based on seed
# ---------------------------------------------------------------------------
class SeedableMockEnv(SingleAgentEnv):
    """
    A mock env that is deterministic based on the reset seed.
    - reset() obs includes the seed.
    - step() gives +1 for a "correct" action, -0.1 otherwise.
    """

    def __init__(self, correct_action: str = "A") -> None:
        super().__init__()
        self.correct_action = correct_action
        self._obs: Observation = "Not reset"
        self._t = 0

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return f"You must reply with only one letter. The correct letter is {self.correct_action}."

    def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:
        self._t = 0
        self._obs = f"Start state for seed {seed}. Correct action is {self.correct_action}."
        return self._obs, {"seed": seed}

    def env_step(self, action: str) -> StepOutcome:
        self._t += 1
        action_clean = action.strip().upper()

        if action_clean == self.correct_action:
            reward = 1.0
            terminated = True
            self._obs = "✅ Correct"
        else:
            reward = -0.1
            terminated = True  # End episode on first action for simplicity
            self._obs = f"❌ Incorrect (got {action_clean})"

        return StepOutcome(
            obs=self._obs,
            reward=reward,
            truncated=False,
            terminated=terminated,
            info={"action_taken": action_clean},
        )

    def env_current_obs(self) -> Observation:
        return self._obs


# ---------------------------------------------------------------------------
# 2. The End-to-End Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grpo_e2e_seed_grouping_and_credit() -> None:
    """
    End-to-end grouped rollout + credit assignment:
    - GRPORequestStrategy expands requests into groups.
    - Env seed controls reset obs; sampling seed controls agent action.
    - GroupNormalizedReturn assigns normalized group advantages.
    """
    group_size = 2
    seed_groups = (100, 200)
    base_seeds = (9000, 8000)

    env_registry = {"env": lambda **kwargs: SeedableMockEnv(correct_action="A")}

    seed_to_action_map = {}
    for base_seed in base_seeds:
        seed_to_action_map[base_seed] = "A"
        seed_to_action_map[base_seed + 1] = "Z"

    credit_assigner = GroupNormalizedReturn(group_size=group_size, normalize_adv=True)

    def create_protocol() -> InteractionProtocol:
        agent = SeedableMockAgent(seed_map=seed_to_action_map)
        return SingleAgentSyncProtocol(agent=agent)

    protocol_registry = {"grpo_protocol": create_protocol}
    engine = RolloutEngine(protocol_registry=protocol_registry, env_registry=env_registry)

    def make_expanded_requests() -> List[RolloutRequest]:
        inference = InferenceSpec(
            sampling=SamplingParams(temperature=0.7, max_tokens=5),
            return_=ReturnSpec.for_eval(return_token_ids=True),
        )
        base_requests = [
            RolloutRequest(
                env=EnvSpec(kind="env", kwargs={}),
                protocol=ProtocolSpec(kind="grpo_protocol", kwargs={}),
                env_seed=seed,
                sampling_seed=base_seed,
                inference=inference,
            )
            for seed, base_seed in zip(seed_groups, base_seeds)
        ]
        strategy = GRPORequestStrategy(group_size=group_size)
        return strategy.expand(base_requests)

    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=credit_assigner,
        requests_fn=make_expanded_requests,
        max_steps=3,
    )

    saw_batch = await batch_source.next_batch()

    assert saw_batch.meta["target_rollouts"] == len(seed_groups) * group_size
    rollouts: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in saw_batch.items:
        meta = item.meta
        rollouts[meta["engine"]["env_seed"]].append(
            {
                "sampling_seed": meta["engine"]["sampling_seed"],
                "prev_obs": meta["prev_obs"],
                "reward": meta["reward"],
                "weight": item.weight,
            }
        )

    assert set(rollouts) == set(seed_groups)
    for seed, base_seed in zip(seed_groups, base_seeds):
        group = rollouts[seed]
        assert len(group) == group_size
        assert {r["sampling_seed"] for r in group} == {base_seed, base_seed + 1}
        obs = {r["prev_obs"] for r in group}
        assert len(obs) == 1
        assert f"seed {seed}" in next(iter(obs))
        rewards = sorted(r["reward"] for r in group)
        assert rewards == [-0.1, 1.0]
        weights = sorted(round(r["weight"], 4) for r in group)
        assert weights == [-1.0, 1.0]
