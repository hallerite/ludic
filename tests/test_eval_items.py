from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import pytest

from ludic.eval.core import run_eval
from ludic.training.stats import Reducer
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.types import Rollout, EnvironmentStep, TokenTrace


@dataclass
class _FakeEngine:
    rollout_batches: List[List[Rollout]]

    async def generate_rollouts(
        self,
        *,
        requests: Sequence[RolloutRequest],
        max_steps: int,
        timeout_s: float | None,
        concurrency: int,
    ) -> List[Rollout]:
        return self.rollout_batches.pop(0)


def _mk_rollout(n_steps: int, *, completion_lens: List[int]) -> Rollout:
    assert len(completion_lens) == n_steps
    steps: List[EnvironmentStep] = []
    for i in range(n_steps):
        completion_token_ids = [0] * int(completion_lens[i])
        steps.append(
            EnvironmentStep(
                index=i,
                prev_obs=f"obs{i}",
                action=f"a{i}",
                parsed_action=f"a{i}",
                next_obs=f"obs{i+1}",
                source_agent_step_id=f"agent_{i}",
                agent_step_ids=[f"agent_{i}"],
                reward=1.0,
                truncated=False,
                terminated=(i == n_steps - 1),
                info={"is_last": i == n_steps - 1},
                trace=TokenTrace(
                    prompt_token_ids=[0],
                    completion_token_ids=completion_token_ids,
                ),
            )
        )
    return Rollout(steps=steps, meta={})


@pytest.mark.asyncio
async def test_eval_defaults_to_step_items_and_normalizes_like_training() -> None:
    reqs = [
        RolloutRequest(env=EnvSpec(kind="x"), protocol=ProtocolSpec(kind="y"), num_episodes=2),
        RolloutRequest(env=EnvSpec(kind="x"), protocol=ProtocolSpec(kind="y"), num_episodes=1),
    ]

    # One call returns all rollouts.
    engine = _FakeEngine(
        rollout_batches=[
            [
                _mk_rollout(2, completion_lens=[3, 5]),
                _mk_rollout(1, completion_lens=[7]),
                _mk_rollout(3, completion_lens=[2, 4, 6]),
            ]
        ]
    )

    reducers = {
        # "samples" should mean step-items, like SAWItems in training.
        "avg_completion_length": Reducer(kind="mean", source="completion_length"),
        # Episode-level rate: only count terminal-step values, normalize by rollouts.
        "terminal_seen_rate": Reducer(kind="count_true", source="is_last", normalize_by="rollouts"),
    }

    records, metrics = await run_eval(
        engine=engine,  # type: ignore[arg-type]
        requests=reqs,
        reducers=reducers,
        max_steps=5,
        timeout_s=None,
        concurrency=2,
    )

    # Records are step-items: 2 + 1 + 3 = 6 steps total.
    assert len(records) == 6

    # Mean completion length over step-items:
    # (3+5+7+2+4+6) / 6 = 27/6 = 4.5
    assert metrics["avg_completion_length"] == pytest.approx(4.5)

    # Exactly one terminal-step per rollout => 3 / 3
    assert metrics["terminal_seen_rate"] == pytest.approx(1.0)
