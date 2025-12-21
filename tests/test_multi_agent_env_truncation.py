from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pytest

from ludic.envs.env import LudicEnv
from ludic.interaction.multi_agent import MultiAgentProtocol
from ludic.types import Info, StepOutcome, EnvironmentStep, Rollout
from tests._mocks import MockAgent, MockClient


class EnvTruncatesBoth(LudicEnv[str, str, str]):
    def __init__(self) -> None:
        self._agents = ["A", "B"]
        self._t = 0

    @property
    def agent_ids(self) -> List[str]:
        return list(self._agents)

    @property
    def active_agents(self) -> List[str]:
        return list(self._agents)

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, Tuple[str, Info]]:
        self._t = 0
        return {"A": ("obsA", {}), "B": ("obsB", {})}

    def step(self, actions: Dict[str, str]) -> Dict[str, StepOutcome]:
        self._t += 1
        truncated = self._t >= 2
        return {
            aid: StepOutcome(
                obs=f"obs_{self._t}",
                reward=0.0,
                truncated=truncated,
                terminated=False,
                info={},
            )
            for aid in actions
        }


def _env_steps(rollout: Rollout) -> List[EnvironmentStep]:
    return [s for s in rollout.steps if isinstance(s, EnvironmentStep)]


@pytest.mark.asyncio
async def test_multi_agent_env_truncation_is_recorded_as_env() -> None:
    env = EnvTruncatesBoth()
    protocol = MultiAgentProtocol(
        agents={
            "A": MockAgent(client=MockClient(text="a")),
            "B": MockAgent(client=MockClient(text="b")),
        }
    )

    rollouts = await protocol.run(env=env, max_steps=10)
    assert len(rollouts) == 2

    for r in rollouts:
        env_steps = _env_steps(r)
        assert len(env_steps) == 2
        assert env_steps[-1].truncated is True
        assert r.meta.get("episode_truncated") is True
        assert r.meta.get("truncation_reason") == "env"
        assert env_steps[-1].info.get("truncation_reason") is None
