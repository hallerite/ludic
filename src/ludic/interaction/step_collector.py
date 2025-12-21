from __future__ import annotations
from collections import defaultdict
import uuid
from typing import Any, Dict, List, Optional

from ludic.types import Rollout, Step

class TraceCollector:
    """
    Accumulates local histories for multiple agents during a single global episode.
    
    This acts as a buffer between the raw execution of a multi-agent protocol
    and the strict, flattened `Rollout` format required by the Trainer.
    """
    
    def __init__(self, **global_meta: Any) -> None:
        """
        Args:
            **global_meta: Metadata shared across all rollouts generated 
                           from this episode (e.g. env_name, protocol_name).
        """
        self._traces: Dict[str, List[Step]] = defaultdict(list)
        self._global_meta = global_meta

    def add(self, agent_id: str, step: Step) -> None:
        """
        Record one completely processed step for a specific agent.
        
        Args:
            agent_id: The identifier for the agent (must match the ID used in
                      Rollout.meta["agent_id"]).
            step: The Step object containing ONLY that agent's view:
                  - AgentStep or EnvironmentStep in that agent's timeline.
        """
        self._traces[agent_id].append(step)

    def extract_rollouts(
        self,
        *,
        episode_truncated: bool = False,
        truncation_reason: Optional[str] = None,
    ) -> List[Rollout]:
        """
        Convert all collected traces into separate, flat Rollout objects.

        Each Rollout represents the single-agent trajectory of one participant
        in the multi-agent episode.

        Args:
            episode_truncated: Whether the episode was truncated (time limit or env).
            truncation_reason: "max_steps" | "env" | None describing the cause.

        Returns:
            A list of Rollout objects, one per agent that generated at least one step.
        """
        rollouts = []
        for agent_id, steps in self._traces.items():
            if not steps:
                continue

            # Create a clean, single-agent rollout
            # We treat each agent's trace as a distinct episode for training purposes.
            r = Rollout(
                id=str(uuid.uuid4()),
                steps=steps,
                meta={
                    **self._global_meta,
                    "agent_id": agent_id,
                    "episode_truncated": episode_truncated,
                    "truncation_reason": truncation_reason,
                }
            )
            rollouts.append(r)

        return rollouts

    def extract_rollouts_with_per_agent_status(self) -> List[Rollout]:
        """
        Convert all collected traces into Rollout objects, deriving
        truncation status from each agent's last step.

        This is used when agents finish independently (some terminate,
        some get truncated, some hit max_steps).

        Returns:
            A list of Rollout objects, one per agent that generated at least one step.
        """
        rollouts = []
        for agent_id, steps in self._traces.items():
            if not steps:
                continue

            last_env_step = next((s for s in reversed(steps) if s.kind == "env"), None)
            if last_env_step is None:
                last_step = steps[-1]
                if last_step.terminated:
                    episode_truncated = False
                    truncation_reason = None
                elif last_step.truncated:
                    episode_truncated = True
                    truncation_reason = last_step.info.get("truncation_reason", "max_steps")
                else:
                    episode_truncated = False
                    truncation_reason = None
            else:
                if last_env_step.terminated:
                    episode_truncated = False
                    truncation_reason = None
                elif last_env_step.truncated:
                    episode_truncated = True
                    truncation_reason = last_env_step.info.get("truncation_reason", "env")
                else:
                    episode_truncated = False
                    truncation_reason = None

            r = Rollout(
                id=str(uuid.uuid4()),
                steps=steps,
                meta={
                    **self._global_meta,
                    "agent_id": agent_id,
                    "episode_truncated": episode_truncated,
                    "truncation_reason": truncation_reason,
                },
            )
            rollouts.append(r)

        return rollouts
