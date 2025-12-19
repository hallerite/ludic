from __future__ import annotations
import random
import uuid
from typing import List, Protocol
from dataclasses import replace
from ludic.training.types import RolloutRequest

class RequestStrategy(Protocol):
    """
    Interface for expanding a logical request (e.g., "Do task X") into
    concrete execution requests (e.g., "Do task X 4 times with different seeds").

    This allows decoupling the 'what' (curriculum) from the 'how' (algorithm structure).
    """
    def expand(self, requests: List[RolloutRequest]) -> List[RolloutRequest]:
        ...

class IdentityStrategy:
    """
    Default strategy: 1 Request -> 1 Execution.
    Used for standard PPO or Reinforce where you don't need grouped execution.
    """
    def expand(self, requests: List[RolloutRequest]) -> List[RolloutRequest]:
        return requests

class GRPORequestStrategy:
    """
    Expands N requests into N * G requests for Group Relative Policy Optimization.

    Invariants enforced:
    1. All G variants share the same Env seed (same prompt/problem/environment configuration).
    2. All G variants have distinct Sampling seeds (diverse answers/actions).
    3. All G variants share the same `group_id` in their meta, enabling correct
       grouping during credit assignment (even if prompts are identical across groups).

    This ensures that when we group them later for advantage estimation (Group Normalization),
    we are comparing apples to apples (same problem, different solutions).
    """
    def __init__(self, group_size: int):
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        self.group_size = group_size
        self._rng = random.Random()

    def expand(self, requests: List[RolloutRequest]) -> List[RolloutRequest]:
        expanded_requests = []

        for base_req in requests:
            # 1. Determine the single environment seed for this group.
            #    If the user provided a specific seed, we respect it (lock it for the whole group).
            #    If not, we generate one random seed and lock THAT for the whole group.
            if base_req.env_seed is not None:
                group_env_seed = base_req.env_seed
            else:
                group_env_seed = self._rng.randint(0, 2**32 - 1)

            # 2. Get the base sampling seed (if any) to ensure deterministic expansion.
            base_sampling_seed = (
                base_req.sampling_seed
                if base_req.sampling_seed is not None
                else self._rng.randint(0, 2**32 - 1)
            )

            # 3. Generate a unique group_id for this base request's expansion.
            group_id = str(uuid.uuid4())

            # 4. Create G variants for this group.
            for i in range(self.group_size):
                sampling_seed = int(base_sampling_seed + i)

                # Merge group_id into request meta
                new_meta = {**base_req.meta, "group_id": group_id}

                # Create a copy of the request, forcing the group env seed
                # and the diverse sampling seed.
                new_req = replace(
                    base_req,
                    env_seed=group_env_seed,
                    sampling_seed=sampling_seed,
                    meta=new_meta,
                    # Crucial: Each expanded request represents exactly ONE execution trace.
                    # The original 'num_episodes' on the base request is interpreted as
                    # "Number of groups to generate", so we effectively unroll that loop here if needed,
                    # but typically the curriculum provides explicit request objects.
                    # Here we assume base_req is a single intent unit.
                    num_episodes=1,
                )
                expanded_requests.append(new_req)

        return expanded_requests
