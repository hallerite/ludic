from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import torch
from ludic.types import Rollout
from ludic.training.types import RolloutStepKey


# ---- Credit Assigners ----

@dataclass
class GroupNormalizedReturn:
    """
    Computes advantage as (Episodic Return - Group-Mean Episodic Return).

    This is the core advantage estimation for GRPO.

    Contract:
    - Rollouts must have `group_id` in `rollout.meta["request_meta"]["group_id"]`.
    - Each group must have exactly `group_size` rollouts.
    - Raises ValueError if either condition is violated.

    Args:
        group_size: Number of rollouts per group.
        normalize_adv: Whether to normalize advantages to zero mean / unit std
            within each group.
        positive_only: If True, clip negative advantages to 0 so only
            rewarding trajectories receive credit (no punishments).
    """
    group_size: int
    normalize_adv: bool = False
    positive_only: bool = False

    def __post_init__(self):
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:

        out: Dict[RolloutStepKey, float] = {}

        # Group by group_id from request meta
        groups: Dict[str, List[Rollout]] = defaultdict(list)
        for r in rollouts:
            group_id = r.meta.get("request_meta", {}).get("group_id")
            if group_id is None:
                raise ValueError(
                    f"Rollout {r.id} missing group_id in meta['request_meta']. "
                    "GroupNormalizedReturn requires each rollout to have a group_id."
                )
            groups[group_id].append(r)

        for group_id, group_rollouts in groups.items():
            # Validate group size
            actual_size = len(group_rollouts)
            if actual_size != self.group_size:
                raise ValueError(
                    f"Group size mismatch for group_id={group_id}: "
                    f"expected {self.group_size}, got {actual_size}."
                )

            # 1. Get total reward (RM score) for each rollout in the group
            rewards = torch.tensor(
                [r.total_reward for r in group_rollouts],
                dtype=torch.float32
            )

            # 2. Compute the baseline (group mean)
            baseline = rewards.mean()

            # 3. Compute advantages (A_i = R_i - b)
            advantages = rewards - baseline

            if self.positive_only:
                advantages = torch.clamp(advantages, min=0.0)

            # 4. (Optional) Normalize advantages (zero mean, unit std)
            if self.normalize_adv:
                std = advantages.std(unbiased=False)
                # Add epsilon to prevent divide-by-zero if std is 0
                advantages = advantages / (std + 1e-8) 

            # 5. Assign the computed advantage to *every step*
            #    in the corresponding rollout.
            for i, r in enumerate(group_rollouts):
                adv = advantages[i].item()
                for step in r.steps:
                    key: RolloutStepKey = (r.id, step.index)
                    out[key] = adv
        
        return out


@dataclass
class MonteCarloReturn:
    """
    Monte Carlo return per step:

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    For each rollout and each step, assigns the discounted sum of *future*
    rewards including the current step.

    This is the standard REINFORCE-style return (optionally discounted).
    """

    gamma: float = 1.0

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        out: Dict[RolloutStepKey, float] = {}

        for r in rollouts:
            # process steps in reverse to accumulate returns
            G = 0.0
            returns: List[float] = []

            for step in reversed(r.steps):
                G = float(step.reward) + self.gamma * G
                returns.append(G)

            returns.reverse()  # now aligned with r.steps order

            for step, g in zip(r.steps, returns):
                key: RolloutStepKey = (r.id, step.index)
                out[key] = g

        return out


@dataclass
class PerStepReward:
    """
    Assigns each step's weight equal to its immediate reward:

        w_t = r_t

    This is sometimes useful for simple bandit-style or myopic settings.
    """

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        out: Dict[RolloutStepKey, float] = {}

        for r in rollouts:
            for step in r.steps:
                key: RolloutStepKey = (r.id, step.index)
                out[key] = float(step.reward)

        return out

@dataclass
class EpisodicReturn:
    """
    Assigns the same episodic return to every step in a rollout:

        R_ep = sum_t r_t  (undiscounted total reward for the episode)
        w_t  = R_ep       for all steps t in that rollout

    This is useful when you care only about the overall episode score and
    want each action in a successful episode to receive identical credit.
    """

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        out: Dict[RolloutStepKey, float] = {}

        for r in rollouts:
            R_ep = float(r.total_reward)
            for step in r.steps:
                key: RolloutStepKey = (r.id, step.index)
                out[key] = R_ep

        return out


@dataclass
class ConstantCredit:
    """
    Assigns a constant weight to every step in every rollout.

    This is the credit assignment for SFT / behavioral cloning:
    all actions are treated equally (weight=1.0 by default).

    Can also be used for AWR-style offline RL by setting the constant
    to exp(advantage / temperature) externally, though typically you'd
    use a more sophisticated assigner for that.

    Args:
        value: The constant weight to assign to all steps. Default 1.0.
    """

    value: float = 1.0

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        out: Dict[RolloutStepKey, float] = {}

        for r in rollouts:
            for step in r.steps:
                key: RolloutStepKey = (r.id, step.index)
                out[key] = self.value

        return out
