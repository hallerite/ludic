from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Hashable, TYPE_CHECKING

import torch
from ludic.types import Rollout
from ludic.training.types import RolloutStepKey

if TYPE_CHECKING:
    from ludic.training.value_tracker import ValueTracker

PromptKeyFn = Callable[[Rollout], Hashable]


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
    """
    group_size: int
    normalize_adv: bool = False

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


# ---- Helper for SPO prompt key extraction ----


def default_prompt_key_fn(r: Rollout) -> Hashable:
    """
    Default prompt key extraction for SPO.

    Priority:
    1. prompt_id from request_meta (if provided)
    2. First step's prev_obs (the prompt string)
    """
    request_meta = r.meta.get("request_meta", {})
    if "prompt_id" in request_meta:
        return request_meta["prompt_id"]
    if r.steps:
        return r.steps[0].prev_obs
    return r.id  # fallback to rollout id


class SPOReturn:
    """
    Single-stream Policy Optimization (SPO) credit assignment.

    Computes advantages using a persistent value tracker instead of group baselines:

        A(x, y) = R(x, y) - v̂(x)

    where v̂(x) is a Bayesian estimate of the prompt's success probability,
    maintained across batches.

    Key differences from GRPO:
    - No grouping: each rollout is independent
    - Global normalization across entire batch
    - Persistent baseline via ValueTracker
    - Naturally handles variable-time rollouts (no sync barrier)

    Reference: "Single-stream Policy Optimization" (Xu & Ding, 2025)

    Args:
        tracker: ValueTracker instance (persistent, shared across batches).
        normalize_adv: Whether to normalize advantages globally (recommended).
        prompt_key_fn: Function to extract prompt key from rollout.
            Default looks for prompt_id in request_meta, falls back to prev_obs.
        update_tracker: Whether to update tracker after computing advantages.
    """

    def __init__(
        self,
        tracker: "ValueTracker",
        normalize_adv: bool = True,
        prompt_key_fn: Optional[PromptKeyFn] = None,
        update_tracker: bool = True,
    ):
        self.tracker = tracker
        self.normalize_adv = normalize_adv
        self.prompt_key_fn = prompt_key_fn or default_prompt_key_fn
        self.update_tracker = update_tracker

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        """
        Compute SPO advantages for a batch of rollouts.

        Steps:
        1. Extract prompt key and reward for each rollout
        2. Compute raw advantage: A = R - v̂_prev (using pre-update baseline)
        3. (Optional) Update tracker with new observations
        4. (Optional) Normalize advantages globally
        5. Assign advantage to all steps in each rollout
        """
        if not rollouts:
            return {}

        out: Dict[RolloutStepKey, float] = {}

        # 1. Compute raw advantages using pre-update baselines
        raw_advantages: List[float] = []
        rollout_rewards: List[tuple] = []  # (rollout, prompt_key, reward)

        for r in rollouts:
            prompt_key = self.prompt_key_fn(r)
            reward = float(r.total_reward)
            baseline = self.tracker.get_baseline(prompt_key)
            raw_adv = reward - baseline
            raw_advantages.append(raw_adv)
            rollout_rewards.append((r, prompt_key, reward))

        # 2. Update tracker with new observations (after getting baselines)
        if self.update_tracker:
            for r, prompt_key, reward in rollout_rewards:
                # For now, use kl=0 (no KL tracking yet)
                # TODO: integrate KL computation from model if available
                self.tracker.update(prompt_key, reward, kl=0.0)

        # 3. Global normalization
        advantages = torch.tensor(raw_advantages, dtype=torch.float32)
        if self.normalize_adv and len(advantages) > 1:
            mean = advantages.mean()
            std = advantages.std(unbiased=False)
            advantages = (advantages - mean) / (std + 1e-8)

        # 4. Assign to all steps
        for i, (r, _, _) in enumerate(rollout_rewards):
            adv = advantages[i].item()
            for step in r.steps:
                key: RolloutStepKey = (r.id, step.index)
                out[key] = adv

        return out