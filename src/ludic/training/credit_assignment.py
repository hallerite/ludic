from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Protocol, Tuple

import torch
from torch import Tensor

from ludic.types import Rollout
from ludic.training.types import RolloutStepKey


Batch = Mapping[str, Tensor]


# ---------------------------------------------------------------------------
# Credit Modifiers (Level 2: Advantage Modification)
# ---------------------------------------------------------------------------
#
# CreditModifiers operate on collated batches (after credit assignment) to
# modify the per-token advantages/weights before loss computation.
#
# Key use case: OPD where KL penalty is added to advantages, causing it to
# go through importance sampling like task rewards.
#
# See docs/composition.md for the full composition level documentation.
# ---------------------------------------------------------------------------


class CreditModifier(Protocol):
    """
    Modifies batch advantages after collation, before loss computation.

    This is "Level 2: Advantage Modification" - adding per-token signals
    (like KL penalty) to trajectory-level advantages from credit assignment.

    Unlike CompositeLoss (Level 3: Loss Composition), signals added here:
    - Go through importance sampling (multiplied by ratio)
    - Use rollout-time (old policy) values, not current policy
    - Interact with task rewards through the same loss function

    Example:
        >>> modifier = KLCreditModifier(coeff=1.0)
        >>> batch, metrics = modifier.modify(batch)
        >>> # batch["weight"] now includes KL penalty
    """

    name: str

    def modify(self, batch: Batch) -> Tuple[Batch, Dict[str, Any]]:
        """
        Modify batch advantages and return metrics.

        Args:
            batch: Collated batch with at least:
                - "weight": [B, T] per-token advantages from credit assignment
                - Other fields depend on the modifier (e.g., "actor_logps", "teacher_logps")

        Returns:
            Tuple of (modified_batch, metrics_dict).
            The batch should be modified in-place for efficiency.
            Metrics are namespaced under self.name by the caller.
        """
        ...


@dataclass
class KLCreditModifier:
    """
    Add negative reverse KL to advantages for on-policy distillation.

    Implements KL penalty for OPD:
        kl_advantage_t = -coeff * (actor_logps_t - teacher_logps_t)

    The KL is computed at rollout time (old policy), so it goes through
    importance sampling when used with ratio-based losses like PPO/GSPO.

    Args:
        coeff: Coefficient for KL penalty. Higher = stronger teacher matching.
        name: Modifier name for logging. Metrics appear as "{name}/kl_mean", etc.

    Requires batch to have:
        - "actor_logps": [B, T] old policy logprobs from rollout
        - "teacher_logps": [B, T] teacher logprobs

    Example:
        >>> algo = RLAlgorithm(
        ...     credit_assigner=GroupNormalizedReturn(group_size=8),
        ...     credit_modifiers=[KLCreditModifier(coeff=1.0)],
        ...     loss=ClippedSurrogateLoss(...),
        ... )
    """

    coeff: float = 1.0
    name: str = "kl"

    def modify(self, batch: Batch) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
        if "actor_logps" not in batch:
            raise KeyError(
                "KLCreditModifier requires batch['actor_logps']. "
                "Ensure rollouts return actor logprobs (ReturnSpec.for_rl())."
            )
        if "teacher_logps" not in batch:
            raise KeyError(
                "KLCreditModifier requires batch['teacher_logps']. "
                "Ensure agent has a teacher scorer (e.g., make_vllm_teacher_scorer())."
            )

        actor_logps = batch["actor_logps"]  # [B, T]
        teacher_logps = batch["teacher_logps"]  # [B, T]
        action_mask = batch["action_mask"]  # [B, T]
        weight = batch["weight"]  # [B, T]

        # Reverse KL: log π_student - log π_teacher
        # We want to minimize this, so we add NEGATIVE KL to advantages
        reverse_kl = actor_logps - teacher_logps  # [B, T]
        kl_penalty = -self.coeff * reverse_kl  # [B, T]

        # Add to weight (advantage), masked to action tokens only
        # Note: action_mask should already be applied to weight, but we
        # apply it to kl_penalty too for safety
        modified_weight = weight + kl_penalty * action_mask.float()

        # Create modified batch (shallow copy with updated weight)
        modified_batch = dict(batch)
        modified_batch["weight"] = modified_weight

        # Compute metrics (masked to action tokens)
        mask_sum = action_mask.sum()
        if mask_sum > 0:
            masked_kl = reverse_kl * action_mask.float()
            kl_mean = masked_kl.sum() / mask_sum
            kl_std = ((masked_kl - kl_mean * action_mask.float()) ** 2).sum() / mask_sum
            kl_std = kl_std.sqrt()
        else:
            kl_mean = reverse_kl.new_zeros(())
            kl_std = reverse_kl.new_zeros(())

        metrics = {
            "kl_mean": kl_mean.detach(),
            "kl_std": kl_std.detach(),
            "kl_penalty_mean": (kl_penalty * action_mask.float()).sum().detach() / max(mask_sum, 1),
        }

        return modified_batch, metrics


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
