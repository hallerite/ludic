"""
SPO Value Tracker: Bayesian per-prompt value estimation.

This module implements the KL-adaptive value tracker from Single-stream Policy
Optimization (SPO). The tracker maintains a persistent estimate of each prompt's
success probability, providing stable baselines for advantage computation without
requiring grouped samples.

Reference: "Single-stream Policy Optimization" (Xu & Ding, 2025)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Hashable, List, Tuple

PromptKey = Hashable  # str, int, tuple, etc.


@dataclass
class PromptState:
    """State for a single prompt in the tracker."""

    alpha: float  # Beta distribution parameter (successes)
    beta: float  # Beta distribution parameter (failures)
    last_kl: float = 0.0  # KL divergence at last update (for adaptive forgetting)

    @property
    def value(self) -> float:
        """Posterior mean: E[v] = α / (α + β)"""
        return self.alpha / (self.alpha + self.beta)

    @property
    def effective_n(self) -> float:
        """Effective sample count."""
        return self.alpha + self.beta

    @property
    def uncertainty(self) -> float:
        """Standard deviation of Bernoulli with p=value: √(v(1-v))"""
        v = self.value
        return math.sqrt(v * (1.0 - v))


@dataclass
class ValueTracker:
    """
    KL-adaptive Bayesian value tracker for SPO.

    Maintains per-prompt estimates of success probability using Beta-Bernoulli
    conjugate updates with KL-adaptive forgetting.

    The tracker:
    - Provides stable baselines for advantage computation
    - Adapts forgetting rate based on policy drift (KL divergence)
    - Supports prioritized sampling via uncertainty-based weights

    Args:
        d_half: KL divergence that causes 50% forgetting (controls adaptation speed).
        rho_min: Minimum forgetting factor (maximum forgetting).
        rho_max: Maximum forgetting factor (minimum forgetting).
        prior_strength: Strength of the uniform prior (α=β=prior_strength).
        exploration_bonus: Minimum sampling weight to prevent curriculum collapse.
    """

    d_half: float = 0.1
    rho_min: float = 0.875
    rho_max: float = 0.96
    prior_strength: float = 1.0
    exploration_bonus: float = 0.05

    _prompts: Dict[PromptKey, PromptState] = field(default_factory=dict)

    def __post_init__(self):
        # Initial effective sample size at equilibrium
        self._n0 = 1.0 / (1.0 - self.rho_min)

    def _compute_rho(self, kl: float) -> float:
        """Compute forgetting factor from KL divergence."""
        rho = 2.0 ** (-kl / self.d_half)
        return max(self.rho_min, min(self.rho_max, rho))

    def get_baseline(self, key: PromptKey) -> float:
        """
        Get current value estimate for a prompt.

        Returns the prior mean (0.5) if prompt hasn't been seen.
        """
        if key not in self._prompts:
            return 0.5  # Uniform prior mean
        return self._prompts[key].value

    def get_state(self, key: PromptKey) -> Optional[PromptState]:
        """Get full state for a prompt, or None if unseen."""
        return self._prompts.get(key)

    def update(
        self,
        key: PromptKey,
        reward: float,
        kl: float = 0.0,
    ) -> float:
        """
        Update tracker with a new observation.

        Args:
            key: Prompt identifier.
            reward: Observed reward (0 or 1 for binary, general float supported).
            kl: KL divergence since last update on this prompt.

        Returns:
            The new value estimate after update.
        """
        rho = self._compute_rho(kl)

        if key not in self._prompts:
            # Initialize with prior
            self._prompts[key] = PromptState(
                alpha=self.prior_strength,
                beta=self.prior_strength,
                last_kl=kl,
            )

        state = self._prompts[key]

        # Discount prior and add new evidence
        # For binary rewards: α += r, β += (1-r)
        # This generalizes to continuous rewards in [0,1] as soft counts
        state.alpha = rho * state.alpha + reward
        state.beta = rho * state.beta + (1.0 - reward)
        state.last_kl = kl

        return state.value

    def initialize_from_samples(
        self,
        key: PromptKey,
        rewards: List[float],
    ) -> float:
        """
        Initialize a prompt's value estimate from multiple samples.

        Uses the paper's initialization: set effective sample size to equilibrium
        value N0 = 1/(1-ρ_min), then α = N0 * v̂, β = N0 * (1-v̂).

        Args:
            key: Prompt identifier.
            rewards: List of observed rewards.

        Returns:
            The initialized value estimate.
        """
        if not rewards:
            return 0.5

        v0 = sum(rewards) / len(rewards)
        self._prompts[key] = PromptState(
            alpha=self._n0 * v0,
            beta=self._n0 * (1.0 - v0),
            last_kl=0.0,
        )
        return v0

    def get_sampling_weight(self, key: PromptKey) -> float:
        """
        Get prioritized sampling weight for a prompt.

        Weight is proportional to uncertainty: √(v(1-v)) + ε.
        Prompts with v ≈ 0.5 get highest weight (most uncertain).
        """
        if key not in self._prompts:
            # Unseen prompts get high weight (uncertainty = 0.5)
            return 0.5 + self.exploration_bonus
        return self._prompts[key].uncertainty + self.exploration_bonus

    def get_all_weights(self) -> Dict[PromptKey, float]:
        """Get sampling weights for all tracked prompts."""
        return {k: self.get_sampling_weight(k) for k in self._prompts}

    def known_prompts(self) -> List[PromptKey]:
        """Return list of all tracked prompt keys."""
        return list(self._prompts.keys())

    def __len__(self) -> int:
        return len(self._prompts)

    def __contains__(self, key: PromptKey) -> bool:
        return key in self._prompts
