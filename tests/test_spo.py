"""Tests for Single-stream Policy Optimization (SPO) components."""

import pytest
import math

from ludic.types import Rollout, Step
from ludic.training.value_tracker import ValueTracker, PromptState
from ludic.training.credit_assignment import SPOReturn, default_prompt_key_fn
from ludic.training.algorithm import make_spo


# ---- ValueTracker Tests ----


def test_value_tracker_initial_baseline():
    """Unseen prompts return prior mean (0.5)."""
    tracker = ValueTracker()
    assert tracker.get_baseline("unseen") == 0.5


def test_value_tracker_update_success():
    """Updating with success increases value estimate."""
    tracker = ValueTracker()

    # First update with success
    v1 = tracker.update("prompt_A", reward=1.0)
    assert v1 > 0.5  # Should increase from prior

    # Second update with success
    v2 = tracker.update("prompt_A", reward=1.0)
    assert v2 > v1  # Should increase further


def test_value_tracker_update_failure():
    """Updating with failure decreases value estimate."""
    tracker = ValueTracker()

    # First update with failure
    v1 = tracker.update("prompt_A", reward=0.0)
    assert v1 < 0.5  # Should decrease from prior

    # Second update with failure
    v2 = tracker.update("prompt_A", reward=0.0)
    assert v2 < v1  # Should decrease further


def test_value_tracker_converges_to_mean():
    """Tracker converges to empirical mean over many samples."""
    tracker = ValueTracker(rho_min=0.9, rho_max=0.99)

    # Simulate 70% success rate
    for _ in range(50):
        tracker.update("prompt_A", reward=1.0)
    for _ in range(21):  # ~30%
        tracker.update("prompt_A", reward=0.0)

    v = tracker.get_baseline("prompt_A")
    assert 0.6 < v < 0.8  # Should be roughly 0.7


def test_value_tracker_initialize_from_samples():
    """Initialize from multiple samples sets correct value."""
    tracker = ValueTracker()

    rewards = [1.0, 1.0, 0.0, 1.0]  # 75% success
    v = tracker.initialize_from_samples("prompt_A", rewards)

    assert v == 0.75
    assert tracker.get_baseline("prompt_A") == 0.75


def test_value_tracker_sampling_weight():
    """Uncertain prompts get higher sampling weight."""
    tracker = ValueTracker(exploration_bonus=0.0)

    # Initialize prompts with different success rates
    tracker.initialize_from_samples("easy", [1.0] * 10)  # v ≈ 1.0
    tracker.initialize_from_samples("hard", [0.0] * 10)  # v ≈ 0.0
    tracker.initialize_from_samples("medium", [1.0] * 5 + [0.0] * 5)  # v ≈ 0.5

    w_easy = tracker.get_sampling_weight("easy")
    w_hard = tracker.get_sampling_weight("hard")
    w_medium = tracker.get_sampling_weight("medium")

    # Medium (uncertain) should have highest weight
    assert w_medium > w_easy
    assert w_medium > w_hard


def test_value_tracker_kl_adaptive_forgetting():
    """Higher KL causes faster forgetting (lower rho)."""
    tracker = ValueTracker(d_half=0.1, rho_min=0.5, rho_max=0.99)

    # Initialize with high confidence
    tracker.initialize_from_samples("prompt_A", [1.0] * 20)
    v_before = tracker.get_baseline("prompt_A")
    assert v_before > 0.9

    # Update with failure but high KL (policy changed a lot)
    tracker.update("prompt_A", reward=0.0, kl=0.5)  # High KL
    v_after = tracker.get_baseline("prompt_A")

    # Should drop significantly due to high KL forgetting
    assert v_after < v_before


def test_value_tracker_contains_and_len():
    """Test __contains__ and __len__."""
    tracker = ValueTracker()

    assert len(tracker) == 0
    assert "prompt_A" not in tracker

    tracker.update("prompt_A", reward=1.0)

    assert len(tracker) == 1
    assert "prompt_A" in tracker


# ---- SPOReturn Tests ----


def _make_rollout(
    id: str,
    *,
    prompt: str = "prompt_default",
    rewards: list[float],
    prompt_id: str | None = None,
) -> Rollout:
    """Helper to create test rollouts."""
    request_meta = {}
    if prompt_id is not None:
        request_meta["prompt_id"] = prompt_id
    rollout = Rollout(id=id, meta={"request_meta": request_meta})
    obs = prompt

    for i, reward in enumerate(rewards):
        next_obs = f"obs_{i+1}" if i < len(rewards) - 1 else None
        rollout.steps.append(Step(
            index=i,
            prev_obs=obs,
            action=f"action_{i}",
            next_obs=next_obs,
            reward=reward,
            truncated=False,
            terminated=(i == len(rewards) - 1),
            info={},
        ))
        obs = next_obs or ""
    return rollout


def test_spo_return_basic():
    """Basic SPO advantage computation."""
    tracker = ValueTracker()
    assigner = SPOReturn(tracker=tracker, normalize_adv=False)

    # Two rollouts with different outcomes
    r1 = _make_rollout("r1", prompt="prompt_A", rewards=[1.0], prompt_id="p1")
    r2 = _make_rollout("r2", prompt="prompt_B", rewards=[0.0], prompt_id="p2")

    weights = assigner.compute([r1, r2])

    # Both use baseline of 0.5 (prior mean)
    # r1: 1.0 - 0.5 = 0.5
    # r2: 0.0 - 0.5 = -0.5
    assert weights[("r1", 0)] == pytest.approx(0.5)
    assert weights[("r2", 0)] == pytest.approx(-0.5)


def test_spo_return_uses_pre_update_baseline():
    """Advantages use pre-update baselines, then tracker is updated."""
    tracker = ValueTracker()
    assigner = SPOReturn(tracker=tracker, normalize_adv=False)

    # Initialize tracker with some value for prompt p1
    tracker.initialize_from_samples("p1", [1.0, 1.0, 1.0, 0.0])  # v = 0.75

    r1 = _make_rollout("r1", prompt="prompt_A", rewards=[1.0], prompt_id="p1")

    # Compute advantages
    weights = assigner.compute([r1])

    # Should use pre-update baseline (0.75)
    # Advantage = 1.0 - 0.75 = 0.25
    assert weights[("r1", 0)] == pytest.approx(0.25)

    # Tracker should be updated now
    new_baseline = tracker.get_baseline("p1")
    assert new_baseline > 0.75  # Should increase after success


def test_spo_return_global_normalization():
    """Advantages are normalized globally across batch."""
    tracker = ValueTracker()
    assigner = SPOReturn(tracker=tracker, normalize_adv=True)

    r1 = _make_rollout("r1", prompt="p1", rewards=[1.0], prompt_id="p1")
    r2 = _make_rollout("r2", prompt="p2", rewards=[0.0], prompt_id="p2")
    r3 = _make_rollout("r3", prompt="p3", rewards=[1.0], prompt_id="p3")
    r4 = _make_rollout("r4", prompt="p4", rewards=[0.0], prompt_id="p4")

    weights = assigner.compute([r1, r2, r3, r4])

    # After normalization, advantages should have zero mean
    advs = [weights[("r1", 0)], weights[("r2", 0)], weights[("r3", 0)], weights[("r4", 0)]]
    mean_adv = sum(advs) / len(advs)
    assert mean_adv == pytest.approx(0.0, abs=1e-6)


def test_spo_return_no_grouping_needed():
    """SPO works without group_id - uses prompt_id or prev_obs."""
    tracker = ValueTracker()
    assigner = SPOReturn(tracker=tracker, normalize_adv=False)

    # Same prompt seen multiple times (would fail in GRPO without group_id)
    r1 = _make_rollout("r1", prompt="same_prompt", rewards=[1.0])
    r2 = _make_rollout("r2", prompt="same_prompt", rewards=[0.0])

    # Should work fine - each is independent
    weights = assigner.compute([r1, r2])

    # First uses baseline 0.5, second uses updated baseline
    assert ("r1", 0) in weights
    assert ("r2", 0) in weights


def test_spo_return_tracker_not_updated_when_disabled():
    """Can disable tracker updates for evaluation."""
    tracker = ValueTracker()
    assigner = SPOReturn(tracker=tracker, normalize_adv=False, update_tracker=False)

    r1 = _make_rollout("r1", prompt="p1", rewards=[1.0], prompt_id="p1")

    assigner.compute([r1])

    # Tracker should not be updated
    assert "p1" not in tracker


def test_default_prompt_key_fn_uses_prompt_id():
    """Default key function prefers prompt_id."""
    r = _make_rollout("r1", prompt="some_prompt", rewards=[1.0], prompt_id="my_id")
    key = default_prompt_key_fn(r)
    assert key == "my_id"


def test_default_prompt_key_fn_falls_back_to_prev_obs():
    """Default key function falls back to prev_obs."""
    r = _make_rollout("r1", prompt="fallback_prompt", rewards=[1.0])
    key = default_prompt_key_fn(r)
    assert key == "fallback_prompt"


# ---- make_spo Tests ----


def test_make_spo_returns_algo_and_tracker():
    """make_spo returns both algorithm and tracker."""
    algo, tracker = make_spo()

    assert algo.name == "spo"
    assert isinstance(tracker, ValueTracker)
    assert isinstance(algo.credit_assigner, SPOReturn)


def test_make_spo_uses_provided_tracker():
    """make_spo uses provided tracker instead of creating new one."""
    my_tracker = ValueTracker(d_half=0.5)  # Custom config
    algo, tracker = make_spo(tracker=my_tracker)

    assert tracker is my_tracker
    assert algo.credit_assigner.tracker is my_tracker


def test_make_spo_has_preprocess():
    """make_spo installs logprob preprocessor for clipped surrogate."""
    algo, _ = make_spo()

    assert algo.preprocess is not None
