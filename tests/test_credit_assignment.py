import pytest
import math

from ludic.types import Rollout, EnvironmentStep, TokenTrace
from ludic.training.credit_assignment import (
    MonteCarloReturn,
    EpisodicReturn,
    PerStepReward,
    GroupNormalizedReturn,
)

# ---- Helper to build a simple rollout ----

def _make_rollout(
    id: str,
    *,
    prompt: str = "prompt_default",
    rewards: list[float],
    group_id: str | None = None,
) -> Rollout:
    """
    Creates a simple rollout with the given params.
    The rollout's total_reward will be the sum of the rewards list.
    """
    request_meta = {}
    if group_id is not None:
        request_meta["group_id"] = group_id
    rollout = Rollout(id=id, meta={"request_meta": request_meta})
    obs = prompt  # First prev_obs is the prompt

    if not rewards:
        return rollout

    for i, reward in enumerate(rewards):
        next_obs = f"obs_{i+1}" if i < len(rewards) - 1 else None
        rollout.steps.append(
            EnvironmentStep(
                index=i,
                prev_obs=obs,
                action=f"action_{i}",
                parsed_action=f"action_{i}",
                next_obs=next_obs,
                source_agent_step_id=f"agent_{i}",
                agent_step_ids=[f"agent_{i}"],
                reward=reward,
                truncated=False,
                terminated=(i == len(rewards) - 1),
                info={},
                trace=TokenTrace(
                    prompt_token_ids=[1],
                    completion_token_ids=[2],
                ),
            )
        )
        obs = next_obs or ""
    return rollout


# ---- MonteCarloReturn Tests ----

def test_monte_carlo_return_gamma_1():
    """Test standard (gamma=1.0) return-to-go."""
    r1 = _make_rollout("r1", prompt="p1", rewards=[0.0, 0.0, 1.0]) # len 3
    r2 = _make_rollout("r2", prompt="p2", rewards=[-1.0])          # len 1

    assigner = MonteCarloReturn(gamma=1.0)
    weights = assigner.compute([r1, r2])

    # G_t = r_t + r_{t+1} + ...
    # r1: [0.0, 0.0, 1.0]
    # G:  [1.0, 1.0, 1.0]
    assert weights[("r1", 0)] == pytest.approx(1.0)
    assert weights[("r1", 1)] == pytest.approx(1.0)
    assert weights[("r1", 2)] == pytest.approx(1.0)

    # r2: [-1.0]
    # G:  [-1.0]
    assert weights[("r2", 0)] == pytest.approx(-1.0)

    assert len(weights) == 4


def test_monte_carlo_return_gamma_0_9():
    """Test discounted (gamma=0.9) return-to-go."""
    r1 = _make_rollout("r1", prompt="p1", rewards=[1.0, 2.0, 4.0]) # len 3

    assigner = MonteCarloReturn(gamma=0.9)
    weights = assigner.compute([r1])

    # r1: [1.0, 2.0, 4.0]
    # G_2 = 4.0
    # G_1 = 2.0 + 0.9 * G_2 = 2.0 + 0.9 * 4.0 = 2.0 + 3.6 = 5.6
    # G_0 = 1.0 + 0.9 * G_1 = 1.0 + 0.9 * 5.6 = 1.0 + 5.04 = 6.04
    assert weights[("r1", 0)] == pytest.approx(6.04)
    assert weights[("r1", 1)] == pytest.approx(5.6)
    assert weights[("r1", 2)] == pytest.approx(4.0)
    assert len(weights) == 3

# ---- Other Assigner Tests ----

def test_episodic_return():
    """Test that all steps in a rollout get the total sum."""
    r1 = _make_rollout("r1", prompt="p1", rewards=[0.0, 0.0, 1.0]) # total = 1.0
    r2 = _make_rollout("r2", prompt="p2", rewards=[-1.0, -0.5])    # total = -1.5

    assigner = EpisodicReturn()
    weights = assigner.compute([r1, r2])

    assert weights[("r1", 0)] == pytest.approx(1.0)
    assert weights[("r1", 1)] == pytest.approx(1.0)
    assert weights[("r1", 2)] == pytest.approx(1.0)

    assert weights[("r2", 0)] == pytest.approx(-1.5)
    assert weights[("r2", 1)] == pytest.approx(-1.5)
    assert len(weights) == 5


def test_per_step_reward():
    """Test that all steps get their own immediate reward."""
    r1 = _make_rollout("r1", prompt="p1", rewards=[1.0, 2.0, 4.0])

    assigner = PerStepReward()
    weights = assigner.compute([r1])

    assert weights[("r1", 0)] == pytest.approx(1.0)
    assert weights[("r1", 1)] == pytest.approx(2.0)
    assert weights[("r1", 2)] == pytest.approx(4.0)
    assert len(weights) == 3


# ---- GroupNormalizedReturn Tests ----

def test_group_normalized_return_groups_by_group_id():
    """
    Tests that rollouts are grouped by their group_id in request_meta,
    not by prompt content.
    """
    # Group A: group_id="group_A". Total rewards are 10.0 and 20.0
    r1 = _make_rollout("r1", prompt="prompt_A", rewards=[10.0], group_id="group_A")
    r2 = _make_rollout("r2", prompt="prompt_A", rewards=[20.0], group_id="group_A")

    # Group B: group_id="group_B". Total reward is 5.0 and 15.0
    r3 = _make_rollout("r3", prompt="prompt_A", rewards=[5.0], group_id="group_B")  # same prompt!
    r4 = _make_rollout("r4", prompt="prompt_A", rewards=[15.0], group_id="group_B")

    assigner = GroupNormalizedReturn(group_size=2, normalize_adv=False)
    weights = assigner.compute([r1, r2, r3, r4])

    # Group A: rewards=[10, 20], mean=15.0
    # Adv(r1) = 10.0 - 15.0 = -5.0
    # Adv(r2) = 20.0 - 15.0 = 5.0
    assert weights[("r1", 0)] == pytest.approx(-5.0)
    assert weights[("r2", 0)] == pytest.approx(5.0)

    # Group B: rewards=[5, 15], mean=10.0
    # Adv(r3) = 5.0 - 10.0 = -5.0
    # Adv(r4) = 15.0 - 10.0 = 5.0
    assert weights[("r3", 0)] == pytest.approx(-5.0)
    assert weights[("r4", 0)] == pytest.approx(5.0)
    assert len(weights) == 4


def test_group_normalized_return_handles_zero_std_dev():
    """
    Tests that normalization works correctly when all rewards in a
    group are identical (std_dev = 0), avoiding division by zero.
    """
    # Group A: All rewards are 10.0
    r1 = _make_rollout("r1", prompt="prompt_A", rewards=[10.0], group_id="group_A")
    r2 = _make_rollout("r2", prompt="prompt_A", rewards=[10.0], group_id="group_A")

    assigner = GroupNormalizedReturn(group_size=2, normalize_adv=True)
    weights = assigner.compute([r1, r2])

    # Group A: rewards=[10, 10], mean=10.0
    # Adv(pre-norm) = [0.0, 0.0]
    # StdDev = 0.0
    # Adv(post-norm) = 0.0 / (0.0 + 1e-8) = 0.0

    adv1 = weights[("r1", 0)]
    adv2 = weights[("r2", 0)]

    # Check for NaN or Inf
    assert not math.isnan(adv1) and not math.isinf(adv1)
    assert not math.isnan(adv2) and not math.isinf(adv2)

    # Check that advantages are 0
    assert adv1 == pytest.approx(0.0)
    assert adv2 == pytest.approx(0.0)
    assert len(weights) == 2


def test_group_normalized_return_positive_only_clips_negatives():
    """
    Tests that positive_only clips negative advantages to zero (no punishment).
    """
    r1 = _make_rollout("r1", prompt="prompt_A", rewards=[1.0], group_id="group_A")
    r2 = _make_rollout("r2", prompt="prompt_A", rewards=[0.0], group_id="group_A")

    assigner = GroupNormalizedReturn(group_size=2, normalize_adv=False, positive_only=True)
    weights = assigner.compute([r1, r2])

    # Rewards [1.0, 0.0] -> baseline 0.5 -> raw advantages [0.5, -0.5]
    # positive_only -> [0.5, 0.0]
    assert weights[("r1", 0)] == pytest.approx(0.5)
    assert weights[("r2", 0)] == pytest.approx(0.0)
    assert len(weights) == 2


def test_group_normalized_return_requires_group_id():
    """
    Tests that missing group_id raises a clear error.
    """
    # No group_id set
    r1 = _make_rollout("r1", prompt="prompt_A", rewards=[10.0])

    assigner = GroupNormalizedReturn(group_size=1)

    with pytest.raises(ValueError, match="missing group_id"):
        assigner.compute([r1])


def test_group_normalized_return_group_size_mismatch():
    """
    Tests that group_size mismatch raises a clear error.
    """
    r1 = _make_rollout("r1", prompt="prompt_A", rewards=[10.0], group_id="group_A")
    r2 = _make_rollout("r2", prompt="prompt_A", rewards=[20.0], group_id="group_A")
    r3 = _make_rollout("r3", prompt="prompt_A", rewards=[30.0], group_id="group_A")  # 3 in group

    assigner = GroupNormalizedReturn(group_size=4)  # but expect 4

    with pytest.raises(ValueError, match="Group size mismatch"):
        assigner.compute([r1, r2, r3])


def test_group_normalized_return_invalid_group_size():
    """
    Tests that invalid group_size raises at construction.
    """
    with pytest.raises(ValueError, match="group_size must be positive"):
        GroupNormalizedReturn(group_size=0)

    with pytest.raises(ValueError, match="group_size must be positive"):
        GroupNormalizedReturn(group_size=-1)
