from __future__ import annotations

import pytest
import torch

from ludic.training.stats import aggregate_stats
from ludic.training.types import SAWBatch, SAWItem


def _item() -> SAWItem:
    return SAWItem(
        input_ids=[1, 2, 3],
        attention_mask=[1, 1, 1],
        action_mask=[0, 1, 1],
        weight=1.0,
        meta={},
    )


def test_aggregate_stats_weights_micro_batches():
    micro_stats = [
        {"loss": torch.tensor(1.0), "logp_mean": torch.tensor(-1.0)},
        {"loss": torch.tensor(3.0), "logp_mean": torch.tensor(-3.0)},
    ]
    saw_batches = [
        SAWBatch(
            items=[_item(), _item()],
            meta={"target_rollouts": 2, "avg_total_reward": 0.0, "avg_completion_length": 3.0},
        )
    ]
    out = aggregate_stats(
        micro_stats,
        saw_batches,
        micro_batch_sizes=[1, 3],
    )
    assert out["loss"] == pytest.approx(2.5)
    assert out["logp_mean"] == pytest.approx(-2.5)


def test_aggregate_stats_uses_effective_rollouts_when_present():
    micro_stats = [{"loss": torch.tensor(1.0)}]
    saw_batches = [
        SAWBatch(
            items=[_item(), _item()],
            meta={"target_rollouts": 3, "effective_rollouts": 1},
        )
    ]
    out = aggregate_stats(micro_stats, saw_batches)
    assert out["target_rollouts"] == pytest.approx(3.0)
    assert out["effective_rollouts"] == pytest.approx(1.0)
