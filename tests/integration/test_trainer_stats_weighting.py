from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest
import torch
from torch import nn

from ludic.training.algorithm import RLAlgorithm
from ludic.training.credit_assignment import ConstantCredit
from ludic.training.trainer import Trainer
from ludic.training.config import TrainerConfig
from ludic.training.types import SAWBatch, SAWItem, BatchSource
from ludic.training.loss import Loss

pytestmark = [pytest.mark.integration]


class FixedBatchSource(BatchSource):
    def __init__(self, batch: SAWBatch) -> None:
        self._batch = batch

    async def next_batch(self) -> SAWBatch:
        return SAWBatch(items=list(self._batch.items), meta=dict(self._batch.meta))


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)


class WeightMeanLoss(Loss):
    def compute(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]):
        weight_mean = batch["weight"].mean()
        loss = logits.mean() * weight_mean
        stats = {"loss": weight_mean.detach()}
        return loss, stats


def _make_item(length: int, weight: float) -> SAWItem:
    input_ids = [1] * length
    return SAWItem(
        input_ids=input_ids,
        attention_mask=[1] * length,
        action_mask=[1] * length,
        weight=weight,
        meta={},
    )


def test_trainer_weights_micro_batch_stats():
    # Lengths chosen so micro_token_budget=5 produces micro sizes [3, 1].
    items: List[SAWItem] = [
        _make_item(1, 3.0),
        _make_item(1, 3.0),
        _make_item(1, 3.0),
        _make_item(5, 1.0),
    ]
    saw_batch = SAWBatch(items=items, meta={})
    batch_source = FixedBatchSource(saw_batch)

    algo = RLAlgorithm(
        name="weight_mean",
        credit_assigner=ConstantCredit(),
        loss=WeightMeanLoss(),
    )

    trainer = Trainer(
        model=DummyModel(),
        algo=algo,
        batch_source=batch_source,
        publisher=None,
        cfg=TrainerConfig(
            model_device="cpu",
            pad_token_id=0,
            max_seq_len=5,
            micro_token_budget=5,
            sync_every_steps=0,
        ),
    )

    stats = trainer.train_step_sync()
    assert stats["train/loss"] == pytest.approx(2.5)
