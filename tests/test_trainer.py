from __future__ import annotations

from types import SimpleNamespace
from typing import List, Mapping

import pytest
import torch
from torch import nn

from ludic.training.algorithm import RLAlgorithm
from ludic.training.config import TrainerConfig
from ludic.training.credit_assignment import ConstantCredit
from ludic.training.loss import Loss
from ludic.training.stats import Reducer, default_reducers
from ludic.training.trainer import Trainer
from ludic.training.types import SAWBatch, SAWItem, BatchSource


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
        stats = {"loss": float(weight_mean.detach().cpu())}
        return loss, stats


class LogitsMeanLoss(Loss):
    def compute(self, logits: torch.Tensor, batch: dict[str, torch.Tensor]):
        loss = logits.mean()
        stats = {"loss": float(loss.detach().cpu())}
        return loss, stats


class MockPublisher:
    def __init__(self) -> None:
        self.calls: List[dict[str, object]] = []

    def publish(self, state_dict: dict[str, torch.Tensor], version: int | None = None) -> None:
        self.calls.append({"state_dict": state_dict, "version": version})


def _make_item(length: int, weight: float, *, meta: dict | None = None) -> SAWItem:
    input_ids = [1] * length
    return SAWItem(
        input_ids=input_ids,
        attention_mask=[1] * length,
        action_mask=[1] * length,
        weight=weight,
        meta=meta or {},
    )


def _trainer_for_batch(
    batch: SAWBatch,
    *,
    loss: Loss,
    max_seq_len: int = 5,
    micro_token_budget: int = 10,
    max_lag: int | None = None,
    sync_every_steps: int = 0,
    preprocess=None,
    reducers: Mapping[str, Reducer] | None = None,
) -> Trainer:
    algo = RLAlgorithm(
        name="test_algo",
        credit_assigner=ConstantCredit(),
        loss=loss,
        preprocess=preprocess,
    )
    return Trainer(
        model=DummyModel(),
        algo=algo,
        batch_source=FixedBatchSource(batch),
        publisher=None,
        cfg=TrainerConfig(
            model_device="cpu",
            pad_token_id=0,
            max_seq_len=max_seq_len,
            micro_token_budget=micro_token_budget,
            max_lag=max_lag,
            sync_every_steps=sync_every_steps,
        ),
        reducers=reducers,
    )


def test_trainer_requires_publisher_with_sync():
    batch = SAWBatch(items=[_make_item(2, 1.0)], meta={})
    with pytest.raises(ValueError, match="PolicyPublisher"):
        _trainer_for_batch(
            batch,
            loss=LogitsMeanLoss(),
            sync_every_steps=1,
        )


def test_trainer_rejects_micro_budget_smaller_than_max_seq_len():
    batch = SAWBatch(items=[_make_item(2, 1.0)], meta={})
    with pytest.raises(ValueError, match="micro_token_budget"):
        _trainer_for_batch(
            batch,
            loss=LogitsMeanLoss(),
            max_seq_len=5,
            micro_token_budget=4,
        )


def test_trainer_truncates_long_sample():
    batch = SAWBatch(items=[_make_item(6, 1.0)], meta={})
    trainer = _trainer_for_batch(
        batch,
        loss=LogitsMeanLoss(),
        max_seq_len=5,
        micro_token_budget=10,
    )
    stats = trainer.train_step_sync()
    assert "train/loss" in stats


def test_trainer_reports_seq_len_truncated_rate():
    batch = SAWBatch(items=[_make_item(6, 1.0)], meta={})
    trainer = _trainer_for_batch(
        batch,
        loss=LogitsMeanLoss(),
        max_seq_len=5,
        micro_token_budget=10,
        reducers=default_reducers(),
    )
    stats = trainer.train_step_sync()
    assert stats["train/seq_len_truncated_rate"] == pytest.approx(1.0)


def test_trainer_filters_max_lag_items():
    batch = SAWBatch(
        items=[
            _make_item(2, 3.0, meta={"policy_version": -1}),
            _make_item(2, 1.0, meta={"policy_version": 0}),
        ],
        meta={},
    )
    trainer = _trainer_for_batch(
        batch,
        loss=WeightMeanLoss(),
        max_seq_len=5,
        micro_token_budget=10,
        max_lag=0,
    )
    stats = trainer.train_step_sync()
    assert stats["train/loss"] == pytest.approx(1.0)


def test_trainer_calls_preprocess():
    batch = SAWBatch(items=[_make_item(2, 1.0)], meta={})
    called = {"value": False}

    def preprocess(saw_batch: SAWBatch) -> SAWBatch:
        called["value"] = True
        for item in saw_batch.items:
            item.weight = 4.0
        return saw_batch

    trainer = _trainer_for_batch(
        batch,
        loss=WeightMeanLoss(),
        preprocess=preprocess,
    )
    stats = trainer.train_step_sync()
    assert called["value"] is True
    assert stats["train/loss"] == pytest.approx(4.0)


def test_trainer_updates_parameters():
    batch = SAWBatch(items=[_make_item(2, 1.0)], meta={})
    trainer = _trainer_for_batch(batch, loss=LogitsMeanLoss())
    before = [p.detach().clone() for p in trainer.model.parameters()]
    trainer.train_step_sync()
    after = list(trainer.model.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_trainer_publishes_weights_on_schedule():
    batch = SAWBatch(items=[_make_item(2, 1.0)], meta={})
    publisher = MockPublisher()
    trainer = Trainer(
        model=DummyModel(),
        algo=RLAlgorithm(
            name="test_algo",
            credit_assigner=ConstantCredit(),
            loss=LogitsMeanLoss(),
        ),
        batch_source=FixedBatchSource(batch),
        publisher=publisher,
        cfg=TrainerConfig(
            model_device="cpu",
            pad_token_id=0,
            max_seq_len=5,
            micro_token_budget=10,
            sync_every_steps=2,
        ),
    )
    trainer.train_step_sync()
    assert publisher.calls == []
    trainer.train_step_sync()
    assert len(publisher.calls) == 1
    call = publisher.calls[0]
    assert call["version"] == 2
    state_dict = call["state_dict"]
    assert isinstance(state_dict, dict)
    assert set(state_dict.keys()) >= {"embed.weight", "lm_head.weight", "lm_head.bias"}


def test_trainer_applies_param_filter_on_publish():
    batch = SAWBatch(items=[_make_item(2, 1.0)], meta={})
    publisher = MockPublisher()

    def only_embed(name: str, _tensor: torch.Tensor) -> bool:
        return name == "embed.weight"

    trainer = Trainer(
        model=DummyModel(),
        algo=RLAlgorithm(
            name="test_algo",
            credit_assigner=ConstantCredit(),
            loss=LogitsMeanLoss(),
        ),
        batch_source=FixedBatchSource(batch),
        publisher=publisher,
        param_filter=only_embed,
        cfg=TrainerConfig(
            model_device="cpu",
            pad_token_id=0,
            max_seq_len=5,
            micro_token_budget=10,
            sync_every_steps=1,
        ),
    )
    trainer.train_step_sync()
    assert len(publisher.calls) == 1
    state_dict = publisher.calls[0]["state_dict"]
    assert set(state_dict.keys()) == {"embed.weight"}
