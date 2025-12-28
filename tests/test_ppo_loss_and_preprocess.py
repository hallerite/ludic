from __future__ import annotations

from typing import Dict, Any, List

import pytest
import torch

from ludic.training.loss import ClippedSurrogateLoss, TokenClippedSurrogateLoss, CISPOLoss
from ludic.training.batching.micro_batching import collate_saw_items
from ludic.training.types import SAWItem, SAWBatch, ActorTokenLogps, SampleAttachments
from ludic.training.algorithm import RLAlgorithm, validate_actor_logps, make_gspo, make_grpo, make_cispo
from ludic.training.credit_assignment import MonteCarloReturn


def make_item(
    input_ids: List[int],
    action_mask: List[int],
    *,
    meta: Dict[str, Any] | None = None,
    attachments: SampleAttachments | None = None,
) -> SAWItem:
    L = len(input_ids)
    return SAWItem(
        input_ids=input_ids,
        attention_mask=[1] * L,
        action_mask=action_mask,
        weight=1.0,
        meta=meta or {},
        attachments=attachments or SampleAttachments(),
    )


def test_ppoloss_requires_old_logp():
    loss_fn = ClippedSurrogateLoss()
    logits = torch.zeros((1, 3, 2))
    batch = {
        "input_ids": torch.tensor([[0, 1, 0]], dtype=torch.long),
        "action_mask": torch.tensor([[0, 1, 1]], dtype=torch.float32),
        "weight": torch.tensor([1.0], dtype=torch.float32),
    }
    with pytest.raises(KeyError):
        loss_fn.compute(logits, batch)


def test_collate_sums_behavior_logprobs():
    # Two tokens in the action; per-token actor logps are precomputed.
    items = [
        make_item(
            [0, 1, 2],
            [0, 1, 1],
            attachments=SampleAttachments(actor_logps=ActorTokenLogps(token_logps=[-1.0, -2.0])),
        ),
        make_item(
            [0, 2, 1],
            [0, 1, 1],
            attachments=SampleAttachments(actor_logps=ActorTokenLogps(token_logps=[-1.0, -2.0])),
        ),
    ]
    batch = collate_saw_items(items, pad_token_id=0, device=torch.device("cpu"))
    assert "old_logp_action" in batch
    assert "actor_logps" in batch
    expected = torch.tensor([-3.0, -3.0], dtype=torch.float32)
    assert torch.allclose(batch["old_logp_action"], expected)


def test_collate_mixed_actor_logps_raises():
    items = [
        make_item(
            [0, 1, 2],
            [0, 1, 1],
            attachments=SampleAttachments(actor_logps=ActorTokenLogps(token_logps=[-1.0, -2.0])),
        ),
        make_item([0, 2, 1], [0, 1, 1], meta={}),  # missing
    ]
    with pytest.raises(ValueError):
        collate_saw_items(items, pad_token_id=0, device=torch.device("cpu"))


def test_ppopreprocess_requires_actor_logps():
    algo = RLAlgorithm(
        name="ppo",
        credit_assigner=MonteCarloReturn(),
        loss=ClippedSurrogateLoss(),
        preprocess=validate_actor_logps,
    )
    items = [
        make_item([0, 1, 2], [0, 1, 1], meta={}),
        make_item([0, 2, 1], [0, 1, 1], meta={}),
    ]
    saw_batch = SAWBatch(items=items, meta={})
    with pytest.raises(ValueError):
        assert algo.preprocess is not None
        algo.preprocess(saw_batch)


def test_ppopreprocess_async_batch_missing_logprobs_raises():
    algo = RLAlgorithm(
        name="ppo",
        credit_assigner=MonteCarloReturn(),
        loss=ClippedSurrogateLoss(),
        preprocess=validate_actor_logps,
    )
    items = [
        make_item([0, 1, 2], [0, 1, 1], meta={"policy_version": 1}),
    ]
    saw_batch = SAWBatch(items=items, meta={"source": "pipeline_redis"})
    with pytest.raises(ValueError):
        assert algo.preprocess is not None
        algo.preprocess(saw_batch)


def test_gspo_and_grpo_presets_defaults():
    gspo = make_gspo(group_size=2)
    assert isinstance(gspo.loss, ClippedSurrogateLoss)
    assert gspo.loss.clip_eps_low == pytest.approx(3e-4)
    assert gspo.loss.clip_eps_high == pytest.approx(4e-4)
    assert gspo.loss.length_normalize is True

    grpo = make_grpo(group_size=2)
    assert isinstance(grpo.loss, TokenClippedSurrogateLoss)
    assert grpo.loss.clip_eps_low == pytest.approx(0.2)
    assert grpo.loss.clip_eps_high == pytest.approx(0.27)


def test_cispo_preset_defaults():
    """Test make_cispo creates correct algorithm with paper defaults."""
    cispo = make_cispo(group_size=4)

    # Check algorithm metadata
    assert cispo.name == "cispo"
    assert cispo.preprocess is not None

    # Check loss configuration
    assert isinstance(cispo.loss, CISPOLoss)
    assert cispo.loss.clip_eps_low == pytest.approx(1e6)  # Effectively no lower bound
    assert cispo.loss.clip_eps_high == pytest.approx(0.2)
    assert cispo.loss.length_normalize is True


def test_cispo_preset_custom_params():
    """Test make_cispo respects custom parameters."""
    cispo = make_cispo(
        group_size=8,
        clip_eps_high=0.5,
        clip_eps_low=0.1,
        length_normalize=False,
        positive_only=True,
        name="custom_cispo",
    )

    assert cispo.name == "custom_cispo"
    assert isinstance(cispo.loss, CISPOLoss)
    assert cispo.loss.clip_eps_high == pytest.approx(0.5)
    assert cispo.loss.clip_eps_low == pytest.approx(0.1)
    assert cispo.loss.length_normalize is False
