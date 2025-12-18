from __future__ import annotations

import math

import pytest
import torch

from ludic.training.loss import OnPolicyDistillationLoss
from ludic.training.trainer import _collate_saw_items
from ludic.training.types import SAWItem


def make_item(
    *,
    input_ids: list[int],
    action_mask: list[int],
    old_token_logprobs: list[float],
    teacher_token_logprobs: list[float],
    weight: float = 0.0,
) -> SAWItem:
    L = len(input_ids)
    return SAWItem(
        input_ids=input_ids,
        attention_mask=[1] * L,
        action_mask=action_mask,
        weight=weight,
        meta={
            "old_token_logprobs": list(old_token_logprobs),
            "teacher_token_logprobs": list(teacher_token_logprobs),
        },
    )


def test_opd_loss_matches_token_level_objective_uniform_logits():
    # input_ids length 3 => token-level shifted length 2.
    item = make_item(
        input_ids=[0, 1, 2],
        action_mask=[0, 1, 1],
        old_token_logprobs=[-1.0, -1.0],
        teacher_token_logprobs=[-0.5, -0.5],
        weight=0.0,
    )
    batch = _collate_saw_items([item], pad_token_id=0, device=torch.device("cpu"))

    # Uniform logits over vocab=3 => logp = -ln(3) for any token.
    logits = torch.zeros((1, 3, 3), dtype=torch.float32)
    loss_fn = OnPolicyDistillationLoss(opd_coef=1.0, env_weight_coef=0.0, length_normalize=False)
    loss, _stats = loss_fn.compute(logits, batch)

    # adv = teacher - old = 0.5 per token, loss = -mean_t(adv * logp) = -0.5 * (-ln(3)) = 0.5*ln(3)
    expected = 0.5 * math.log(3.0)
    assert pytest.approx(float(loss), rel=1e-6) == expected


def test_opd_loss_env_weight_coef_broadcasts_over_tokens():
    item = make_item(
        input_ids=[0, 1, 2],
        action_mask=[0, 1, 1],
        old_token_logprobs=[-1.0, -1.0],
        teacher_token_logprobs=[-0.5, -0.5],
        weight=2.0,
    )
    batch = _collate_saw_items([item], pad_token_id=0, device=torch.device("cpu"))
    logits = torch.zeros((1, 3, 3), dtype=torch.float32)

    # per-token adv = (teacher-old)=0.5 plus env term 2.0 => 2.5 per token
    loss_fn = OnPolicyDistillationLoss(opd_coef=1.0, env_weight_coef=1.0, length_normalize=False)
    loss, _stats = loss_fn.compute(logits, batch)
    expected = 2.5 * math.log(3.0)
    assert pytest.approx(float(loss), rel=1e-6) == expected

