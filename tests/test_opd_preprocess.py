from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from ludic.training.algorithm import make_opd
from ludic.training.types import SAWBatch, SAWItem, ActorTokenLogps, TeacherTokenLogps, SampleAttachments


def make_item(
    input_ids: List[int],
    action_mask: List[int],
    *,
    meta: Dict[str, Any] | None = None,
    weight: float = 1.0,
    attachments: SampleAttachments | None = None,
) -> SAWItem:
    L = len(input_ids)
    return SAWItem(
        input_ids=input_ids,
        attention_mask=[1] * L,
        action_mask=action_mask,
        weight=weight,
        meta=meta or {},
        attachments=attachments or SampleAttachments(),
    )


def test_opd_preprocess_validates_teacher_and_actor_logprobs():
    algo = make_opd()

    # actor/teacher logprobs are required and used to compute per-action sums.
    items = [
        make_item(
            [0, 1, 2],
            [0, 1, 1],
            attachments=SampleAttachments(
                actor_logps=ActorTokenLogps(token_logps=[-0.1, -0.2]),
                teacher_logps=TeacherTokenLogps(token_logps=[-math.log(3.0), -math.log(3.0)]),
            ),
            weight=7.0,
        )
    ]
    saw_batch = SAWBatch(items=items, meta={})
    assert algo.preprocess is not None
    processed = algo.preprocess(saw_batch)
    it = processed.items[0]

    assert "opd_old_logp_action" not in it.meta
    assert "opd_teacher_logp_action" not in it.meta
    assert it.weight == 7.0


def test_opd_requires_teacher_logprobs_if_no_scorer():
    algo = make_opd()
    items = [
        make_item(
            [0, 1, 2],
            [0, 1, 1],
            attachments=SampleAttachments(actor_logps=ActorTokenLogps(token_logps=[-0.1, -0.2])),
        )
    ]
    saw_batch = SAWBatch(items=items, meta={})
    with pytest.raises(ValueError, match="teacher_logps"):
        assert algo.preprocess is not None
        algo.preprocess(saw_batch)


def test_opd_requires_actor_logprobs():
    algo = make_opd()
    items = [
        make_item(
            [0, 1, 2],
            [0, 1, 1],
            attachments=SampleAttachments(
                teacher_logps=TeacherTokenLogps(token_logps=[-0.3, -0.4])
            ),
        )
    ]
    saw_batch = SAWBatch(items=items, meta={})
    with pytest.raises(ValueError, match="actor_logps"):
        assert algo.preprocess is not None
        algo.preprocess(saw_batch)
