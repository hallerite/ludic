from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from ludic.training.algorithm import make_opd
from ludic.training.types import SAWBatch, SAWItem


def make_item(
    input_ids: List[int],
    action_mask: List[int],
    *,
    meta: Dict[str, Any] | None = None,
    weight: float = 1.0,
) -> SAWItem:
    L = len(input_ids)
    return SAWItem(
        input_ids=input_ids,
        attention_mask=[1] * L,
        action_mask=action_mask,
        weight=weight,
        meta=meta or {},
    )


def test_opd_preprocess_copies_old_logprobs_and_validates_teacher():
    algo = make_opd(subtract_student_logprobs=True)

    # completion_logprobs should be copied into old_token_logprobs.
    items = [
        make_item(
            [0, 1, 2],
            [0, 1, 1],
            meta={
                "completion_logprobs": [-0.1, -0.2],
                "teacher_token_logprobs": [-math.log(3.0), -math.log(3.0)],
            },
            weight=7.0,
        )
    ]
    saw_batch = SAWBatch(items=items, meta={})
    processed = algo.preprocess_batch(saw_batch, model=None, pad_token_id=0)
    it = processed.items[0]

    assert it.meta.get("old_token_logprobs") == [-0.1, -0.2]
    assert it.meta.get("teacher_token_logprobs") == [-math.log(3.0), -math.log(3.0)]
    assert pytest.approx(it.meta["opd_old_logp_action"], rel=1e-6) == (-0.1 + -0.2)
    assert pytest.approx(it.meta["opd_teacher_logp_action"], rel=1e-6) == (-math.log(3.0) * 2)
    assert it.weight == 7.0


def test_opd_requires_teacher_token_logprobs_if_no_scorer():
    algo = make_opd()
    items = [
        make_item(
            [0, 1, 2],
            [0, 1, 1],
            meta={"completion_logprobs": [-0.1, -0.2]},
        )
    ]
    saw_batch = SAWBatch(items=items, meta={})
    with pytest.raises(ValueError, match="teacher_token_logprobs"):
        algo.preprocess_batch(saw_batch, model=None, pad_token_id=0)


def test_opd_requires_old_logprobs_when_subtracting_student():
    algo = make_opd(subtract_student_logprobs=True)
    items = [
        make_item(
            [0, 1, 2],
            [0, 1, 1],
            meta={"teacher_token_logprobs": [-0.3, -0.4]},
        )
    ]
    saw_batch = SAWBatch(items=items, meta={})
    with pytest.raises(ValueError, match="old_token_logprobs"):
        algo.preprocess_batch(saw_batch, model=None, pad_token_id=0)
