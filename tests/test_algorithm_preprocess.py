from __future__ import annotations

import pytest

from ludic.training.algorithm import (
    compose_preprocess,
    drop_zero_weight_samples,
    make_grpo,
    make_reinforce,
    validate_actor_logps,
)
from ludic.training.types import SAWBatch, SAWItem, ActorTokenLogps, SampleAttachments


def _make_item(weight: float, *, has_actor_logps: bool) -> SAWItem:
    input_ids = [1, 2]
    action_mask = [0, 1]
    attachments = SampleAttachments()
    if has_actor_logps:
        attachments = SampleAttachments(actor_logps=ActorTokenLogps(token_logps=[-0.1]))
    return SAWItem(
        input_ids=input_ids,
        attention_mask=[1, 1],
        action_mask=action_mask,
        weight=weight,
        meta={},
        attachments=attachments,
    )


def test_make_grpo_drops_zero_weight_before_actor_logps_validation() -> None:
    algo = make_grpo(group_size=2, drop_zero_weight=True)
    batch = SAWBatch(
        items=[
            _make_item(0.0, has_actor_logps=False),
            _make_item(1.0, has_actor_logps=True),
        ],
        meta={},
    )
    assert algo.preprocess is not None
    processed = algo.preprocess(batch)
    assert len(processed.items) == 1
    assert processed.items[0].attachments.actor_logps is not None


def test_make_reinforce_can_drop_zero_weight() -> None:
    algo = make_reinforce(drop_zero_weight=True)
    batch = SAWBatch(
        items=[
            _make_item(0.0, has_actor_logps=False),
            _make_item(2.0, has_actor_logps=False),
        ],
        meta={},
    )
    assert algo.preprocess is not None
    processed = algo.preprocess(batch)
    assert len(processed.items) == 1
    assert processed.items[0].weight == 2.0


def test_drop_zero_weight_samples_rejects_negative_eps() -> None:
    batch = SAWBatch(
        items=[
            _make_item(1.0, has_actor_logps=False),
        ],
        meta={},
    )
    with pytest.raises(ValueError, match="eps must be >= 0"):
        drop_zero_weight_samples(batch, eps=-1.0)


def test_validate_actor_logps_len_mismatch_raises() -> None:
    item = SAWItem(
        input_ids=[1, 2, 3],
        attention_mask=[1, 1, 1],
        action_mask=[0, 1, 1],
        weight=1.0,
        meta={"rollout_id": "r1", "step_index": 0},
        attachments=SampleAttachments(actor_logps=ActorTokenLogps(token_logps=[-0.1])),
    )
    batch = SAWBatch(items=[item], meta={})
    with pytest.raises(ValueError, match="length mismatch"):
        validate_actor_logps(batch)


def test_validate_actor_logps_type_mismatch_raises() -> None:
    class _BadActorLogps:
        def __init__(self, token_logps: list[object]) -> None:
            self.token_logps = token_logps

    item = SAWItem(
        input_ids=[1, 2],
        attention_mask=[1, 1],
        action_mask=[0, 1],
        weight=1.0,
        meta={"rollout_id": "r1", "step_index": 0},
        attachments=SampleAttachments(actor_logps=_BadActorLogps(["bad"])),
    )
    batch = SAWBatch(items=[item], meta={})
    with pytest.raises(TypeError, match="must be a List\\[float\\]"):
        validate_actor_logps(batch)


def test_actor_token_logps_rejects_non_float_list() -> None:
    with pytest.raises(TypeError, match="must be a List\\[float\\]"):
        ActorTokenLogps(token_logps=["bad"])


def test_compose_preprocess_preserves_order() -> None:
    batch = SAWBatch(
        items=[
            _make_item(1.0, has_actor_logps=False),
        ],
        meta={"order": []},
    )

    def first(saw_batch: SAWBatch) -> SAWBatch:
        saw_batch.meta["order"].append("first")
        return saw_batch

    def second(saw_batch: SAWBatch) -> SAWBatch:
        saw_batch.meta["order"].append("second")
        return saw_batch

    preprocess = compose_preprocess(first, second)
    preprocess(batch)
    assert batch.meta["order"] == ["first", "second"]
