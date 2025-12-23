from __future__ import annotations

from ludic.training.algorithm import make_grpo, make_reinforce
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
