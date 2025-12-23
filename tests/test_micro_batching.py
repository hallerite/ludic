from __future__ import annotations

from typing import List

import pytest
import torch

from ludic.training.batching.micro_batching import (
    collate_micro_batches,
    collate_saw_items,
    split_items_by_token_budget,
)
from ludic.training.types import SAWItem, ActorTokenLogps, SampleAttachments


def _make_item(
    input_ids: List[int],
    action_mask: List[int],
    *,
    weight: float = 1.0,
    attachments: SampleAttachments | None = None,
) -> SAWItem:
    return SAWItem(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        action_mask=action_mask,
        weight=weight,
        meta={},
        attachments=attachments or SampleAttachments(),
    )


def test_split_items_respects_budget():
    items = [
        _make_item([1, 2], [0, 1]),
        _make_item([1, 2, 3, 4, 5], [0, 0, 1, 1, 1]),
        _make_item([1, 2, 3, 4, 5], [0, 0, 1, 1, 1]),
        _make_item([1, 2, 3, 4, 5], [0, 0, 1, 1, 1]),
    ]
    batches = split_items_by_token_budget(items, micro_token_budget=10, max_seq_len=5)
    assert len(batches) == 2
    for batch in batches:
        max_len = max(len(it.input_ids) for it in batch)
        assert max_len * len(batch) <= 10


def test_split_items_padding_effect_forces_split():
    items = [
        _make_item([1, 2, 3], [0, 1, 1]),
        _make_item([1, 2, 3, 4, 5], [0, 0, 1, 1, 1]),
    ]
    batches = split_items_by_token_budget(items, micro_token_budget=8, max_seq_len=5)
    assert len(batches) == 2


def test_split_items_truncates_long_sample():
    items = [
        _make_item(
            [1, 2, 3, 4, 5, 6],
            [0, 0, 0, 1, 1, 1],
            attachments=SampleAttachments(actor_logps=ActorTokenLogps(token_logps=[-0.1, -0.2, -0.3])),
        )
    ]
    batches = split_items_by_token_budget(items, micro_token_budget=12, max_seq_len=5)
    assert len(batches) == 1
    truncated = batches[0][0]
    assert len(truncated.input_ids) == 5
    assert truncated.meta["seq_len_truncated"] is True
    assert truncated.meta["seq_len_original"] == 6
    assert truncated.meta["seq_len_retained"] == 5
    assert truncated.meta["seq_len_retained_frac"] == pytest.approx(5.0 / 6.0)
    assert truncated.attachments.actor_logps is not None
    assert len(truncated.attachments.actor_logps.token_logps) == 2


def test_split_items_budget_must_cover_max_seq_len():
    items = [_make_item([1, 2, 3, 4], [0, 0, 1, 1])]
    with pytest.raises(ValueError, match="micro_token_budget"):
        split_items_by_token_budget(items, micro_token_budget=3, max_seq_len=4)


def test_collate_saw_items_shapes_and_padding():
    items = [
        _make_item([1, 2, 3], [0, 1, 1]),
        _make_item([4, 5], [0, 1]),
    ]
    batch = collate_saw_items(items, pad_token_id=0, device=torch.device("cpu"))
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["action_mask"].tolist() == [[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]


def test_collate_saw_items_actor_logps_length_mismatch():
    items = [
        _make_item(
            [1, 2, 3],
            [0, 1, 1],
            attachments=SampleAttachments(actor_logps=ActorTokenLogps(token_logps=[-1.0])),
        )
    ]
    with pytest.raises(ValueError, match="Length mismatch"):
        collate_saw_items(items, pad_token_id=0, device=torch.device("cpu"))


def test_collate_micro_batches_splits_and_shapes():
    items = [
        _make_item([1, 2, 3, 4], [0, 0, 1, 1]),
        _make_item([5, 6, 7, 8], [0, 0, 1, 1]),
        _make_item([9, 10, 11, 12], [0, 0, 1, 1]),
    ]
    micro_batches = collate_micro_batches(
        items,
        pad_token_id=0,
        device=torch.device("cpu"),
        micro_token_budget=8,
        max_seq_len=4,
    )
    assert len(micro_batches) == 2
    assert micro_batches[0].num_items == 2
    assert micro_batches[1].num_items == 1
    assert micro_batches[0].tensors["input_ids"].shape == (2, 4)
    assert micro_batches[1].tensors["input_ids"].shape == (1, 4)


def test_collate_micro_batches_empty_returns_empty():
    micro_batches = collate_micro_batches(
        [],
        pad_token_id=0,
        device=torch.device("cpu"),
        micro_token_budget=8,
        max_seq_len=4,
    )
    assert micro_batches == []
