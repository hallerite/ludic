from __future__ import annotations

from ludic.training.filters import (
    any_of,
    apply_filter,
    combine,
    drop_incomplete_completions,
    drop_parse_errors,
    drop_truncated,
    keep_all,
)
from ludic.training.types import SAWItem


def _item(**meta) -> SAWItem:
    return SAWItem(
        input_ids=[1],
        attention_mask=[1],
        action_mask=[1],
        weight=1.0,
        meta=dict(meta),
    )


def test_keep_all_keeps_everything() -> None:
    assert keep_all(_item()) is True


def test_drop_truncated() -> None:
    assert drop_truncated(_item(truncated=False)) is True
    assert drop_truncated(_item(truncated=True)) is False
    # Missing key defaults to keep
    assert drop_truncated(_item()) is True


def test_drop_incomplete_completions() -> None:
    assert drop_incomplete_completions(_item(finish_reason="stop")) is True
    assert drop_incomplete_completions(_item(finish_reason="length")) is False
    # Missing key defaults to keep
    assert drop_incomplete_completions(_item()) is True


def test_drop_parse_errors() -> None:
    assert drop_parse_errors(_item(parse_error=False)) is True
    assert drop_parse_errors(_item(parse_error=True)) is False
    # Missing key defaults to keep
    assert drop_parse_errors(_item()) is True


def test_combine_and_logic() -> None:
    f = combine(drop_truncated, drop_parse_errors)
    assert f(_item(truncated=False, parse_error=False)) is True
    assert f(_item(truncated=True, parse_error=False)) is False
    assert f(_item(truncated=False, parse_error=True)) is False


def test_any_of_or_logic() -> None:
    f = any_of(drop_truncated, drop_parse_errors)
    assert f(_item(truncated=False, parse_error=False)) is True
    # Either predicate keeping the item is enough
    assert f(_item(truncated=True, parse_error=False)) is True
    assert f(_item(truncated=False, parse_error=True)) is True
    # Both drop => drop
    assert f(_item(truncated=True, parse_error=True)) is False


def test_apply_filter() -> None:
    items = [
        _item(step=0, truncated=False),
        _item(step=1, truncated=True),
        _item(step=2, truncated=False),
    ]
    kept = apply_filter(items, drop_truncated)
    assert [i.meta["step"] for i in kept] == [0, 2]

