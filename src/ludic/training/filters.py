"""
Sample filtering utilities for training data.

These filters operate on SAWItems and can be used to drop samples
based on metadata predicates. Filters return True to KEEP a sample.
"""
from __future__ import annotations
from typing import Sequence

from ludic.training.types import SAWItem, SampleFilter


def keep_all(item: SAWItem) -> bool:
    """Default filter that keeps all samples."""
    return True


def drop_truncated(item: SAWItem) -> bool:
    """Drop samples marked truncated (env/protocol or seq-len truncation)."""
    return not item.meta.get("truncated", False)


def drop_incomplete_completions(item: SAWItem) -> bool:
    """Drop samples where the completion hit max_tokens (keep complete)."""
    return item.meta.get("finish_reason") != "length"


def drop_parse_errors(item: SAWItem) -> bool:
    """Drop samples where parsing failed (keep successful parses)."""
    return not item.meta.get("parse_error", False)


def combine(*filters: SampleFilter) -> SampleFilter:
    """
    Combine multiple filters with AND logic.

    A sample is kept only if ALL filters return True.

    Example:
        combined = combine(drop_truncated, drop_incomplete_completions)
        filtered_items = [item for item in items if combined(item)]
    """
    def combined_filter(item: SAWItem) -> bool:
        return all(f(item) for f in filters)
    return combined_filter


def any_of(*filters: SampleFilter) -> SampleFilter:
    """
    Combine multiple filters with OR logic.

    A sample is kept if ANY filter returns True.
    """
    def or_filter(item: SAWItem) -> bool:
        return any(f(item) for f in filters)
    return or_filter


def apply_filter(
    items: Sequence[SAWItem],
    sample_filter: SampleFilter,
) -> list[SAWItem]:
    """Apply a filter to a sequence of SAWItems, returning kept items."""
    return [item for item in items if sample_filter(item)]
