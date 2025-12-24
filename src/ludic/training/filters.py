"""
Sample filtering utilities for training data.

These filters operate on SAWItems and can be used to drop samples
based on metadata predicates. Filters return True to KEEP a sample.
"""
from __future__ import annotations
from typing import Sequence

from ludic.training.types import SAWItem, SampleFilter
from ludic.types import Step, AgentStep, EnvironmentStep


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


def keep_env_steps(item: SAWItem) -> bool:
    """Keep only environment steps."""
    return item.meta.get("step_kind") == "env"


def keep_agent_steps(item: SAWItem) -> bool:
    """Keep only agent steps."""
    return item.meta.get("step_kind") == "agent"


def default_step_selector(step: Step) -> bool:
    """
    Default rollout step selection for step-wise training:
      - all env steps
      - env-targeted agent steps that failed parsing

    Note: online RolloutEngine batching now concatenates full agent turns by default,
    so this selector is primarily used by offline/legacy pipelines.
    """
    if isinstance(step, EnvironmentStep):
        return True
    if isinstance(step, AgentStep):
        return step.action_target == "env" and bool(step.info.get("parse_error"))
    return False


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
