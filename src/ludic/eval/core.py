"""
Evaluation helpers built on RolloutEngine.

This module is intentionally training-agnostic: it consumes RolloutEngine and
RolloutRequests and produces reduced metrics over step-level eval items.

Eval items are dicts shaped like training `SAWItem.meta` (eval uses env steps,
while training batches concatenate full turns into one item).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Mapping, Sequence

from ludic.training.batching.rollout_engine import RolloutEngine
from ludic.training.stats import Reducer, apply_reducers_to_records
from ludic.training.types import RolloutRequest
from ludic.types import Rollout

def _eval_items_from_rollout(rollout: Rollout) -> List[Dict[str, Any]]:
    """
    Convert a rollout into eval "items" that match the training reducer mental model.

Each item is a dict that mirrors the structure of `SAWItem.meta` produced by
`RolloutEngine.generate_batch()`: evaluation records only env steps, while
training batches concatenate full turns into one item.
    """
    items: List[Dict[str, Any]] = []
    for step in rollout.steps:
        if step.kind != "env":
            continue
        info = step.info or {}
        trace = step.trace
        completion_ids = trace.completion_token_ids if trace is not None else []
        comp_len = len(completion_ids)

        meta: Dict[str, Any] = {
            "rollout_id": rollout.id,
            "step_index": step.index,
            "reward": float(step.reward),
            "prev_obs": step.prev_obs if hasattr(step, "prev_obs") else "",
            "action": step.action,
            "total_reward": float(rollout.total_reward),
            "completion_length": int(comp_len),
            "truncated": bool(step.truncated),
            "terminated": bool(step.terminated),
            "step_kind": step.kind,
            "turn_id": step.turn_id,
            **(rollout.meta),
        }
        meta.update(info)
        items.append(meta)
    return items


async def run_eval(
    *,
    engine: RolloutEngine,
    requests: Sequence[RolloutRequest],
    reducers: Mapping[str, Reducer],
    max_steps: int,
    timeout_s: float | None = None,
    concurrency: int = 8,
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Generate rollouts via RolloutEngine and reduce them into metrics.

    Returns (records, reducer_stats).
    """
    reqs = list(requests)
    if not reqs:
        return ([], {})

    rollouts = await engine.generate_rollouts(
        requests=reqs,
        max_steps=max_steps,
        timeout_s=timeout_s,
        concurrency=concurrency,
    )

    records: List[Dict[str, Any]] = []
    for rollout in rollouts:
        records.extend(_eval_items_from_rollout(rollout))

    reducer_stats = apply_reducers_to_records(
        records,
        reducers,
        sample_count=float(len(records)),
        rollout_count=float(len(rollouts)),
    )
    return records, reducer_stats


def run_eval_sync(
    *,
    engine: RolloutEngine,
    requests: Sequence[RolloutRequest],
    reducers: Mapping[str, Reducer],
    max_steps: int,
    timeout_s: float | None = None,
    concurrency: int = 8,
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Synchronous wrapper around run_eval for convenience."""
    return asyncio.run(
        run_eval(
            engine=engine,
            requests=requests,
            reducers=reducers,
            max_steps=max_steps,
            timeout_s=timeout_s,
            concurrency=concurrency,
        )
    )
