from __future__ import annotations

from typing import List, Optional

from ludic.training.types import BatchSource, SAWBatch, TeacherTokenLogps
from ludic.training.teacher import (
    TeacherLogprobScorer,
    AsyncTeacherLogprobScorer,
    score_action_token_logprobs_async,
)


async def annotate_teacher_logprobs(
    saw_batch: SAWBatch,
    *,
    teacher_scorer: TeacherLogprobScorer | AsyncTeacherLogprobScorer,
    teacher_chunk_size: Optional[int] = None,
) -> SAWBatch:
    """
    Attach per-action-token teacher-forced chosen-token logprobs to SAWItems.

    Writes:
      - item.extras includes TeacherTokenLogps(length == #action_tokens)
    """
    items = saw_batch.items
    missing_indices = [i for i, it in enumerate(items) if it.teacher_logps is None]
    if not missing_indices:
        return saw_batch

    subset = [items[i] for i in missing_indices]
    chunk_size = teacher_chunk_size or len(subset)

    for start in range(0, len(subset), chunk_size):
        chunk = subset[start : start + chunk_size]
        input_ids = [list(it.input_ids) for it in chunk]
        attention_mask = [list(it.attention_mask) for it in chunk]
        action_mask = [list(it.action_mask) for it in chunk]

        scored = await score_action_token_logprobs_async(
            teacher_scorer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
        )
        if not isinstance(scored, list) or len(scored) != len(chunk):
            raise ValueError("Teacher scorer returned an invalid batch shape.")

        for it, per_token in zip(chunk, scored):
            expected_len = int(sum(int(x) for x in it.action_mask))
            if not isinstance(per_token, list) or len(per_token) != expected_len:
                raise ValueError(
                    "Teacher scorer length mismatch: expected "
                    f"{expected_len} action-token logprobs, got {len(per_token) if isinstance(per_token, list) else type(per_token)}."
                )
            values = [float(v) for v in per_token]
            it.add_extra(TeacherTokenLogps(token_logps=values))

    return saw_batch


class TeacherAnnotatedBatchSource(BatchSource):
    """
    BatchSource wrapper that annotates each batch with teacher logprobs.

    This is the canonical placement for networked teacher calls because
    BatchSource.next_batch() is async and runs upstream of the Trainer.
    """

    def __init__(
        self,
        inner: BatchSource,
        *,
        teacher_scorer: TeacherLogprobScorer | AsyncTeacherLogprobScorer,
        teacher_chunk_size: Optional[int] = None,
    ) -> None:
        self._inner = inner
        self._teacher_scorer = teacher_scorer
        self._teacher_chunk_size = teacher_chunk_size

    async def next_batch(self) -> SAWBatch:
        saw_batch = await self._inner.next_batch()
        return await annotate_teacher_logprobs(
            saw_batch,
            teacher_scorer=self._teacher_scorer,
            teacher_chunk_size=self._teacher_chunk_size,
        )
