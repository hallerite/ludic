from __future__ import annotations

import math
from typing import Sequence, List

import pytest

from ludic.training.batching.teacher_annotated import TeacherAnnotatedBatchSource
from ludic.training.teacher import TeacherLogprobScorer
from ludic.training.types import BatchSource, SAWBatch, SAWItem, TeacherTokenLogps


class DummyBatchSource(BatchSource):
    async def next_batch(self) -> SAWBatch:
        item = SAWItem(
            input_ids=[0, 1, 2],
            attention_mask=[1, 1, 1],
            action_mask=[0, 1, 1],
            weight=1.0,
            meta={},
        )
        return SAWBatch(items=[item], meta={})


class DummyTeacherScorer(TeacherLogprobScorer):
    def score_action_token_logprobs(
        self,
        *,
        input_ids: Sequence[Sequence[int]],
        attention_mask: Sequence[Sequence[int]],
        action_mask: Sequence[Sequence[int]],
    ) -> List[List[float]]:
        per_token = -math.log(3.0)
        out: List[List[float]] = []
        for am in action_mask:
            action_len = sum(1 for x in am if int(x) == 1)
            out.append([per_token] * action_len)
        return out


@pytest.mark.asyncio
async def test_teacher_annotated_batch_source_adds_teacher_logprobs():
    base = DummyBatchSource()
    annotated = TeacherAnnotatedBatchSource(base, teacher_scorer=DummyTeacherScorer(), teacher_chunk_size=8)
    batch = await annotated.next_batch()
    assert len(batch.items) == 1
    it = batch.items[0]
    assert it.attachments.teacher_logps == TeacherTokenLogps(token_logps=[-math.log(3.0), -math.log(3.0)])
    assert "teacher_token_logprobs" not in it.meta
