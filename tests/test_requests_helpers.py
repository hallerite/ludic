from __future__ import annotations

import queue

import pytest

from ludic.training import (
    RequestsExhausted,
    make_dataset_queue_requests_fn,
    make_requests_fn_from_queue,
)
from ludic.inference import InferenceSpec, SamplingParams, ReturnSpec


def test_make_requests_fn_from_queue_empty_raises() -> None:
    q: queue.Queue[int] = queue.Queue()

    fn = make_requests_fn_from_queue(
        q,
        batch_size=2,
        build_request=lambda x: x,  # type: ignore[arg-type]
        on_empty="raise",
    )

    with pytest.raises(RequestsExhausted):
        fn()


def test_make_dataset_queue_requests_fn_basic() -> None:
    samples_q: queue.Queue[tuple[int, dict]] = queue.Queue()
    samples_q.put((7, {"question": "q1", "id": "abc"}))
    samples_q.put((8, {"question": "q2"}))

    inf = InferenceSpec(
        sampling=SamplingParams(temperature=0.5, max_tokens=17),
        return_=ReturnSpec.for_eval(return_token_ids=True),
    )

    fn = make_dataset_queue_requests_fn(
        samples_q,
        batch_size=8,
        env_kind="gsm8k",
        protocol_kind="single_agent",
        inference=inf,
        group_size=1,
        request_meta_fn=lambda idx, sample: {"sample_index": idx, "question_id": sample.get("id", idx)},
        env_seed_fn=lambda idx, _sample: idx,
        sampling_seed_fn=lambda idx, _sample: idx,
    )

    reqs = fn()
    assert len(reqs) == 2

    r0 = reqs[0]
    assert r0.env.kind == "gsm8k"
    assert r0.env.kwargs["sample"]["question"] == "q1"
    assert r0.protocol.kind == "single_agent"
    assert r0.env_seed == 7
    assert r0.meta["sample_index"] == 7
    assert r0.meta["question_id"] == "abc"

    r1 = reqs[1]
    assert r1.env_seed == 8
    assert r1.meta["question_id"] == 8


def test_make_dataset_queue_requests_fn_group_expands_sampling_seeds() -> None:
    samples_q: queue.Queue[tuple[int, dict]] = queue.Queue()
    samples_q.put((0, {"question": "q"}))

    inf = InferenceSpec(
        sampling=SamplingParams(temperature=1.0, max_tokens=8),
        return_=ReturnSpec.for_eval(return_token_ids=True),
    )

    fn = make_dataset_queue_requests_fn(
        samples_q,
        batch_size=1,
        env_kind="gsm8k",
        protocol_kind="single_agent",
        inference=inf,
        group_size=3,
        env_seed_fn=lambda idx, _sample: idx,
        sampling_seed_fn=lambda idx, _sample: 100,
    )

    reqs = fn()
    assert len(reqs) == 3
    assert {r.env_seed for r in reqs} == {0}
    assert [r.sampling_seed for r in reqs] == [100, 101, 102]
