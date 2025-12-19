from __future__ import annotations

import queue
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypeVar

from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.types import JSON
from ludic.inference.request import InferenceSpec

from .intra_batch_control import GRPORequestStrategy, RequestStrategy

T = TypeVar("T")
QItem = TypeVar("QItem")


class RequestsExhausted(RuntimeError):
    """
    Raised when a requests_fn is asked for work but the underlying data source is empty.
    """


def make_requests_fn_from_queue(
    q: queue.Queue[QItem],
    *,
    batch_size: int,
    build_request: Callable[[QItem], RolloutRequest],
    strategy: Optional[RequestStrategy] = None,
    on_empty: Literal["raise", "return_empty"] = "raise",
) -> Callable[[], List[RolloutRequest]]:
    """
    Build a `requests_fn` that consumes items from a Queue and turns them into RolloutRequests.

    This is the simplest "curriculum" building block: a thread-safe queue of intent items.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    def _fn() -> List[RolloutRequest]:
        reqs: List[RolloutRequest] = []
        for _ in range(batch_size):
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            reqs.append(build_request(item))

        if not reqs:
            if on_empty == "return_empty":
                return []
            raise RequestsExhausted("Queue is empty; no more requests to generate.")

        if strategy is not None:
            return strategy.expand(reqs)
        return reqs

    return _fn


def make_dataset_queue_requests_fn(
    samples_q: queue.Queue[Tuple[int, T]],
    *,
    batch_size: int,
    env_kind: str,
    protocol_kind: str,
    inference: Optional[InferenceSpec] = None,
    env_kwargs_fn: Optional[Callable[[T], Dict[str, JSON]]] = None,
    protocol_kwargs: Optional[Dict[str, JSON]] = None,
    request_meta_fn: Optional[Callable[[int, T], Dict[str, JSON]]] = None,
    env_seed_fn: Optional[Callable[[int, T], int]] = None,
    sampling_seed_fn: Optional[Callable[[int, T], int]] = None,
    group_size: int = 1,
    on_empty: Literal["raise", "return_empty"] = "raise",
) -> Callable[[], List[RolloutRequest]]:
    """
    Convenience `requests_fn` builder for dataset-style training loops.

    This covers the common "single-sample env" pattern where each env instance wraps one
    example (question/problem/etc.), and the protocol is typically run with `max_steps=1`.

    This helper:
    - pops (idx, sample) tuples from a Queue
    - wraps them into RolloutRequests
    - optionally expands requests using GRPORequestStrategy(group_size)
    """
    env_kwargs_fn_final: Callable[[T], Dict[str, JSON]]
    if env_kwargs_fn is None:
        env_kwargs_fn_final = lambda sample: {"sample": sample}  # type: ignore[return-value]
    else:
        env_kwargs_fn_final = env_kwargs_fn

    protocol_kwargs_final: Dict[str, JSON] = dict(protocol_kwargs) if protocol_kwargs is not None else {}

    env_seed_fn_final: Callable[[int, T], int]
    if env_seed_fn is None:
        env_seed_fn_final = lambda idx, _sample: idx
    else:
        env_seed_fn_final = env_seed_fn

    sampling_seed_fn_final: Callable[[int, T], int]
    if sampling_seed_fn is None:
        sampling_seed_fn_final = lambda idx, _sample: idx
    else:
        sampling_seed_fn_final = sampling_seed_fn

    meta_fn = request_meta_fn

    strategy: Optional[RequestStrategy] = None
    if group_size != 1:
        strategy = GRPORequestStrategy(group_size=group_size)

    def _build(item: Tuple[int, T]) -> RolloutRequest:
        idx, sample = item
        meta: Dict[str, JSON] = {}
        if meta_fn is not None:
            meta = dict(meta_fn(idx, sample))
        return RolloutRequest(
            env=EnvSpec(kind=env_kind, kwargs=env_kwargs_fn_final(sample)),
            protocol=ProtocolSpec(kind=protocol_kind, kwargs=protocol_kwargs_final),
            num_episodes=1,
            env_seed=int(env_seed_fn_final(idx, sample)),
            sampling_seed=int(sampling_seed_fn_final(idx, sample)),
            inference=inference,
            meta=meta,
        )

    return make_requests_fn_from_queue(
        samples_q,
        batch_size=batch_size,
        build_request=_build,
        strategy=strategy,
        on_empty=on_empty,
    )


def make_dataset_sequence_requests_fn(
    samples: Sequence[T],
    *,
    batch_size: int,
    env_kind: str,
    protocol_kind: str,
    inference: Optional[InferenceSpec] = None,
    env_kwargs_fn: Optional[Callable[[T], Dict[str, JSON]]] = None,
    protocol_kwargs: Optional[Dict[str, JSON]] = None,
    request_meta_fn: Optional[Callable[[int, T], Dict[str, JSON]]] = None,
    env_seed_fn: Optional[Callable[[int, T], int]] = None,
    sampling_seed_fn: Optional[Callable[[int, T], int]] = None,
    group_size: int = 1,
    shuffle: bool = False,
    rng_seed: int = 0,
) -> Callable[[], List[RolloutRequest]]:
    """
    Dataset-style `requests_fn` over a fixed in-memory sequence of samples.

    - If shuffle=True, samples are visited in a pseudo-random order (deterministic via rng_seed).
    - Once exhausted, it loops forever (research scaffolding default); wrap your own if you want a stop condition.
    """
    import random

    if not samples:
        raise ValueError("samples must be non-empty")

    rng = random.Random(rng_seed)
    order = list(range(len(samples)))
    if shuffle:
        rng.shuffle(order)
    pos = 0

    env_kwargs_fn_final: Callable[[T], Dict[str, JSON]]
    if env_kwargs_fn is None:
        env_kwargs_fn_final = lambda sample: {"sample": sample}  # type: ignore[return-value]
    else:
        env_kwargs_fn_final = env_kwargs_fn

    protocol_kwargs_final: Dict[str, JSON] = dict(protocol_kwargs) if protocol_kwargs is not None else {}

    env_seed_fn_final: Callable[[int, T], int]
    if env_seed_fn is None:
        env_seed_fn_final = lambda idx, _sample: idx
    else:
        env_seed_fn_final = env_seed_fn

    sampling_seed_fn_final: Callable[[int, T], int]
    if sampling_seed_fn is None:
        sampling_seed_fn_final = lambda idx, _sample: idx
    else:
        sampling_seed_fn_final = sampling_seed_fn

    meta_fn = request_meta_fn

    strategy: Optional[RequestStrategy] = None
    if group_size != 1:
        strategy = GRPORequestStrategy(group_size=group_size)

    def _fn() -> List[RolloutRequest]:
        nonlocal pos, order
        reqs: List[RolloutRequest] = []
        for _ in range(batch_size):
            if pos >= len(order):
                pos = 0
                order = list(range(len(samples)))
                if shuffle:
                    rng.shuffle(order)
            idx = order[pos]
            pos += 1
            sample = samples[idx]
            meta: Dict[str, JSON] = {}
            if meta_fn is not None:
                meta = dict(meta_fn(idx, sample))
            reqs.append(
                RolloutRequest(
                    env=EnvSpec(kind=env_kind, kwargs=env_kwargs_fn_final(sample)),
                    protocol=ProtocolSpec(kind=protocol_kind, kwargs=protocol_kwargs_final),
                    num_episodes=1,
                    env_seed=int(env_seed_fn_final(idx, sample)),
                    sampling_seed=int(sampling_seed_fn_final(idx, sample)),
                    inference=inference,
                    meta=meta,
                )
            )

        if strategy is not None:
            return strategy.expand(reqs)
        return reqs

    return _fn
