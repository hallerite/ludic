from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.distributed as dist

from ludic.distributed.interfaces import PolicyPublisher
from ludic.distributed.publisher import Rank0OnlyPublisher
from ludic.distributed.adapters.vllm import create_vllm_publisher


class _Recorder(PolicyPublisher):
    def __init__(self) -> None:
        self.calls: list[tuple[int | None, int]] = []

    def publish(self, state_dict: Mapping[str, torch.Tensor], version: Optional[int] = None) -> None:
        self.calls.append((version, len(state_dict)))


def test_rank0_only_publisher_is_lazy_and_rank_gated(monkeypatch) -> None:
    created = {"count": 0}
    recorder = _Recorder()

    def make_inner() -> PolicyPublisher:
        created["count"] += 1
        return recorder

    pub = Rank0OnlyPublisher(make_inner)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    pub.publish({"a": torch.tensor([1])}, version=7)
    assert created["count"] == 0
    assert recorder.calls == []


def test_rank0_only_publisher_constructs_on_rank0(monkeypatch) -> None:
    created = {"count": 0}
    recorder = _Recorder()

    def make_inner() -> PolicyPublisher:
        created["count"] += 1
        return recorder

    pub = Rank0OnlyPublisher(make_inner)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    pub.publish({"a": torch.tensor([1]), "b": torch.tensor([2])}, version=3)
    assert created["count"] == 1
    assert recorder.calls == [(3, 2)]


def test_create_vllm_publisher_rank0_only_is_lazy(monkeypatch) -> None:
    """
    Regression test: rank0_only wrapper should not touch client attributes on non-rank0.
    """

    class _DummyClient:
        pass

    pub = create_vllm_publisher(_DummyClient(), rank0_only=True)  # type: ignore[arg-type]

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 1)

    # Should no-op without trying to build the inner vLLM publisher.
    pub.publish({"a": torch.tensor([1])}, version=1)
