from __future__ import annotations

from typing import Dict

from ludic.training.loggers import TeeLogger


class _StubLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[int, Dict[str, float]]] = []

    def log(self, step: int, stats: Dict[str, float]) -> None:
        self.calls.append((step, stats))


class _FailingLogger:
    def log(self, step: int, stats: Dict[str, float]) -> None:
        raise RuntimeError("boom")


def test_tee_logger_forwards_and_survives_failures() -> None:
    ok = _StubLogger()
    bad = _FailingLogger()
    tee = TeeLogger(bad, ok)

    tee.log(3, {"train/loss": 1.0})

    assert ok.calls == [(3, {"train/loss": 1.0})]
