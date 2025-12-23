from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from ludic.training.batching.rollout_engine import RolloutEngine
from ludic.training.stats import Reducer
from ludic.training.types import RolloutRequest

from .core import run_eval


EvalResult = Tuple[List[Dict[str, object]], Dict[str, float]]


class Evaluator(Protocol):
    async def eval(
        self,
        *,
        requests: Optional[Sequence[RolloutRequest]] = None,
        reducers: Optional[Mapping[str, Reducer]] = None,
        max_steps: Optional[int] = None,
        timeout_s: Optional[float] = None,
        concurrency: Optional[int] = None,
    ) -> EvalResult:
        ...


@dataclass(frozen=True)
class EngineEvaluator:
    """
    Default evaluator implementation backed by RolloutEngine + reducers.

    This is a thin adapter: it turns the eval configuration into a single object
    so Trainer doesn't need to know about RolloutEngine registries.
    """

    engine: RolloutEngine
    requests_fn: Callable[[], List[RolloutRequest]]
    reducers: Mapping[str, Reducer]
    max_steps: int = 1
    timeout_s: float | None = None
    concurrency: int = 32

    async def eval(
        self,
        *,
        requests: Optional[Sequence[RolloutRequest]] = None,
        reducers: Optional[Mapping[str, Reducer]] = None,
        max_steps: Optional[int] = None,
        timeout_s: Optional[float] = None,
        concurrency: Optional[int] = None,
    ) -> EvalResult:
        reqs = list(requests) if requests is not None else list(self.requests_fn())
        if not reqs:
            return ([], {})

        used_reducers = reducers if reducers is not None else self.reducers
        used_max_steps = int(max_steps) if max_steps is not None else int(self.max_steps)
        used_timeout = timeout_s if timeout_s is not None else self.timeout_s
        used_concurrency = int(concurrency) if concurrency is not None else int(self.concurrency)

        return await run_eval(
            engine=self.engine,
            requests=reqs,
            reducers=used_reducers,
            max_steps=used_max_steps,
            timeout_s=used_timeout,
            concurrency=used_concurrency,
        )

