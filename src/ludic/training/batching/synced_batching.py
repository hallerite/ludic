from __future__ import annotations
from typing import Callable, List, Optional

from ludic.training.types import (
    BatchSource,
    SAWBatch,
    RolloutRequest,
    CreditAssigner,
    SampleFilter,
)
from .rollout_engine import RolloutEngine


class RolloutBatchSource(BatchSource):
    """
    Synchronous BatchSource.
    It holds the Engine and blocks the Trainer while generating data.

    Note: RolloutEngine now concatenates each agent turn into a single training
    sample (one SAWItem per turn), rather than emitting per-step samples.
    """

    def __init__(
        self,
        *,
        orchestrator: RolloutEngine,
        credit_assigner: CreditAssigner,
        requests_fn: Callable[[], List[RolloutRequest]],
        max_steps: int,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
        sample_filter: Optional[SampleFilter] = None,
    ) -> None:
        """
        Args:
            orchestrator: RolloutEngine for generating rollouts.
            credit_assigner: Computes per-step weights.
            requests_fn: Callable that returns rollout requests for each batch.
            max_steps: Maximum steps per episode.
            timeout_s: Timeout for inference calls.
            concurrency: Number of concurrent episodes.
            sample_filter: Optional filter function to drop samples based on metadata.
                Returns True to KEEP a sample, False to DROP it.
                Use ludic.training.filters for common predicates.
        """
        self._engine = orchestrator
        self._credit_assigner = credit_assigner
        self._requests_fn = requests_fn
        self._max_steps = max_steps
        self._timeout_s = timeout_s
        self._concurrency = concurrency
        self._sample_filter = sample_filter

    async def next_batch(self) -> SAWBatch:
        """
        Pull requests -> Generate (blocking) -> Return Batch.
        """
        requests = self._requests_fn()
        return await self._engine.generate_batch(
            requests=requests,
            max_steps=self._max_steps,
            credit_assigner=self._credit_assigner,
            timeout_s=self._timeout_s,
            concurrency=self._concurrency,
            sample_filter=self._sample_filter,
        )
