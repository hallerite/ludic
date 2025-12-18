from __future__ import annotations

import asyncio
import inspect
from typing import Awaitable, Callable, List, Protocol, Sequence, TypeVar


class TeacherLogprobScorer(Protocol):
    """
    Abstract interface for obtaining teacher-forced chosen-token logprobs.

    Implementations may call an inference backend (e.g. vLLM/OpenAI-compatible)
    or any other service. The key contract is alignment:

    - `input_ids`, `attention_mask`, `action_mask` are the *full* [state || action]
      sequence for each sample.
    - Return value is a list of per-sample lists, each containing one float per
      token where action_mask==1, in left-to-right order.
    """

    def score_action_token_logprobs(
        self,
        *,
        input_ids: Sequence[Sequence[int]],
        attention_mask: Sequence[Sequence[int]],
        action_mask: Sequence[Sequence[int]],
    ) -> List[List[float]]:
        ...


class AsyncTeacherLogprobScorer(Protocol):
    """
    Async variant of TeacherLogprobScorer for networked backends.
    """

    async def score_action_token_logprobs(
        self,
        *,
        input_ids: Sequence[Sequence[int]],
        attention_mask: Sequence[Sequence[int]],
        action_mask: Sequence[Sequence[int]],
    ) -> List[List[float]]:
        ...


_T = TypeVar("_T")


async def maybe_await(fn: Callable[[], _T | Awaitable[_T]]) -> _T:
    value = fn()
    if inspect.isawaitable(value):
        return await value
    return value  # type: ignore[return-value]


async def score_action_token_logprobs_async(
    scorer: TeacherLogprobScorer | AsyncTeacherLogprobScorer,
    *,
    input_ids: Sequence[Sequence[int]],
    attention_mask: Sequence[Sequence[int]],
    action_mask: Sequence[Sequence[int]],
) -> List[List[float]]:
    """
    Call a teacher scorer without blocking the event loop.

    - If scorer is async, await it.
    - If scorer is sync, run it in a thread via asyncio.to_thread.
    """
    fn = getattr(scorer, "score_action_token_logprobs")
    if inspect.iscoroutinefunction(fn):
        return await fn(input_ids=input_ids, attention_mask=attention_mask, action_mask=action_mask)

    return await asyncio.to_thread(fn, input_ids=input_ids, attention_mask=attention_mask, action_mask=action_mask)
