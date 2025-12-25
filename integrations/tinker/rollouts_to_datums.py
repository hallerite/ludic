from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import torch
import tinker

from ludic.types import Rollout, Step, TokenTrace
from ludic.training.types import CreditAssigner, RolloutStepKey


def _is_prefix(prefix: Sequence[int], seq: Sequence[int]) -> bool:
    return len(prefix) <= len(seq) and list(seq[: len(prefix)]) == list(prefix)


def _require_trace(step: Step, *, rollout_id: str) -> TokenTrace:
    if step.trace is None:
        raise ValueError(f"Missing TokenTrace for rollout {rollout_id}, step {step.index}.")
    return step.trace


def _coerce_logprobs(
    completion_logprobs: Optional[List[float]],
    completion_ids: Sequence[int],
    *,
    rollout_id: str,
    step_index: int,
    require_logprobs: bool,
) -> Optional[List[float]]:
    if completion_logprobs is None:
        if require_logprobs:
            raise ValueError(
                f"Missing completion_logprobs for rollout {rollout_id}, step {step_index}."
            )
        return None
    if len(completion_logprobs) != len(completion_ids):
        raise ValueError(
            "completion_logprobs length mismatch for rollout "
            f"{rollout_id}, step {step_index} "
            f"({len(completion_logprobs)} vs {len(completion_ids)})."
        )
    return [float(v) for v in completion_logprobs]


@dataclass
class _SequenceAccumulator:
    full_sequence: List[int]
    token_logprobs: List[float]
    token_advantages: List[float]

    @classmethod
    def empty(cls) -> "_SequenceAccumulator":
        return cls(full_sequence=[], token_logprobs=[], token_advantages=[])

    def extend_prompt(self, prompt_tokens: Sequence[int]) -> None:
        self.full_sequence.extend(prompt_tokens)
        self.token_logprobs.extend([0.0] * len(prompt_tokens))
        self.token_advantages.extend([0.0] * len(prompt_tokens))

    def extend_completion(
        self, completion_tokens: Sequence[int], completion_logprobs: Sequence[float], weight: float
    ) -> None:
        self.full_sequence.extend(completion_tokens)
        self.token_logprobs.extend([float(v) for v in completion_logprobs])
        self.token_advantages.extend([float(weight)] * len(completion_tokens))

    def to_datum(self) -> Optional[tinker.Datum]:
        if len(self.full_sequence) < 2:
            return None
        input_tokens = self.full_sequence[:-1]
        target_tokens = self.full_sequence[1:]
        logprobs = self.token_logprobs[1:]
        advantages = self.token_advantages[1:]

        if not (len(input_tokens) == len(target_tokens) == len(logprobs) == len(advantages)):
            raise ValueError(
                "Token alignment mismatch when building Datum: "
                f"{len(input_tokens)} input, {len(target_tokens)} target, "
                f"{len(logprobs)} logprobs, {len(advantages)} advantages."
            )

        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints(list(input_tokens)),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(
                    torch.tensor(target_tokens, dtype=torch.int64)
                ),
                "logprobs": tinker.TensorData.from_torch(
                    torch.tensor(logprobs, dtype=torch.float32)
                ),
                "advantages": tinker.TensorData.from_torch(
                    torch.tensor(advantages, dtype=torch.float32)
                ),
            },
        )


def rollouts_to_datums(
    rollouts: Iterable[Rollout],
    credit_assigner: CreditAssigner,
    *,
    require_logprobs: bool = True,
    step_filter: Optional[Callable[[Step], bool]] = None,
) -> List[tinker.Datum]:
    """
    Convert Ludic rollouts into Tinker Datum objects.

    This performs prefix-merge behavior: if a step's prompt is an extension of
    the previous prompt+completion sequence, we append it to the current datum.
    Otherwise, we flush the current datum and start a new one.
    """
    rollouts_list = list(rollouts)
    weights = credit_assigner.compute(rollouts_list)

    datums: list[tinker.Datum] = []
    for rollout in rollouts_list:
        acc = _SequenceAccumulator.empty()
        for step in rollout.steps:
            if step_filter is not None and not step_filter(step):
                datum = acc.to_datum()
                if datum is not None:
                    datums.append(datum)
                acc = _SequenceAccumulator.empty()
                continue

            weight = weights.get((rollout.id, step.index))
            if weight is None:
                raise KeyError(
                    "CreditAssigner missing weight for "
                    f"rollout {rollout.id}, step {step.index}."
                )

            trace = _require_trace(step, rollout_id=rollout.id)
            prompt_ids = list(trace.prompt_token_ids)
            completion_ids = list(trace.completion_token_ids)
            completion_logprobs = _coerce_logprobs(
                trace.completion_logprobs,
                completion_ids,
                rollout_id=rollout.id,
                step_index=step.index,
                require_logprobs=require_logprobs,
            )

            if completion_logprobs is None:
                completion_logprobs = [0.0] * len(completion_ids)

            if not acc.full_sequence:
                acc.extend_prompt(prompt_ids)
            elif _is_prefix(acc.full_sequence, prompt_ids):
                delta_prompt = prompt_ids[len(acc.full_sequence) :]
                if delta_prompt:
                    acc.extend_prompt(delta_prompt)
            else:
                datum = acc.to_datum()
                if datum is not None:
                    datums.append(datum)
                acc = _SequenceAccumulator.empty()
                acc.extend_prompt(prompt_ids)

            acc.extend_completion(completion_ids, completion_logprobs, float(weight))

        datum = acc.to_datum()
        if datum is not None:
            datums.append(datum)

    return datums
