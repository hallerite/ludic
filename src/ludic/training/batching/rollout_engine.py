from __future__ import annotations

import asyncio
import math
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ludic.envs.env import LudicEnv
from ludic.interaction.base import InteractionProtocol
from ludic.types import Rollout, Step, TokenTrace
from ludic.inference.request import InferenceSpec

from ludic.training.types import (
    CreditAssigner,
    SAWItem,
    SAWBatch,
    ActorTokenLogps,
    SampleAttachments,
    RolloutRequest,
    ProtocolSpec,
    EnvSpec,
    SampleFilter,
)

# ---------------------------------------------------------------------------
# Factory aliases
# ---------------------------------------------------------------------------

EnvFactory = Callable[..., LudicEnv]
ProtocolFactory = Callable[..., InteractionProtocol]

EnvRegistry = Dict[str, EnvFactory]
ProtocolRegistry = Dict[str, ProtocolFactory]

_INFO_TRACE_KEYS = {
    "prompt_token_ids",
    "completion_token_ids",
    "completion_logprobs",
}

def _require_finite(value: float, *, what: str, rollout_id: str, step_index: int) -> None:
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {what} for rollout {rollout_id}, step {step_index}: {value!r}")


def _get_credit_weight(
    weights: Mapping[Tuple[str, int], float],
    *,
    rollout_id: str,
    step_index: int,
) -> float:
    key = (rollout_id, step_index)
    try:
        w_raw = weights[key]
    except KeyError as exc:
        raise KeyError(
            "CreditAssigner did not provide a weight for "
            f"(rollout_id={rollout_id!r}, step_index={step_index}). "
            "All steps must be covered."
        ) from exc
    w = float(w_raw)
    _require_finite(w, what="credit weight", rollout_id=rollout_id, step_index=step_index)
    return w


def _require_token_trace(
    step: Step,
    *,
    rollout_id: str,
    step_index: int,
) -> TokenTrace:
    trace = step.trace
    if trace is None:
        raise ValueError(
            f"Missing rollout-time token trace for rollout {rollout_id}, step {step_index}. "
            "Online RL batching requires Step.trace with prompt/completion token IDs "
            "(and optional completion_logprobs)."
        )
    return trace


def _coerce_completion_logprobs(
    completion_logprobs: object,
    *,
    completion_ids: Sequence[int],
    rollout_id: str,
    step_index: int,
) -> Optional[List[float]]:
    if completion_logprobs is None:
        return None
    if not isinstance(completion_logprobs, list) or not all(
        isinstance(v, (int, float)) for v in completion_logprobs
    ):
        raise ValueError(
            f"Invalid completion_logprobs type for rollout {rollout_id}, step {step_index}; "
            "expected List[float]."
        )
    if len(completion_logprobs) != len(completion_ids):
        raise ValueError(
            f"completion_logprobs length mismatch for rollout {rollout_id}, step {step_index} "
            f"({len(completion_logprobs)} vs {len(completion_ids)})."
        )
    return [float(v) for v in completion_logprobs]


def _base_item_meta(
    *,
    rollout: Rollout,
    step_index: int,
    reward: float,
    comp_len: int,
    prev_obs: str,
    action: str,
    truncated: bool,
    terminated: bool,
    prompt_len: int | None = None,
) -> Dict[str, Any]:
    return {
        "rollout_id": rollout.id,
        "step_index": step_index,
        "reward": reward,
        "prev_obs": prev_obs,
        "action": action,
        "total_reward": rollout.total_reward,
        "completion_length": comp_len,
        "prompt_length": prompt_len,
        "truncated": truncated,
        "terminated": terminated,
        **(rollout.meta),  # Rollout-level meta (includes episode_truncated, truncation_reason)
    }

def _strip_trace_info(info: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in info.items() if k not in _INFO_TRACE_KEYS}


def _build_saw_item_from_token_trace(
    *,
    rollout: Rollout,
    step_index: int,
    reward: float,
    weight: float,
    prev_obs: str,
    action: str,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    step_info: Mapping[str, Any],
    completion_logprobs: Optional[List[float]],
    truncated: bool,
    terminated: bool,
) -> Tuple[SAWItem, int]:
    """
    Build one SAWItem from a rollout step + rollout-time token trace.

    This is the canonical “token-in / token-out” collation path for online RL:
      - `prompt_ids` + `completion_ids` become `input_ids` with a matching
        `action_mask` that is 1 on completion tokens.
      - `completion_logprobs` (if provided by the inference client) are attached as
        typed `attachments.actor_logps`, aligned 1:1 with `completion_ids`.
      - Step metadata is stored in JSON `meta` for logging/filtering/debugging.

    Returns:
      - (SAWItem, comp_len) where comp_len is the number of completion tokens.
    """
    input_ids = list(prompt_ids) + list(completion_ids)
    attention_mask = [1] * len(input_ids)
    action_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    comp_len = len(completion_ids)

    meta = _base_item_meta(
        rollout=rollout,
        step_index=step_index,
        reward=reward,
        comp_len=comp_len,
        prev_obs=prev_obs,
        action=action,
        truncated=truncated,
        terminated=terminated,
        prompt_len=len(prompt_ids),
    )
    meta.update(_strip_trace_info(step_info))
    attachments = SampleAttachments()
    if completion_logprobs is not None:
        attachments = SampleAttachments(
            actor_logps=ActorTokenLogps(token_logps=completion_logprobs)
        )

    return (
        SAWItem(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            weight=weight,
            meta=meta,
            attachments=attachments,
        ),
        comp_len,
    )


class RolloutEngine:
    """
    Stateless rollout executor.
    
    Responsibilities:
      1. Instantiating Envs and Protocols from Requests.
      2. Executing Episodes (generate_rollouts).
      3. Collating Training Data (generate_batch).
    """

    def __init__(
        self,
        *,
        env_registry: EnvRegistry,
        protocol_registry: ProtocolRegistry,
        jsonl_path: Optional[str] = None,
    ) -> None:
        self.env_registry = dict(env_registry)
        self.protocol_registry = dict(protocol_registry)
        self.jsonl_path = jsonl_path

        if self.jsonl_path:
            Path(os.path.dirname(self.jsonl_path) or ".").mkdir(
                parents=True, exist_ok=True
            )

    # ---- registry helpers ------------------------------------------------
    def _build_env(self, spec: EnvSpec) -> LudicEnv:
        """Instantiate an Env from an EnvSpec via the env_registry."""
        try:
            factory = self.env_registry[spec.kind]
        except KeyError as exc:
            raise KeyError(f"Unknown env kind: {spec.kind!r}") from exc
        return factory(**spec.kwargs)

    def _build_protocol(self, spec: ProtocolSpec) -> InteractionProtocol:
        """Instantiate an InteractionProtocol from a ProtocolSpec via the registry."""
        try:
            factory = self.protocol_registry[spec.kind]
        except KeyError as exc:
            raise KeyError(f"Unknown protocol kind: {spec.kind!r}") from exc
        return factory(**spec.kwargs)

    # ---- internal helpers ------------------------------------------------

    async def _run_one_request(
        self,
        request: RolloutRequest,
        episode_idx: int,
        sem: asyncio.Semaphore,
        *,
        max_steps: int,
        timeout_s: Optional[float],
    ) -> List[Rollout]:
        """
        Run a single rollout for a given RolloutRequest.
        """
        async with sem:
            # 1. Create a fresh, independent protocol worker (and its agent)
            protocol = self._build_protocol(request.protocol)

            # 2. Create a fresh env
            env = self._build_env(request.env)

            # 3. Determine seeds
            run_env_seed = request.env_seed if request.env_seed is not None else episode_idx
            is_forced_env_seed = request.env_seed is not None
            run_sampling_seed = (
                request.sampling_seed if request.sampling_seed is not None else episode_idx
            )
            is_forced_sampling_seed = request.sampling_seed is not None

            inf = request.inference or InferenceSpec()

            # 4. Run the episode using the fresh protocol and env
            rollouts = await protocol.run(
                env=env,
                max_steps=max_steps,
                env_seed=run_env_seed,
                sampling_seed=run_sampling_seed,
                inference=inf,
                timeout_s=timeout_s,
            )

            # 5. Log metadata for ALL returned rollouts
            for r in rollouts:
                r.meta.setdefault("episode_idx", episode_idx)
                
                # We flatten the request metadata into the rollout metadata
                # so keys like 'policy_version' are accessible at the top level.
                if request.meta:
                    r.meta.update(request.meta)

                r.meta.setdefault("request_meta", {})
                r.meta["request_meta"].update(request.meta)
                r.meta.setdefault("engine", {})
                r.meta["engine"].update(
                    {
                        "max_steps": max_steps,
                        "timeout_s": timeout_s,
                        "env_kind": request.env.kind,
                        "protocol_kind": request.protocol.kind,
                        "env_seed": run_env_seed,
                        "sampling_seed": run_sampling_seed,
                        "forced_env_seed": is_forced_env_seed,
                        "forced_sampling_seed": is_forced_sampling_seed,
                        "return_spec": {
                            "return_token_ids": bool(inf.return_.return_token_ids),
                            "return_chosen_logprobs": bool(inf.return_.return_chosen_logprobs),
                            "top_logprobs_k": int(inf.return_.top_logprobs_k),
                        },
                    }
                )

                if self.jsonl_path:
                    self._append_jsonl(r)

            return rollouts

    def _append_jsonl(self, rollout: Rollout) -> None:
        assert self.jsonl_path is not None
        payload = {
            "id": rollout.id,
            "meta": rollout.meta,
            "steps": [
                {
                    "index": s.index,
                    "prev_obs": s.prev_obs,
                    "action": s.action,
                    "next_obs": s.next_obs,
                    "reward": s.reward,
                    "truncated": s.truncated,
                    "terminated": s.terminated,
                    "info": s.info,
                    "ts_ns": s.ts_ns,
                }
                for s in rollout.steps
            ],
            "total_reward": rollout.total_reward,
            "length": rollout.length,
            "duration_ns": rollout.duration_ns,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ---- rollout generation ----------------------------------------------

    async def generate_rollouts(
        self,
        *,
        requests: List[RolloutRequest],
        max_steps: int,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
    ) -> List[Rollout]:
        """
        Run all rollouts described by `requests` and return them.
        """
        if not requests:
            return []

        sem = asyncio.Semaphore(max(1, concurrency))
        tasks: List[asyncio.Task[List[Rollout]]] = []

        global_idx = 0
        for req in requests:
            for _ in range(req.num_episodes):
                tasks.append(
                    asyncio.create_task(
                        self._run_one_request(
                            request=req,
                            episode_idx=global_idx,
                            sem=sem,
                            max_steps=max_steps,
                            timeout_s=timeout_s,
                        )
                    )
                )
                global_idx += 1

        results = await asyncio.gather(*tasks)
        # Flatten the list of lists (one list per episode -> single flat list of rollouts)
        return [r for sublist in results for r in sublist]

    # ---- SAW batch generation --------------------------------------------

    async def generate_batch(
        self,
        *,
        requests: List[RolloutRequest],
        max_steps: int,
        credit_assigner: CreditAssigner,
        timeout_s: Optional[float] = None,
        concurrency: int = 8,
        sample_filter: Optional[SampleFilter] = None,
    ) -> SAWBatch:
        """
        High-level entrypoint for RL-style training:

        1. Generates rollouts.
        2. Computes credit (advantages/rewards).
        3. Collates into SAWItems (handling tokenization/masking).
        4. Optionally filters samples based on metadata.

        Tokenization strategy:
        - Online RL requires rollout-time token IDs:
          Step.trace must contain prompt/completion token IDs.

        Filtering:
        - If sample_filter is provided, it's applied after SAWItems are created.
        - Filter returns True to KEEP a sample, False to DROP it.
        - Use ludic.training.filters for common predicates.
        """
        rollouts = await self.generate_rollouts(
            requests=requests,
            max_steps=max_steps,
            timeout_s=timeout_s,
            concurrency=concurrency,
        )
        weights = credit_assigner.compute(rollouts)

        items_with_lengths: List[Tuple[SAWItem, int, int]] = []

        for r in rollouts:
            for step in r.steps:
                w = _get_credit_weight(weights, rollout_id=r.id, step_index=step.index)
                reward = float(step.reward)
                _require_finite(reward, what="reward", rollout_id=r.id, step_index=step.index)

                info = step.info or {}
                trace = _require_token_trace(step, rollout_id=r.id, step_index=step.index)
                prompt_ids = list(trace.prompt_token_ids)
                completion_ids = list(trace.completion_token_ids)
                prompt_len = len(prompt_ids)
                completion_logprobs = _coerce_completion_logprobs(
                    trace.completion_logprobs,
                    completion_ids=completion_ids,
                    rollout_id=r.id,
                    step_index=step.index,
                )
                require_chosen_logprobs = bool(
                    (r.meta.get("engine") or {})
                    .get("return_spec", {})
                    .get("return_chosen_logprobs", False)
                )
                if require_chosen_logprobs and completion_logprobs is None:
                    raise ValueError(
                        f"Missing completion_logprobs for rollout {r.id}, step {step.index}, "
                        "but the rollout was executed with return_spec.return_chosen_logprobs=True. "
                        "Fix your inference client to return chosen-token logprobs (e.g. ReturnSpec.for_rl())."
                    )
                item, comp_len = _build_saw_item_from_token_trace(
                    rollout=r,
                    step_index=step.index,
                    reward=reward,
                    weight=w,
                    prev_obs=step.prev_obs,
                    action=step.action,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    step_info=info,
                    completion_logprobs=completion_logprobs,
                    truncated=step.truncated,
                    terminated=step.terminated,
                )
                items_with_lengths.append((item, prompt_len, comp_len))

        # ---- Apply sample filter ------------------------------------------
        num_before_filter = len(items_with_lengths)
        if sample_filter is not None:
            items_with_lengths = [
                (item, prompt_len, comp_len)
                for (item, prompt_len, comp_len) in items_with_lengths
                if sample_filter(item)
            ]
        num_after_filter = len(items_with_lengths)
        num_filtered = num_before_filter - num_after_filter

        items: List[SAWItem] = [item for (item, _prompt_len, _comp_len) in items_with_lengths]
        prompt_lengths: List[int] = [p for (_item, p, _c) in items_with_lengths]
        completion_lengths: List[int] = [comp_len for (_item, _p, comp_len) in items_with_lengths]

        # ---- Build batch-level metadata -----------------------------------
        # Note: num_rollouts reflects total number of *agent trajectories*, not global env episodes.
        meta = {
            "num_rollouts": len(rollouts),
            "num_samples_before_filter": num_before_filter,
            "num_samples": num_after_filter,
            "num_samples_filtered": num_filtered,
            "avg_total_reward": (
                float(sum(r.total_reward for r in rollouts) / len(rollouts))
                if rollouts else 0.0
            ),
            "avg_completion_length": (
                float(sum(completion_lengths) / len(completion_lengths))
                if completion_lengths else 0.0
            ),
            "avg_prompt_length": (
                float(sum(prompt_lengths) / len(prompt_lengths))
                if prompt_lengths else 0.0
            ),
        }

        return SAWBatch(items=items, meta=meta)
