"""
Eval a ReTool agent (code-interleaved) on GSM8K.

Example:
    uv run python examples/math_tool/eval_math_tool_gsm8k_vllm.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --host 127.0.0.1 --port 8000 \
        --limit 200

Requires: uv sync --extra examples
"""

from __future__ import annotations

import argparse
import asyncio
import io
import re
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Callable, Dict, List, Optional

from environments.gsm8k import GSM8KEnv
from ludic.agent import ReToolAgent
from ludic.context import FullDialog
from ludic.eval.cli import (
    add_common_eval_args,
    inference_spec_from_cli,
    maybe_start_vllm,
    write_jsonl,
)
from ludic.inference import VLLMChatClient
from ludic.interaction import SingleAgentSyncProtocol
from ludic.parsers import ParseResult
from ludic.training import (
    EnvSpec,
    ProtocolSpec,
    Reducer,
    RolloutRequest,
    apply_reducers_to_records,
)
from ludic.training.batching.rollout_engine import RolloutEngine
from ludic.types import Rollout


async def python_code_sandbox(code: str):
    """
    Simple async code sandbox for Python execution.

    In production, use a proper sandboxed environment like:
    - dockerized Python container
    - RestrictedPython library
    - Online execution API
    """
    stdout = io.StringIO()
    stderr = io.StringIO()

    try:
        # Execute with builtins but with safety restrictions
        exec_globals = {
            "__builtins__": {
                k: v
                for k, v in __builtins__.items()
                if k not in ("open", "exec", "eval", "compile", "__import__")
            },
            "print": print,
            "result": None,
        }

        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(code, exec_globals)

        # Try to extract a result if available
        output = stdout.getvalue()
        if "result" in exec_globals and exec_globals["result"] is not None:
            output = str(exec_globals["result"])
        elif not output:
            output = "(code executed with no output)"

        from ludic.agents import CodeExecutionResult

        return CodeExecutionResult(
            output=output,
            success=True,
        )
    except Exception as e:
        from ludic.agents import CodeExecutionResult

        return CodeExecutionResult(
            output="",
            success=False,
            error=str(e),
        )


def load_gsm8k(split: str, limit: int | None) -> List[dict]:
    """Load GSM8K dataset samples."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "This example requires 'datasets' package. "
            "Install with: uv sync --extra examples"
        ) from e

    ds = load_dataset("gsm8k", "main", split=split)
    samples: List[dict] = []
    for idx, row in enumerate(ds):
        samples.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "id": row.get("id", idx),
            }
        )
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError(f"No GSM8K samples loaded for split={split}")
    return samples


def gsm8k_parser(raw: str) -> ParseResult:
    """
    Extract GSM8K final numeric answer from ReTool-style output.

    - Looks for \\boxed{...}
    - Falls back to last numeric token if no box
    """
    boxed = re.search(r"\\boxed\{([^}]*)\}", raw, flags=re.DOTALL)
    if boxed:
        answer = boxed.group(1).strip()
        if answer:
            return ParseResult(action=answer, reward=0.1, obs=None)

    cleaned = raw.replace(",", "").strip()
    numeric_tokens = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", cleaned)
    if numeric_tokens:
        return ParseResult(
            action=numeric_tokens[-1].strip(),
            reward=0.0,
            obs=None,
        )

    return ParseResult(
        action=None,
        reward=-1.0,
        obs="Could not find a final answer (expected \\boxed{...}).",
    )


def _parse_error_for_metrics(record: Dict[str, Any]) -> Optional[bool]:
    if record.get("step_kind") == "agent" and record.get("action_target") == "internal":
        return None
    return record.get("parse_error")


GSM8K_REDUCERS: Dict[str, Reducer] = {
    "accuracy": Reducer(
        kind="count_true", source="correct", normalize_by="samples", as_percent=True
    ),
    "parse_error_rate": Reducer(
        kind="count_true",
        source=_parse_error_for_metrics,
        normalize_by="samples",
        as_percent=True,
    ),
    "avg_completion_tokens": Reducer(kind="mean", source="completion_length"),
    "avg_code_blocks_per_rollout": Reducer(kind="mean", source="code_blocks_total"),
    "avg_code_blocks_success_per_rollout": Reducer(
        kind="mean", source="code_blocks_success"
    ),
    "avg_code_blocks_error_per_rollout": Reducer(
        kind="mean", source="code_blocks_error"
    ),
    "code_block_error_rate": Reducer(
        kind="mean",
        source="code_block_error_rate",
        as_percent=True,
    ),
    "total_code_blocks": Reducer(kind="sum", source="code_blocks_total"),
}


def build_retool_engine(
    *,
    client: VLLMChatClient,
    model: str,
    parser: Callable[[str], ParseResult],
    env_registry: Dict[str, Callable[..., GSM8KEnv]],
    system_prompt: str | None,
    max_code_blocks: int,
) -> RolloutEngine:
    def protocol_factory() -> SingleAgentSyncProtocol:
        agent = ReToolAgent(
            client=client,
            model=model,
            ctx=FullDialog(system_prompt=system_prompt),
            parser=parser,
            code_sandbox=python_code_sandbox,
            max_code_blocks=max_code_blocks,
        )
        return SingleAgentSyncProtocol(agent=agent)

    return RolloutEngine(
        env_registry=dict(env_registry),
        protocol_registry={"single_agent": protocol_factory},
    )


def make_requests(
    samples: List[dict], args: argparse.Namespace
) -> List[RolloutRequest]:
    inf = inference_spec_from_cli(args)
    return [
        RolloutRequest(
            env=EnvSpec(
                kind="gsm8k",
                kwargs={"sample": sample, "system_prompt": args.system_prompt},
            ),
            protocol=ProtocolSpec(kind="single_agent"),
            env_seed=idx,
            sampling_seed=idx,
            inference=inf,
            num_episodes=1,
            meta={"sample_index": idx},
        )
        for idx, sample in enumerate(samples)
    ]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a ReTool agent (code-interleaved) on GSM8K."
    )
    add_common_eval_args(parser)
    parser.add_argument("--split", type=str, default="test", help="GSM8K split.")
    parser.add_argument("--limit", type=int, default=None, help="Max samples.")
    parser.add_argument(
        "--max-code-blocks", type=int, default=10, help="Max code blocks per episode."
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You are a careful math tutor. Think step-by-step. "
            "When you need to compute something, write Python code inside <code>```python\n...\n```</code> tags. "
            "The code will be executed and the result will be shown to you. "
            "Place your final numeric answer inside \\boxed{...}."
        ),
    )
    parser.add_argument(
        "--decode-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode prompt/completion token IDs into text in JSONL output.",
    )
    parser.add_argument(
        "--decode-max-tokens",
        type=int,
        default=512,
        help="Max tokens to decode per prompt/completion (0 = no limit).",
    )
    parser.set_defaults(out="math_tool_gsm8k_eval.jsonl")
    return parser


def _code_block_stats(rollout: Rollout) -> Dict[str, float]:
    """Calculate code block statistics from a rollout."""
    total_blocks = 0
    error_blocks = 0

    for step in rollout.steps:
        if step.kind != "agent":
            continue
        if step.action_target != "internal":
            continue

        total_blocks += 1
        if not step.info.get("code_success", True):
            error_blocks += 1

    success_blocks = max(total_blocks - error_blocks, 0)
    error_rate = error_blocks / total_blocks if total_blocks > 0 else 0.0

    return {
        "code_blocks_total": float(total_blocks),
        "code_blocks_success": float(success_blocks),
        "code_blocks_error": float(error_blocks),
        "code_block_error_rate": float(error_rate),
    }


def _decode_token_ids(
    tokenizer: Any,
    token_ids: List[int],
    max_tokens: Optional[int],
) -> str:
    truncated = False
    ids = list(token_ids)
    if max_tokens is not None and max_tokens > 0 and len(ids) > max_tokens:
        ids = ids[:max_tokens]
        truncated = True
    text = tokenizer.decode(ids, skip_special_tokens=False)
    if truncated:
        text = f"{text}\n...[truncated]"
    return text


def _eval_records_from_rollouts(
    rollouts: List[Rollout],
    *,
    tokenizer: Optional[Any],
    decode_max_tokens: Optional[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for rollout in rollouts:
        code_stats = _code_block_stats(rollout)
        for step in rollout.steps:
            info = step.info or {}
            trace = step.trace
            completion_ids = trace.completion_token_ids if trace is not None else []
            comp_len = len(completion_ids)
            meta: Dict[str, Any] = {
                "rollout_id": rollout.id,
                "step_index": step.index,
                "reward": float(step.reward),
                "action": step.action,
                "total_reward": float(rollout.total_reward),
                "completion_length": int(comp_len),
                "truncated": bool(step.truncated),
                "terminated": bool(step.terminated),
                "step_kind": step.kind,
                "turn_id": step.turn_id,
                **(rollout.meta),
            }
            if step.kind == "env":
                meta["prev_obs"] = step.prev_obs if hasattr(step, "prev_obs") else ""
            elif step.kind == "agent":
                meta["action_target"] = step.action_target
            meta.update(info)
            if tokenizer is not None and trace is not None:
                if trace.prompt_token_ids:
                    meta["decoded_prompt"] = _decode_token_ids(
                        tokenizer,
                        trace.prompt_token_ids,
                        decode_max_tokens,
                    )
                if trace.completion_token_ids:
                    meta["decoded_completion"] = _decode_token_ids(
                        tokenizer,
                        trace.completion_token_ids,
                        decode_max_tokens,
                    )
            meta.update(code_stats)
            records.append(meta)
    return records


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    samples = load_gsm8k(args.split, args.limit)
    print(f"Loaded {len(samples)} GSM8K samples from split '{args.split}'")

    with maybe_start_vllm(args):
        client = VLLMChatClient(
            host=args.host, port=args.port, enable_weight_updates=False
        )
        engine = build_retool_engine(
            client=client,
            model=args.model,
            parser=gsm8k_parser,
            env_registry={
                "gsm8k": lambda sample, system_prompt=None: GSM8KEnv(
                    sample=sample, system_prompt=system_prompt
                )
            },
            system_prompt=args.system_prompt,
            max_code_blocks=args.max_code_blocks,
        )
        requests = make_requests(samples, args)
        tokenizer = None
        if args.decode_tokens:
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise SystemExit(
                    "Decoding tokens requires 'transformers'. "
                    "Install with: uv sync --extra examples"
                ) from e
            tokenizer = AutoTokenizer.from_pretrained(
                args.model, trust_remote_code=True
            )
        decode_max_tokens = (
            args.decode_max_tokens if args.decode_max_tokens > 0 else None
        )

        def _fmt_metric(name: str, value: float) -> str:
            reducer = GSM8K_REDUCERS.get(name)
            if reducer is not None and reducer.as_percent:
                return f"{name}={value:.2%}"
            return f"{name}={value:.4g}"

        rollouts = asyncio.run(
            engine.generate_rollouts(
                requests=requests,
                max_steps=args.max_steps,
                timeout_s=args.timeout_s,
                concurrency=args.concurrency,
            )
        )
        records = _eval_records_from_rollouts(
            rollouts,
            tokenizer=tokenizer,
            decode_max_tokens=decode_max_tokens,
        )
        metrics = apply_reducers_to_records(
            records,
            GSM8K_REDUCERS,
            sample_count=float(len(rollouts)),
            rollout_count=float(len(rollouts)),
        )

        print("\n---- GSM8K Evaluation (ReTool) ----")
        for k, v in metrics.items():
            print(_fmt_metric(k, float(v)))

        if args.out:
            write_jsonl(args.out, records)
            print(f"Wrote {len(records)} step records to {args.out}")


if __name__ == "__main__":
    main()
