"""
Eval a vLLM-served model on GSM8K.


Example:
    uv run python examples/gsm8k/eval_gsm8k_vllm.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --host 127.0.0.1 --port 8000 \
        --limit 200

Requires: uv sync --extra examples
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List

from transformers import AutoTokenizer

from ludic.inference import VLLMChatClient
from ludic.inference import HFChatTemplate
from ludic.parsers import ParseResult, think_prefix_parser
from ludic.eval.core import run_eval_sync
from ludic.training import (
    EnvSpec,
    ProtocolSpec,
    Reducer,
    RolloutRequest,
)
from environments.gsm8k import GSM8KEnv
from ludic.eval.cli import (
    add_common_eval_args,
    build_single_agent_engine,
    maybe_start_vllm,
    inference_spec_from_cli,
    write_jsonl,
)


def load_gsm8k(split: str, limit: int | None) -> List[dict]:
    """Load GSM8K dataset samples."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "This example requires the 'datasets' package. "
            "Install with: uv sync --extra examples"
        ) from e

    ds = load_dataset("gsm8k", "main", split=split)
    samples: List[dict] = []
    for idx, row in enumerate(ds):
        samples.append({
            "question": row["question"],
            "answer": row["answer"],
            "id": row.get("id", idx),
        })
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError(f"No GSM8K samples loaded for split={split}")
    return samples


def gsm8k_parser(raw: str) -> ParseResult:
    """
    Extract GSM8K final numeric answer.

    - Optionally strips <think>...</think> prefix
    - Looks for \\boxed{...} or #### delimiter
    - Falls back to last numeric token
    """
    cot = think_prefix_parser(raw)
    text = cot.action if cot.action is not None else raw.strip()

    # Prefer explicit answer delimiters
    boxed = re.search(r"\\boxed\{([^}]*)\}", text, flags=re.DOTALL)
    if boxed:
        answer = boxed.group(1).strip()
        if answer:
            return ParseResult(action=answer, reward=cot.reward, obs=None)

    if "####" in text:
        answer = text.split("####")[-1].strip()
        if answer:
            return ParseResult(action=answer, reward=cot.reward, obs=None)

    # Fallback: last numeric token
    cleaned = text.replace(",", "").strip()
    numeric_tokens = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", cleaned)
    if numeric_tokens:
        return ParseResult(action=numeric_tokens[-1].strip(), reward=cot.reward, obs=None)

    return ParseResult(
        action=None,
        reward=-1.0,
        obs="Could not find a final answer (expected \\boxed{...} or '#### ...').",
    )


GSM8K_REDUCERS: Dict[str, Reducer] = {
    "accuracy": Reducer(kind="count_true", source="correct", normalize_by="samples", as_percent=True),
    "parse_error_rate": Reducer(kind="count_true", source="parse_error", normalize_by="samples", as_percent=True),
    "avg_completion_tokens": Reducer(kind="mean", source="completion_length"),
}


def make_requests(samples: List[dict], args: argparse.Namespace) -> List[RolloutRequest]:
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
    parser = argparse.ArgumentParser(description="Evaluate a model on GSM8K math problems.")
    add_common_eval_args(parser)
    parser.add_argument("--split", type=str, default="test", help="GSM8K split.")
    parser.add_argument("--limit", type=int, default=None, help="Max samples.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a careful math tutor. Think in <think></think> and put your final numeric answer after '####'.",
    )
    parser.set_defaults(out="gsm8k_eval.jsonl")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    samples = load_gsm8k(args.split, args.limit)
    print(f"Loaded {len(samples)} GSM8K samples from split '{args.split}'")

    with maybe_start_vllm(args):
        client = VLLMChatClient(host=args.host, port=args.port, enable_weight_updates=False)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        chat_template = HFChatTemplate(tokenizer)
        engine = build_single_agent_engine(
            client=client,
            model=args.model,
            parser=gsm8k_parser,
            env_registry={
                "gsm8k": lambda sample, system_prompt=None: GSM8KEnv(sample=sample, system_prompt=system_prompt)
            },
            chat_template=chat_template,
            system_prompt=args.system_prompt,
        )
        requests = make_requests(samples, args)

        def _fmt_metric(name: str, value: float) -> str:
            reducer = GSM8K_REDUCERS.get(name)
            if reducer is not None and reducer.as_percent:
                return f"{name}={value:.2%}"
            return f"{name}={value:.4g}"

        records, metrics = run_eval_sync(
            engine=engine,
            requests=requests,
            reducers=GSM8K_REDUCERS,
            max_steps=args.max_steps,
            timeout_s=args.timeout_s,
            concurrency=args.concurrency,
        )

        print("\n---- GSM8K Evaluation ----")
        for k, v in metrics.items():
            print(_fmt_metric(k, float(v)))

        if args.out:
            write_jsonl(args.out, records)
            print(f"Wrote {len(records)} step records to {args.out}")


if __name__ == "__main__":
    main()
