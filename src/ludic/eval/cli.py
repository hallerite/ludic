"""
Helpers for writing small eval CLIs that run RolloutEngine rollouts.

These utilities are used by example scripts and can be used by external users.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional

from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import (
    VLLMChatClient,
    start_vllm_server,
    wait_for_vllm_health,
    ChatTemplate,
    InferenceSpec,
    SamplingParams,
    ReturnSpec,
)
from ludic.interaction import SingleAgentSyncProtocol
from ludic.parsers import ParseResult
from ludic.training.batching.rollout_engine import RolloutEngine


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--start-server", action="store_true", help="Launch a local vLLM server before eval.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=None, help="Per-call timeout.")
    parser.add_argument("--concurrency", type=int, default=64, help="Parallel episodes.")
    parser.add_argument("--max-steps", type=int, default=1, help="Max steps per episode.")
    parser.add_argument("--out", type=str, default=None, help="Output JSONL path.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="vLLM GPU memory utilization if starting a local server.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override vLLM max sequence length to fit smaller GPUs.",
    )


@contextlib.contextmanager
def maybe_start_vllm(args: argparse.Namespace) -> Iterator[None]:
    proc = None
    if getattr(args, "start_server", False):
        proc = start_vllm_server(
            args.model,
            args.host,
            args.port,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=getattr(args, "max_model_len", None),
        )
        wait_for_vllm_health(args.host, args.port, proc)
    try:
        yield None
    finally:
        if proc is not None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
                proc.wait(timeout=10)


def build_single_agent_engine(
    *,
    client: VLLMChatClient,
    model: str,
    parser: Callable[[str], ParseResult],
    env_registry: Mapping[str, Callable[..., Any]],
    chat_template: ChatTemplate,
    system_prompt: Optional[str] = None,
    stop_on_parse_error: bool = False,
    context_factory: Optional[Callable[[Optional[str]], Any]] = None,
) -> RolloutEngine:
    make_ctx = context_factory or (lambda sp: FullDialog(system_prompt=sp))

    def protocol_factory() -> SingleAgentSyncProtocol:
        agent = Agent(
            client=client,
            model=model,
            ctx=make_ctx(system_prompt),
            parser=parser,
            chat_template=chat_template,
        )
        return SingleAgentSyncProtocol(
            agent=agent,
            stop_on_parse_error=stop_on_parse_error,
        )

    return RolloutEngine(
        env_registry=dict(env_registry),
        protocol_registry={"single_agent": protocol_factory},
    )


def inference_spec_from_cli(args: argparse.Namespace) -> InferenceSpec:
    return InferenceSpec(
        sampling=SamplingParams(
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
        ),
        return_=ReturnSpec.for_eval(return_token_ids=True),
    )


def write_jsonl(path: str, records: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(dict(rec), ensure_ascii=False) + "\n")
