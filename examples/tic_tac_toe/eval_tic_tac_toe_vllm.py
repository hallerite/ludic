"""
Eval a vLLM-served model on Tic-Tac-Toe.

Example:
    uv run python examples/tic_tac_toe/eval_tic_tac_toe_vllm.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --host 127.0.0.1 --port 8000 \
        --episodes 200
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from transformers import AutoTokenizer

from ludic.inference import VLLMChatClient
from ludic.inference import HFChatTemplate
from ludic.parsers import compose_parsers, think_prefix_parser, xml_tag_parser
from ludic.eval.core import run_eval_sync
from ludic.training import (
    EnvSpec,
    ProtocolSpec,
    Reducer,
    RolloutRequest,
)
from environments.tic_tac_toe import TicTacToeEnv
from ludic.eval.cli import (
    add_common_eval_args,
    build_single_agent_engine,
    maybe_start_vllm,
    inference_spec_from_cli,
    write_jsonl,
)
from ludic.context import FullDialog, TruncatedThinkingContext


TICTACTOE_REDUCERS: Dict[str, Reducer] = {
    "win_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "win", normalize_by="rollouts", as_percent=True),
    "loss_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "loss", normalize_by="rollouts", as_percent=True),
    "draw_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "draw", normalize_by="rollouts", as_percent=True),
    "gto_move_rate": Reducer(kind="count_true", source="gto_action", normalize_by="samples", as_percent=True),
    "illegal_rate": Reducer(kind="count_true", source="illegal_move", normalize_by="rollouts", as_percent=True),
    "parse_error_rate": Reducer(kind="count_true", source="parse_error", normalize_by="rollouts", as_percent=True),
    "truncated_rate": Reducer(kind="count_true", source="truncated", normalize_by="rollouts", as_percent=True),
    "avg_completion_tokens": Reducer(kind="mean", source="completion_length"),
}


def make_requests(episodes: int, args: argparse.Namespace) -> List[RolloutRequest]:
    inf = inference_spec_from_cli(args)
    start_flags = [True] * ((episodes + 1) // 2) + [False] * (episodes // 2)
    requests: List[RolloutRequest] = []
    for seed, agent_starts in enumerate(start_flags):
        requests.append(
            RolloutRequest(
                env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": agent_starts}),
                protocol=ProtocolSpec(kind="single_agent"),
                env_seed=seed,
                sampling_seed=seed,
                inference=inf,
                num_episodes=1,
                meta={"seed": seed, "agent_starts": agent_starts},
            )
        )
    return requests


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a model on Tic-Tac-Toe.")
    add_common_eval_args(parser)
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes.")
    parser.set_defaults(temperature=0.6, max_tokens=250, max_steps=5, out="tictactoe_eval.jsonl")
    parser.add_argument(
        "--ctx",
        choices=["full", "truncated"],
        default="full",
        help="Context strategy: 'full' (FullDialog) or 'truncated' (TruncatedThinkingContext).",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    ctx_choice = args.ctx

    base_env = TicTacToeEnv(agent_starts=True)
    system_prompt = (
        (base_env.suggested_sysprompt or "")
        + "\n\nThink through the board in <think>...</think>. "
        "After </think>, output exactly one XML tag of the form <move>A1</move> and nothing else."
    )

    with maybe_start_vllm(args):
        client = VLLMChatClient(host=args.host, port=args.port, enable_weight_updates=False)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        chat_template = HFChatTemplate(tokenizer)
        ctx_factory = (
            (lambda sp: TruncatedThinkingContext(system_prompt=sp))
            if ctx_choice == "truncated"
            else (lambda sp: FullDialog(system_prompt=sp))
        )
        engine = build_single_agent_engine(
            client=client,
            model=args.model,
            parser=compose_parsers(
                lambda raw: think_prefix_parser(raw, success_reward=0.0, error_reward=-1.0),
                lambda _: xml_tag_parser("move", exact=True, success_reward=0.0, error_reward=-1.0)(_),
            ),
            env_registry={"tictactoe": lambda agent_starts=True: TicTacToeEnv(agent_starts=agent_starts)},
            chat_template=chat_template,
            system_prompt=system_prompt,
            context_factory=ctx_factory,
            stop_on_parse_error=True,
        )
        requests = make_requests(args.episodes, args)

        records, metrics = run_eval_sync(
            engine=engine,
            requests=requests,
            reducers=TICTACTOE_REDUCERS,
            max_steps=args.max_steps,
            timeout_s=args.timeout_s,
            concurrency=args.concurrency,
        )

        print("\n---- Tic-Tac-Toe Evaluation ----")
        for k, v in metrics.items():
            reducer = TICTACTOE_REDUCERS.get(k)
            if reducer is not None and reducer.as_percent:
                print(f"{k}={float(v):.2%}")
            else:
                print(f"{k}={float(v):.4g}")

        if args.out:
            write_jsonl(args.out, records)
            print(f"Wrote {len(records)} step records to {args.out}")


if __name__ == "__main__":
    main()
