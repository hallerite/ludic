"""
Rejection sampling example: generate Tic-Tac-Toe rollouts and keep only wins.

This demonstrates using RolloutEngine for data generation with filtering,
useful for creating SFT datasets from successful trajectories.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient, InferenceSpec, SamplingParams
from ludic.interaction import SingleAgentSyncProtocol
from ludic.parsers import xml_tag_parser
from ludic.training import RolloutEngine, EnvSpec, ProtocolSpec, RolloutRequest
from ludic.types import Rollout

from environments.tic_tac_toe import TicTacToeEnv


def build_system_prompt() -> str:
    """Build system prompt from env's suggested prompt plus XML format instructions."""
    env = TicTacToeEnv()
    base = env.suggested_sysprompt or ""
    extra = """
When you choose a move, respond ONLY with a single XML tag containing the move,
for example:

    <move>A1</move>

Do not include any other text, commentary, or tags.
"""
    return (base.rstrip() + "\n\n" + extra.strip()).strip()


def rollout_to_dict(r: Rollout) -> dict[str, Any]:
    """Convert a Rollout to a JSON-serializable dict."""
    return {
        "id": r.id,
        "meta": r.meta,
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
            for s in r.steps
        ],
        "total_reward": r.total_reward,
        "length": r.length,
        "duration_ns": r.duration_ns,
    }


async def generate_filtered_data(args: argparse.Namespace) -> None:
    print(f"Connecting to vLLM at http://{args.host}:{args.port}...")

    client = VLLMChatClient(
        host=args.host,
        port=args.port,
        connection_timeout_s=300.0,
        enable_weight_updates=False,
    )

    prompt_text = build_system_prompt()

    def create_protocol():
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,
                model=args.model,
                ctx=FullDialog(system_prompt=prompt_text),
                parser=xml_tag_parser("move"),
            ),
            stop_on_parse_error=True,
        )

    env_registry = {"tictactoe": lambda **kwargs: TicTacToeEnv(**kwargs)}
    protocol_registry = {"single_agent": create_protocol}

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=None,  # Disable auto-logging; we filter manually
    )

    inference = InferenceSpec(
        sampling=SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens),
    )

    requests = [
        RolloutRequest(
            env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": True}),
            protocol=ProtocolSpec(kind="single_agent", kwargs={}),
            num_episodes=args.episodes // 2,
            inference=inference,
            meta={"setup": "agent_starts"},
        ),
        RolloutRequest(
            env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": False}),
            protocol=ProtocolSpec(kind="single_agent", kwargs={}),
            num_episodes=args.episodes - (args.episodes // 2),
            inference=inference,
            meta={"setup": "opponent_starts"},
        ),
    ]

    print(f"Running {args.episodes} episodes with concurrency={args.concurrency}...")

    rollouts = await engine.generate_rollouts(
        requests=requests,
        max_steps=args.max_steps,
        concurrency=args.concurrency,
    )

    # Rejection sampling: keep only rollouts meeting reward threshold
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    accepted_count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rollouts:
            if r.total_reward >= args.min_reward:
                f.write(json.dumps(rollout_to_dict(r), ensure_ascii=False) + "\n")
                accepted_count += 1

    print(f"Total rollouts: {len(rollouts)}")
    print(f"Accepted (reward >= {args.min_reward}): {accepted_count}")
    if rollouts:
        print(f"Rejection rate: {100 * (1 - accepted_count / len(rollouts)):.1f}%")
    print(f"Saved to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Tic-Tac-Toe rollouts via rejection sampling (keep wins only)."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=50, help="Total episodes to generate.")
    parser.add_argument("--max-steps", type=int, default=9, help="Max steps per episode.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-reward", type=float, default=1.0, help="Minimum reward to keep (1.0 = wins only).")
    parser.add_argument("--output", default="data/tictactoe_winners_only.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(generate_filtered_data(args))


if __name__ == "__main__":
    main()
