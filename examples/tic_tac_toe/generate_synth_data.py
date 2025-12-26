"""
Generate synthetic training data for Tic-Tac-Toe SFT.

This script:
1. Generates rollouts using FullDialog (model sees full thinking history)
2. Filters to keep only winning trajectories (rejection sampling)
3. Optionally rebuilds prompts to mimic TruncatedThinkingContext:
   prior assistant turns become <think>[TRUNCATED]</think><answer>, but the
   action for the current step stays intact. (--no-transform keeps full history)
4. Saves lean JSONL (minimal fields; we retokenize downstream) for use with
   OfflineBatchSource + make_sft()

The goal is to create cold-start data that teaches the model the truncation
convention BEFORE doing RL with TruncatedThinkingContext.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer

from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient, InferenceSpec, SamplingParams, ReturnSpec, HFChatTemplate
from ludic.interaction import SingleAgentSyncProtocol
from ludic.parsers import compose_parsers, think_prefix_parser, xml_tag_parser
from ludic.training import RolloutEngine, EnvSpec, ProtocolSpec, RolloutRequest
from ludic.types import Rollout

from environments.tic_tac_toe import TicTacToeEnv

# Same parser as train_tic_tac_toe.py (no per-turn success reward).
TICTACTOE_PARSER = compose_parsers(
    partial(think_prefix_parser, success_reward=0.0, error_reward=-1.0),
    xml_tag_parser("move", exact=True, success_reward=0.0, error_reward=-1.0),
)


def build_system_prompt() -> str:
    """Build system prompt matching train_tic_tac_toe.py"""
    base_prompt = TicTacToeEnv().suggested_sysprompt or ""
    return (
        base_prompt
        + "\n\nThink through the board in <think>...</think>. "
        "After </think>, output exactly one XML tag of the form <move>A1</move> and nothing else."
    )


def get_result(r: Rollout) -> str | None:
    """Get the game result from the final step's info."""
    if not r.steps:
        return None
    return r.steps[-1].info.get("result")


def is_win(r: Rollout) -> bool:
    """Check if rollout ended in a win."""
    return get_result(r) == "win"


STRICT_THINK_PATTERN = re.compile(
    r"^(\s*<think>)(.*?)(</think>\s*)(.+)$",
    flags=re.DOTALL | re.IGNORECASE,
)


def _truncate_assistant_text(text: str, placeholder: str) -> str:
    """
    Apply the same strict truncation as TruncatedThinkingContext.
    """
    m = STRICT_THINK_PATTERN.match(text)
    if not m:
        return text
    return f"{m.group(1)}{placeholder}{m.group(3)}{m.group(4)}"


def _messages_to_prompt(messages: List[dict[str, str]]) -> str:
    """
    Flatten chat messages into a lean text prompt.
    """
    role_prefix = {"system": "System", "user": "User", "assistant": "Assistant"}
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix = role_prefix.get(role, role)
        parts.append(f"{prefix}: {content}")
    return "\n\n".join(parts)


def _truncate_history_messages(
    messages: List[dict[str, str]],
    placeholder: str,
) -> List[dict[str, str]]:
    truncated: List[dict[str, str]] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            new_msg = dict(msg)
            new_msg["content"] = _truncate_assistant_text(content, placeholder)
            truncated.append(new_msg)
        else:
            truncated.append(msg)
    return truncated


def apply_prompt_format(
    rollout: Rollout,
    *,
    system_prompt: str,
    placeholder: str = "[TRUNCATED]",
    lean: bool = True,
    truncate_history: bool = True,
    min_completion_tokens: int = 0,
    max_completion_tokens: int = 0,
    stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Build per-step prompts that mirror what the agent saw:
    system prompt + full prior chat history + current user observation.
    """
    steps: List[dict[str, Any]] = []
    history: List[dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    for s in rollout.steps:
        full_messages: List[dict[str, str]] = list(history)
        # Ensure current observation is present as the latest user turn
        if (
            not full_messages
            or full_messages[-1].get("role") != "user"
            or full_messages[-1].get("content") != s.prev_obs
        ):
            full_messages.append({"role": "user", "content": s.prev_obs})

        include_step = True
        if min_completion_tokens > 0 or max_completion_tokens > 0:
            trace = s.trace
            if trace is None:
                include_step = False
                if stats is not None:
                    stats["missing_trace"] = stats.get("missing_trace", 0) + 1
            else:
                comp_len = len(trace.completion_token_ids)
                if max_completion_tokens > 0 and comp_len > max_completion_tokens:
                    include_step = False
                    if stats is not None:
                        stats["too_long"] = stats.get("too_long", 0) + 1
                if min_completion_tokens > 0 and comp_len < min_completion_tokens:
                    include_step = False
                    if stats is not None:
                        stats["too_short"] = stats.get("too_short", 0) + 1

        if include_step:
            truncated_messages = _truncate_history_messages(full_messages, placeholder)
            chat_messages = truncated_messages if truncate_history else list(full_messages)

            prompt_text = _messages_to_prompt(chat_messages)

            step_dict: Dict[str, Any] = {
                "index": s.index,
                "prev_obs": prompt_text,
                "action": s.action,
                "reward": s.reward,
                "truncated": s.truncated,
                "terminated": s.terminated,
            }
            if not lean:
                step_dict.update(
                    {
                        "next_obs": s.next_obs,
                        "info": s.info,
                        "ts_ns": s.ts_ns,
                    }
                )
            else:
                step_dict["info"] = {}

            # Store chat-format prompt/completion for downstream apply_chat_template
            info_field = step_dict.setdefault("info", {}) or {}
            info_field["chat_prompt_messages"] = chat_messages
            info_field["chat_prompt_messages_full"] = list(full_messages)
            info_field["chat_completion"] = {"role": "assistant", "content": s.action}
            step_dict["info"] = info_field
            steps.append(step_dict)
            if stats is not None:
                stats["kept_steps"] = stats.get("kept_steps", 0) + 1

        # Update full history to include this action and next observation (if any)
        history = list(full_messages)
        history.append({"role": "assistant", "content": s.action})
        if s.next_obs is not None:
            history.append({"role": "user", "content": s.next_obs})

    rollout_dict: Dict[str, Any] = {"id": rollout.id, "steps": steps}
    if not lean:
        rollout_dict.update(
            {
                "meta": rollout.meta,
                "total_reward": rollout.total_reward,
                "length": rollout.length,
                "duration_ns": rollout.duration_ns,
            }
        )
    return rollout_dict


async def generate_synth_data(args: argparse.Namespace) -> None:
    print(f"Connecting to vLLM at http://{args.host}:{args.port}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    chat_template = HFChatTemplate(tokenizer)

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
                ctx=FullDialog(system_prompt=prompt_text),  # Full dialog for generation
                parser=TICTACTOE_PARSER,
                chat_template=chat_template,
            ),
            stop_on_parse_error=True,
        )

    env_registry = {"tictactoe": lambda **kwargs: TicTacToeEnv(**kwargs)}
    protocol_registry = {"single_agent": create_protocol}

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=None,
    )

    return_spec = ReturnSpec()
    if args.min_completion_tokens > 0 or args.max_completion_tokens > 0:
        return_spec = ReturnSpec.for_eval(return_token_ids=True)
    inference = InferenceSpec(
        sampling=SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens),
        return_=return_spec,
    )

    # Generate both agent-starts and opponent-starts scenarios
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

    print(f"Generating {args.episodes} episodes with concurrency={args.concurrency}...")
    rollouts = await engine.generate_rollouts(
        requests=requests,
        max_steps=args.max_steps,
        concurrency=args.concurrency,
    )

    # Stats before filtering
    total = len(rollouts)
    results = {"win": 0, "loss": 0, "draw": 0, "other": 0}
    for r in rollouts:
        res = get_result(r)
        if res in results:
            results[res] += 1
        else:
            results["other"] += 1
    print(f"Generated {total} rollouts: {results['win']} wins, {results['loss']} losses, {results['draw']} draws")

    # Filter and transform
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    accepted = 0
    dropped_empty = 0
    stats: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rollouts:
            # Filter by result label, not reward (parser rewards can inflate totals)
            if is_win(r):
                rollout_dict = apply_prompt_format(
                    r,
                    system_prompt=prompt_text,
                    placeholder=args.placeholder,
                    lean=args.lean,
                    truncate_history=args.transform,
                    min_completion_tokens=args.min_completion_tokens,
                    max_completion_tokens=args.max_completion_tokens,
                    stats=stats,
                )
                if not rollout_dict["steps"]:
                    dropped_empty += 1
                    continue

                f.write(json.dumps(rollout_dict, ensure_ascii=False) + "\n")
                accepted += 1

    print(f"Saved {accepted} winning rollouts to: {out_path.resolve()}")
    if args.min_completion_tokens > 0 or args.max_completion_tokens > 0:
        missing_trace = stats.get("missing_trace", 0)
        too_long = stats.get("too_long", 0)
        too_short = stats.get("too_short", 0)
        kept_steps = stats.get("kept_steps", 0)
        if missing_trace:
            print(f"Skipped {missing_trace} steps missing token traces (enable return_token_ids).")
        if too_short:
            print(f"Skipped {too_short} steps with completion length < {args.min_completion_tokens}.")
        if too_long:
            print(f"Skipped {too_long} steps with completion length > {args.max_completion_tokens}.")
        print(f"Kept {kept_steps} steps after length filtering.")
        if dropped_empty:
            print(f"Skipped {dropped_empty} rollouts with no remaining steps.")
    if args.transform:
        print(f"  (transformed to TruncatedThinking format with placeholder: '{args.placeholder}')")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Tic-Tac-Toe data for SFT cold-start."
    )
    # Model/server
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    # Generation
    parser.add_argument("--episodes", type=int, default=5000, help="Total episodes to generate.")
    parser.add_argument("--max-steps", type=int, default=5, help="Max steps per episode.")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument(
        "--min-completion-tokens",
        type=int,
        default=0,
        help="Drop steps with completion length below this value (0 disables).",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=0,
        help="Drop steps with completion length above this value (0 disables).",
    )

    # Transformation
    parser.add_argument("--transform", action="store_true", default=True,
                        help="Truncate history to TruncatedThinking format (default: True)")
    parser.add_argument("--no-transform", action="store_false", dest="transform",
                        help="Keep full assistant history in prompts")
    parser.add_argument("--placeholder", default="[TRUNCATED]",
                        help="Placeholder for truncated thinking blocks")
    parser.add_argument("--lean", action="store_true", default=True,
                        help="Drop heavy metadata to keep JSONL small (default: True)")
    parser.add_argument("--no-lean", action="store_false", dest="lean",
                        help="Keep full step/meta fields")

    # Output
    parser.add_argument("--output", default="data/tictactoe_sft_train_data.jsonl")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(generate_synth_data(args))


if __name__ == "__main__":
    main()
