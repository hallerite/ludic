"""
Tic-Tac-Toe training scaffold using Ludic rollouts + Tinker training backend.

This mirrors examples/tic_tac_toe/train_tic_tac_toe.py but replaces the Ludic
Trainer with Tinker TrainingClient. Ludic still owns envs, protocols, parsers,
and rollout generation.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import time
from typing import Dict, List, Tuple

import torch
import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils import ml_log

from environments.tic_tac_toe import TicTacToeEnv
from integrations.tinker import TinkerChatClient, rollouts_to_datums
from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.eval import run_eval
from ludic.inference import InferenceSpec, SamplingParams, ReturnSpec
from ludic.interaction import SingleAgentSyncProtocol
from ludic.parsers import ParseResult
from ludic.training import (
    EnvSpec,
    ProtocolSpec,
    RolloutEngine,
    RolloutRequest,
    GRPORequestStrategy,
    Reducer,
)
from ludic.training.credit_assignment import GroupNormalizedReturn


_MOVE_TAG_RE = re.compile(r"<move>(.*?)</move>\s*$", flags=re.IGNORECASE | re.DOTALL)


def parse_move_tag_at_end(raw: str) -> ParseResult:
    match = _MOVE_TAG_RE.search(raw.strip())
    if not match:
        return ParseResult(
            action=None,
            reward=-1.0,
            obs="Invalid action format: end with <move>...</move>.",
        )
    move = match.group(1).strip()
    if not move:
        return ParseResult(
            action=None,
            reward=-1.0,
            obs="Invalid action format: empty <move> tag.",
        )
    return ParseResult(action=move, reward=0.0, obs=None)


TICTACTOE_PARSER = parse_move_tag_at_end


def make_start_flags(agent_starts_as: str, count: int) -> List[bool]:
    if agent_starts_as == "x":
        return [True] * count
    if agent_starts_as == "o":
        return [False] * count
    return [True] * ((count + 1) // 2) + [False] * (count // 2)


def build_requests_fn(
    rng: torch.Generator,
    num_requests: int,
    inference: InferenceSpec,
    agent_starts_as: str,
) -> List[RolloutRequest]:
    next_mixed_start = None

    def _next_mixed_start() -> bool:
        nonlocal next_mixed_start
        if next_mixed_start is None:
            next_mixed_start = bool(torch.randint(0, 2, (1,), generator=rng).item())
        start = next_mixed_start
        next_mixed_start = not next_mixed_start
        return start

    reqs: List[RolloutRequest] = []
    if agent_starts_as == "mixed" and num_requests == 1:
        start_flags = [_next_mixed_start()]
    else:
        start_flags = make_start_flags(agent_starts_as, num_requests)
    perm = torch.randperm(len(start_flags), generator=rng).tolist()
    for idx in perm:
        agent_starts = bool(start_flags[idx])
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        reqs.append(
            RolloutRequest(
                env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": agent_starts}),
                protocol=ProtocolSpec(kind="single_agent", kwargs={}),
                num_episodes=1,
                env_seed=int(seed),
                sampling_seed=int(seed),
                inference=inference,
                meta={"agent_starts": agent_starts},
            )
        )
    return reqs


def _count_results(rollouts) -> Tuple[int, int, int]:
    win = loss = draw = 0
    for rollout in rollouts:
        if not rollout.steps:
            continue
        result = rollout.steps[-1].info.get("result")
        if result == "win":
            win += 1
        elif result == "loss":
            loss += 1
        elif result == "draw":
            draw += 1
    return win, loss, draw


def _count_flag(rollouts, key: str) -> int:
    count = 0
    for rollout in rollouts:
        for step in rollout.steps:
            if step.info.get(key):
                count += 1
                break
    return count


def _gto_move_rate(rollouts) -> float:
    gto = 0
    legal = 0
    for rollout in rollouts:
        for step in rollout.steps:
            info = step.info
            if info.get("illegal_move") or info.get("parse_error"):
                continue
            if "gto_action" in info:
                legal += 1
                if info.get("gto_action"):
                    gto += 1
    return float(gto) / float(legal) if legal > 0 else 0.0


async def run_training(args: argparse.Namespace) -> None:
    # Tinker setup
    service_client = tinker.ServiceClient(base_url=args.base_url)
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.model,
        rank=args.lora_rank,
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    tokenizer = training_client.get_tokenizer()

    tinker_client = TinkerChatClient(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        policy_version="0",
    )

    # Registries
    env_registry = {"tictactoe": lambda agent_starts=True: TicTacToeEnv(agent_starts=agent_starts)}

    base_prompt = TicTacToeEnv().suggested_sysprompt or ""
    system_prompt = (
        base_prompt
        + "\n\nChoose the best move. You may include brief reasoning, but your response must end "
        "with exactly one XML tag of the form <move>A1</move> (A1..C3). "
        "Example: The center is strongest. <move>B2</move>"
    )

    def protocol_factory():
        ctx = FullDialog(system_prompt=system_prompt)
        parser = TICTACTOE_PARSER
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=tinker_client,
                model=args.model,
                ctx=ctx,
                parser=parser,
            ),
            stop_on_parse_error=True,
        )

    protocol_registry = {"single_agent": protocol_factory}

    rollout_log_path = os.path.abspath(args.rollout_log)
    os.makedirs(os.path.dirname(rollout_log_path) or ".", exist_ok=True)
    open(rollout_log_path, "a", encoding="utf-8").close()

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=rollout_log_path,
    )

    train_inference = InferenceSpec(
        sampling=SamplingParams(
            temperature=args.train_temperature,
            max_tokens=args.max_completion_tokens,
        ),
        return_=ReturnSpec.for_rl(top_logprobs_k=1),
    )
    eval_inference = InferenceSpec(
        sampling=SamplingParams(
            temperature=args.eval_temperature,
            max_tokens=args.max_completion_tokens,
        ),
        return_=ReturnSpec.for_eval(return_token_ids=True),
    )

    credit_assigner = GroupNormalizedReturn(
        group_size=args.group_size,
        normalize_adv=args.normalize_adv,
        positive_only=args.positive_only,
    )

    adam_params = tinker.AdamParams(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    wandb_project = args.wandb_project
    if not wandb_project:
        env_project = os.environ.get("WANDB_PROJECT")
        if env_project:
            wandb_project = env_project
        elif os.environ.get("WANDB_API_KEY"):
            wandb_project = "Ludic"
    logger = ml_log.setup_logging(
        log_dir=args.log_path,
        wandb_project=wandb_project,
        wandb_name=args.wandb_name,
        config=vars(args),
    )

    eval_reducers = {
        "eval/win_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "win",
            normalize_by="rollouts",
            as_percent=True,
        ),
        "eval/loss_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "loss",
            normalize_by="rollouts",
            as_percent=True,
        ),
        "eval/draw_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "draw",
            normalize_by="rollouts",
            as_percent=True,
        ),
        "eval/gto_move_rate": Reducer(
            kind="mean",
            source=lambda item: (
                None
                if item.get("illegal_move") or item.get("parse_error")
                else (1.0 if item.get("gto_action") else 0.0)
            ),
            as_percent=True,
        ),
        "eval/illegal_rate": Reducer(
            kind="count_true",
            source="illegal_move",
            normalize_by="rollouts",
            as_percent=True,
        ),
        "eval/parse_error_rate": Reducer(
            kind="count_true",
            source="parse_error",
            normalize_by="rollouts",
            as_percent=True,
        ),
        "eval/truncated_rate": Reducer(
            kind="count_true",
            source="truncated",
            normalize_by="rollouts",
            as_percent=True,
        ),
        "eval/avg_completion_tokens": Reducer(
            kind="mean",
            source="completion_length",
        ),
    }

    def eval_requests() -> List[RolloutRequest]:
        start_flags = make_start_flags(args.agent_starts_as, args.eval_episodes)
        return [
            RolloutRequest(
                env=EnvSpec(kind="tictactoe", kwargs={"agent_starts": agent_starts}),
                protocol=ProtocolSpec(kind="single_agent", kwargs={}),
                num_episodes=1,
                env_seed=int(seed),
                sampling_seed=int(seed),
                inference=eval_inference,
                meta={"eval_seed": seed, "agent_starts": agent_starts},
            )
            for seed, agent_starts in enumerate(start_flags)
        ]

    async def run_eval_if_needed(step: int) -> None:
        if args.eval_episodes <= 0:
            return
        records, stats = await run_eval(
            engine=engine,
            requests=eval_requests(),
            reducers=eval_reducers,
            max_steps=args.max_steps_per_episode,
            concurrency=args.eval_concurrency,
        )
        _ = records
        logger.log_metrics(stats, step=step)

    eval_ran_before = False
    if args.eval_before_start and args.eval_episodes > 0:
        await run_eval_if_needed(step=0)
        eval_ran_before = True

    rng = torch.Generator()
    rng.manual_seed(args.seed if args.seed is not None else 0)

    step = 0
    while True:
        if args.train_steps > 0 and step >= args.train_steps:
            break

        base_requests = args.rollouts_per_update // args.group_size
        base_req = build_requests_fn(
            rng,
            base_requests,
            train_inference,
            args.agent_starts_as,
        )
        requests = GRPORequestStrategy(group_size=args.group_size).expand(base_req)

        t_start = time.time()
        rollouts = await engine.generate_rollouts(
            requests=requests,
            max_steps=args.max_steps_per_episode,
            concurrency=args.concurrency,
        )

        datums = rollouts_to_datums(
            rollouts,
            credit_assigner=credit_assigner,
            require_logprobs=True,
        )
        if not datums:
            step += 1
            continue

        fwd_bwd_future = await training_client.forward_backward_async(
            datums, loss_fn=args.loss_fn
        )
        optim_future = await training_client.optim_step_async(adam_params)
        _fwd_bwd_result = await fwd_bwd_future.result_async()
        _optim_result = await optim_future.result_async()

        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        tinker_client.set_sampling_client(sampling_client, policy_version=str(step + 1))

        win, loss, draw = _count_results(rollouts)
        total = max(len(rollouts), 1)
        completion_lens = [
            len(step.trace.completion_token_ids)
            for rollout in rollouts
            for step in rollout.steps
            if step.trace is not None
        ]
        metrics: Dict[str, float] = {
            "train/avg_total_reward": sum(r.total_reward for r in rollouts) / total,
            "train/win_rate": win / total,
            "train/loss_rate": loss / total,
            "train/draw_rate": draw / total,
            "train/illegal_rate": _count_flag(rollouts, "illegal_move") / total,
            "train/parse_error_rate": _count_flag(rollouts, "parse_error") / total,
            "train/gto_move_rate": _gto_move_rate(rollouts),
            "train/rollouts": len(rollouts),
            "train/datums": len(datums),
            "train/avg_completion_length": sum(completion_lens) / max(len(completion_lens), 1),
            "time/total": time.time() - t_start,
        }
        logger.log_metrics(metrics, step=step)

        if args.eval_every > 0 and args.eval_episodes > 0:
            if not (eval_ran_before and step == 0) and step % args.eval_every == 0:
                await run_eval_if_needed(step=step)

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{step + 1:06d}",
                log_path=args.log_path,
                loop_state={"step": step + 1},
                kind=args.save_kind,
            )

        step += 1

    if args.final_save:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=args.log_path,
            loop_state={"step": step},
            kind=args.save_kind,
        )
    logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tic-Tac-Toe using Ludic + Tinker.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", default=None, help="Tinker service base URL.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--rollouts-per-update", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument("--max-steps-per-episode", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument(
        "--agent-starts-as",
        choices=["x", "o", "mixed"],
        default="mixed",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--train-temperature", type=float, default=0.7)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-before-start", action="store_true", default=True)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-concurrency", type=int, default=32)
    parser.add_argument("--eval-temperature", type=float, default=0.7)
    parser.add_argument("--rollout-log", default="tictactoe_train_rollouts_tinker.jsonl")
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--positive-only", action="store_true", default=False)
    parser.add_argument("--normalize-adv", action="store_true", default=True)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--loss-fn", default="cispo")
    parser.add_argument("--log-path", default="ludic-ttt-tinker")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-kind", choices=["state", "sampler", "both"], default="both")
    parser.add_argument("--final-save", action="store_true", default=False)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    if args.rollouts_per_update <= 0:
        raise ValueError("--rollouts-per-update must be > 0.")
    if args.rollouts_per_update % args.group_size != 0:
        raise ValueError("--rollouts-per-update must be divisible by --group-size.")

    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
