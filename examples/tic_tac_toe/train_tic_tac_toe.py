"""
Minimal Tic-Tac-Toe training scaffold.

This wires together:
  - TicTacToeEnv single-agent episodes
  - SingleAgentSyncProtocol with a shared VLLMChatClient
  - RolloutBatchSource + GroupNormalizedReturn credit
  - Trainer with REINFORCE loss
  - Optional periodic eval of win rate
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from environments.tic_tac_toe import TicTacToeEnv
from ludic.agent import Agent
from ludic.context import FullDialog, TruncatedThinkingContext
from ludic.inference import VLLMChatClient, InferenceSpec, SamplingParams, ReturnSpec, HFChatTemplate
from ludic.interaction import SingleAgentSyncProtocol
from ludic.distributed.adapters import create_vllm_publisher
from ludic.parsers import compose_parsers, think_prefix_parser, xml_tag_parser
from ludic.eval import EngineEvaluator
from ludic.training import (
    RLAlgorithm,
    RolloutEngine,
    RolloutBatchSource,
    Trainer,
    TrainerConfig,
    CheckpointConfig,
    EnvSpec,
    ProtocolSpec,
    RolloutRequest,
    GroupNormalizedReturn,
    GRPORequestStrategy,
    ReinforceLoss,
)
from ludic.training import Reducer, RichLiveLogger, PrintLogger, TeeLogger, WandbLogger, default_reducers

# STRICT: require <think>...</think> then exactly one <move>...</move>.
# Success reward is set to 0.0 so multiple turns do not gain extra parser reward.
TICTACTOE_PARSER = compose_parsers(
    partial(think_prefix_parser, success_reward=0.0, error_reward=-1.0),
    xml_tag_parser("move", exact=True, success_reward=0.0, error_reward=-1.0),
)


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
):
    next_mixed_start = None

    def _next_mixed_start() -> bool:
        nonlocal next_mixed_start
        if next_mixed_start is None:
            next_mixed_start = bool(torch.randint(0, 2, (1,), generator=rng).item())
        start = next_mixed_start
        next_mixed_start = not next_mixed_start
        return start

    def _fn() -> List[RolloutRequest]:
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

    return _fn


def main():
    parser = argparse.ArgumentParser(description="Train a model on Tic-Tac-Toe using Ludic + vLLM.")
    parser.add_argument("--model", default="hallerite/Qwen2.5-7B-TTT")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for sampling episode seeds.")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument(
        "--rollouts-per-update",
        type=int,
        default=8,
        help="Total rollouts per update (must be divisible by --group-size).",
    )
    parser.add_argument("--train-steps", type=int, default=30, help="Number of trainer steps.")
    parser.add_argument("--max-steps-per-episode", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=8, help="Group size for grouped advantages (GRPO-style).")
    parser.add_argument(
        "--agent-starts-as",
        choices=["x", "o", "mixed"],
        default="mixed",
        help="Which side the agent plays as: x (start), o (second), or mixed (50/50).",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (RL-friendly defaults from LoRA best-practice guides).")
    parser.add_argument(
        "--lora-alpha-mult",
        type=float,
        default=2.0,
        help="Multiplier applied to rank to set lora_alpha (alpha = rank * mult).",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout probability.")
    parser.add_argument("--train-temperature", type=float, default=1.0, help="Sampling temperature for training rollouts.")
    parser.add_argument("--eval-every", type=int, default=10, help="Eval every N train steps.")
    parser.add_argument("--eval-before-start", action="store_true", default=True, help="Run eval once before training begins.")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Number of episodes for eval.")
    parser.add_argument("--eval-concurrency", type=int, default=32)
    parser.add_argument("--eval-temperature", type=float, default=0.6, help="Sampling temperature for eval passes.")
    parser.add_argument("--rollout-log", type=str, default="tictactoe_train_rollouts.jsonl")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_tictactoe", help="Checkpoint output directory.")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint every N steps (0 to disable).")
    parser.add_argument("--max-to-keep", type=int, default=2, help="Max checkpoints to keep.")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max tokens per sample.")
    parser.add_argument("--micro-token-budget", type=int, default=8192, help="Max padded tokens per micro-batch.")
    parser.add_argument("--max-completion-tokens", type=int, default=512, help="Max completion tokens per rollout.")
    parser.add_argument("--ctx", choices=["full", "truncated"], default="full",
                        help="Context strategy: 'full' (FullDialog) or 'truncated' (TruncatedThinkingContext)")
    parser.add_argument("--final-save", action="store_true", help="Save a final checkpoint after training completes.")
    parser.add_argument("--positive-only", action="store_true", help="Only learn from positive advantages; clip negative ones to 0.")
    parser.add_argument(
        "--logger",
        type=str,
        default="rich",
        help="Comma-separated loggers: rich, print, wandb, none.",
    )

    args = parser.parse_args()
    if args.rollouts_per_update <= 0:
        raise ValueError("--rollouts-per-update must be > 0.")
    if args.rollouts_per_update % args.group_size != 0:
        raise ValueError("--rollouts-per-update must be divisible by --group-size.")
    if args.max_completion_tokens > args.max_seq_len:
        raise ValueError("--max-completion-tokens must be <= --max-seq-len.")

    rollout_log_path = os.path.abspath(args.rollout_log)
    os.makedirs(os.path.dirname(rollout_log_path) or ".", exist_ok=True)
    open(rollout_log_path, "a", encoding="utf-8").close()

    # Seeds for deterministic episode resets
    rng = torch.Generator()
    rng.manual_seed(args.seed if args.seed is not None else 0)

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    chat_template = HFChatTemplate(tokenizer)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Apply a lightweight LoRA adapter to train only a small subset of params.
    # Apply LoRA to all linear projections (per “LoRA Without Regret” guidance: all-linear > attention-only).
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=int(args.lora_rank * args.lora_alpha_mult),
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules="all-linear",
    )
    model = get_peft_model(base_model, lora_config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.print_trainable_parameters()

    # Shared client for inference
    client = VLLMChatClient(host=args.host, port=args.port, enable_weight_updates=True)
    publisher = create_vllm_publisher(client)

    # Registries
    env_registry = {"tictactoe": lambda agent_starts=True: TicTacToeEnv(agent_starts=agent_starts)}

    # Extend the env's suggested system prompt with explicit CoT + XML move instructions.
    base_prompt = TicTacToeEnv().suggested_sysprompt or ""
    system_prompt = (
        base_prompt
        + "\n\nThink through the board in <think>...</think>. After </think>, output exactly one XML tag of the form <move>A1</move> and nothing else."
    )

    def protocol_factory():
        if args.ctx == "truncated":
            ctx = TruncatedThinkingContext(system_prompt=system_prompt)
        else:
            ctx = FullDialog(system_prompt=system_prompt)
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,
                model=args.model,
                ctx=ctx,
                parser=TICTACTOE_PARSER,
                chat_template=chat_template,
            ),
            stop_on_parse_error=True,
        )

    protocol_registry = {"single_agent": protocol_factory}

    # Algorithm
    algo = RLAlgorithm(
        name="grpo",
        credit_assigner=GroupNormalizedReturn(
            group_size=args.group_size,
            normalize_adv=True,
            positive_only=args.positive_only,
        ),
        loss=ReinforceLoss(length_normalize=True),
    )

    # Engine + batch source
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
        return_=ReturnSpec.for_rl(),
    )
    base_requests = args.rollouts_per_update // args.group_size
    base_requests_fn = build_requests_fn(rng, base_requests, train_inference, args.agent_starts_as)
    # Expand each logical request into a group with shared env seed and diverse sampling seeds.
    def requests_fn() -> List[RolloutRequest]:
        return GRPORequestStrategy(group_size=args.group_size).expand(base_requests_fn())
    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=algo.credit_assigner,
        requests_fn=requests_fn,
        max_steps=args.max_steps_per_episode,
        concurrency=args.concurrency,
    )

    # Trainer
    cfg = TrainerConfig(
        model_device="cuda" if torch.cuda.is_available() else "cpu",
        max_seq_len=args.max_seq_len,
        micro_token_budget=args.micro_token_budget,
        max_grad_norm=0.5,
        pad_token_id=tokenizer,
        lr=5e-5,
        eval_at_start=bool(args.eval_before_start and args.eval_episodes and args.eval_episodes > 0),
        eval_every_n_steps=(args.eval_every if args.eval_every and args.eval_every > 0 else None),
        eval_concurrency=args.eval_concurrency,
        eval_max_steps=args.max_steps_per_episode,
    )
    checkpoint_cfg = CheckpointConfig(
        output_dir=args.checkpoint_dir,
        every_n_steps=args.checkpoint_every,
        max_to_keep=args.max_to_keep,
        save_optimizer=True,
    )
    reducers = {
        "win_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "win",
            normalize_by="rollouts",
        ),
        "loss_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "loss",
            normalize_by="rollouts",
        ),
        "draw_rate": Reducer(
            kind="count_true",
            source="result",
            transform=lambda v: v == "draw",
            normalize_by="rollouts",
        ),
        "gto_move_rate": Reducer(
            kind="mean",
            source=lambda item: (
                None
                if item.meta.get("illegal_move") or item.meta.get("parse_error")
                else (1.0 if item.meta.get("gto_action") else 0.0)
            ), # this is to normalize over legal moves only
            as_percent=True,
        ),
        "illegal_rate": Reducer(
            kind="count_true",
            source="illegal_move",
            normalize_by="rollouts",
            as_percent=True,
        ),
        "parse_error_rate": Reducer(
            kind="count_true",
            source="parse_error",
            normalize_by="rollouts",
            as_percent=True,
        ),
        "truncated_rate": Reducer(
            kind="count_true",
            source="truncated",
            normalize_by="rollouts",
            as_percent=True,
        ),
        "avg_prompt_length": Reducer(
            kind="mean",
            source="prompt_length",
        ),
        "avg_completion_length": Reducer(
            kind="mean",
            source="completion_length",
        ),
        "total_completion_tokens": Reducer(
            kind="sum",
            source="completion_length",
        ),
    }
    reducers = {**default_reducers(), **reducers}

    logger_keys = [
        "train/loss",
        "train/avg_total_reward",
        "train/win_rate",
        "train/loss_rate",
        "train/draw_rate",
        "train/gto_move_rate",
        "train/illegal_rate",
        "train/parse_error_rate",
        "train/truncated_rate",
        "train/completion_truncated_rate",
        "train/seq_len_truncated_rate",
        "train/avg_prompt_length",
        "train/avg_completion_length",
        "train/total_completion_tokens",
        "eval/win_rate",
        "eval/loss_rate",
        "eval/draw_rate",
        "eval/gto_move_rate",
        "eval/illegal_rate",
        "eval/parse_error_rate",
        "eval/truncated_rate",
        "eval/avg_completion_tokens",
        "train/target_rollouts",
        "train/num_samples",
    ]

    raw_logger = args.logger or "rich"
    logger_tokens = [tok.strip().lower() for tok in raw_logger.replace("+", ",").split(",") if tok.strip()]
    valid_loggers = {"rich", "print", "wandb", "none"}
    unknown = [tok for tok in logger_tokens if tok not in valid_loggers]
    if unknown:
        raise SystemExit(f"Unknown logger(s): {unknown}. Valid: {sorted(valid_loggers)}")
    if "none" in logger_tokens:
        logger_tokens = ["none"]

    train_logger = None
    console_logger = None
    if "print" in logger_tokens:
        console_logger = PrintLogger(prefix="[trainer]", keys=logger_keys, precision=4)
    elif "rich" in logger_tokens:
        if not sys.stdout.isatty():
            console_logger = PrintLogger(prefix="[trainer]", keys=logger_keys, precision=4)
        else:
            console_logger = RichLiveLogger(
                keys=logger_keys,
                spark_key="train/avg_total_reward",
                history=100,
                precision=4,
            )

    wandb_logger = None
    if "wandb" in logger_tokens:
        wandb_logger = WandbLogger(config=dict(vars(args)))

    if logger_tokens != ["none"]:
        if console_logger and wandb_logger:
            train_logger = TeeLogger(console_logger, wandb_logger)
        else:
            train_logger = console_logger or wandb_logger

    eval_reducers = {
        "win_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "win", normalize_by="rollouts", as_percent=True),
        "loss_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "loss", normalize_by="rollouts", as_percent=True),
        "draw_rate": Reducer(kind="count_true", source="result", transform=lambda v: v == "draw", normalize_by="rollouts", as_percent=True),
        "gto_move_rate": Reducer(
            kind="mean",
            source=lambda item: (
                None
                if item.get("illegal_move") or item.get("parse_error")
                else (1.0 if item.get("gto_action") else 0.0)
            ),
            as_percent=True,
        ),
        "illegal_rate": Reducer(kind="count_true", source="illegal_move", normalize_by="rollouts", as_percent=True),
        "parse_error_rate": Reducer(kind="count_true", source="parse_error", normalize_by="rollouts", as_percent=True),
        "truncated_rate": Reducer(kind="count_true", source="truncated", normalize_by="rollouts", as_percent=True),
        "avg_completion_tokens": Reducer(kind="mean", source="completion_length"),
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
                inference=InferenceSpec(
                sampling=SamplingParams(
                    temperature=args.eval_temperature,
                    max_tokens=args.max_completion_tokens,
                ),
                    return_=ReturnSpec.for_eval(return_token_ids=True),
                ),
                meta={"eval_seed": seed, "agent_starts": agent_starts},
            )
            for seed, agent_starts in enumerate(start_flags)
        ]

    trainer = Trainer(
        model=model,
        algo=algo,
        batch_source=batch_source,
        publisher=publisher,
        enable_gradient_checkpointing=True,
        cfg=cfg,
        checkpoint_config=checkpoint_cfg,
        train_logger=train_logger,
        reducers=reducers,
        evaluator=(
            None
            if not args.eval_episodes or args.eval_episodes <= 0
            else EngineEvaluator(
                engine=RolloutEngine(env_registry=env_registry, protocol_registry=protocol_registry),
                requests_fn=eval_requests,
                reducers=eval_reducers,
                max_steps=cfg.eval_max_steps,
                timeout_s=cfg.eval_timeout_s,
                concurrency=cfg.eval_concurrency,
            )
        ),
    )
    trainer.train_sync(args.train_steps)
    if args.final_save:
        try:
            trainer.save_checkpoint(metadata={"final": True})
        except RuntimeError:
            pass  # No checkpointer configured


if __name__ == "__main__":
    main()
