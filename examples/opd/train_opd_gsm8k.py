"""
GSPO + OPD hybrid training on GSM8K using vLLM.

This example combines:
  - GSPO (Group-Sorted Policy Optimization): Task rewards from GSM8K correctness
  - OPD (On-Policy Distillation): Dense per-token supervision from teacher

The hybrid approach uses a composite loss:
  1. ClippedSurrogateLoss (GSPO): Policy gradient with group-normalized advantages
  2. ReverseKLLoss (OPD): KL(student || teacher) = log π_student - log π_teacher

This gives you both:
  - Task-specific learning from environment rewards (sparse but grounded)
  - Distribution matching from teacher (dense per-token feedback)

The key insight: teacher logprobs are an *intrinsic scorer* attached to the Agent.
The scorer runs during Agent.act() and scores flow through to training.

Reference: https://thinkingmachines.ai/blog/on-policy-distillation

Usage:
    python train_opd_gsm8k.py \
        --student-model Qwen/Qwen2.5-0.5B-Instruct \
        --teacher-model Qwen/Qwen2.5-7B-Instruct \
        --limit 1000

Requirements:
    - vLLM servers running for student (port 8000) and teacher (port 8001)
"""

from __future__ import annotations

import argparse
import os
import sys
import queue
from typing import List, Dict, Any

import torch
from datasets import load_dataset  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer

from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient, InferenceSpec, SamplingParams, ReturnSpec, HFChatTemplate
from ludic.interaction import SingleAgentSyncProtocol
from ludic.parsers import boxed_parser
from ludic.distributed.adapters import create_vllm_publisher
from ludic.eval import EngineEvaluator
from ludic.training import (
    RolloutEngine,
    RolloutBatchSource,
    Trainer,
    TrainerConfig,
    CheckpointConfig,
    make_dataset_queue_requests_fn,
    RequestsExhausted,
    RolloutRequest,
    EnvSpec,
    ProtocolSpec,
    RLAlgorithm,
)
from ludic.training import Reducer, RichLiveLogger, PrintLogger, TeeLogger, WandbLogger, default_reducers
from ludic.training.scoring import make_vllm_teacher_scorer
from ludic.training.credit_assignment import GroupNormalizedReturn
from ludic.training.loss import ClippedSurrogateLoss, ReverseKLLoss, CompositeLoss, LossTerm
from ludic.training.algorithm import validate_actor_logps
from environments.gsm8k import GSM8KEnv


def load_gsm8k(split: str, limit: int | None) -> List[Dict[str, Any]]:
    """Load GSM8K dataset samples."""
    ds = load_dataset("gsm8k", "main", split=split)
    samples: List[Dict[str, Any]] = []
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
    return samples


def main():
    parser = argparse.ArgumentParser(description="OPD training on GSM8K")

    # Model configuration
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Student model name/path")
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Teacher model name/path")

    # vLLM server configuration
    parser.add_argument("--student-host", default="127.0.0.1")
    parser.add_argument("--student-port", type=int, default=8000)
    parser.add_argument("--teacher-host", default="127.0.0.1")
    parser.add_argument("--teacher-port", type=int, default=8001)

    # Data configuration
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit training samples (None = use all)")

    # Training configuration
    parser.add_argument("--rollouts-per-update", type=int, default=256,
                        help="Total rollouts per update (must be divisible by --group-size)")
    parser.add_argument("--group-size", type=int, default=8,
                        help="Group size for grouped advantages")
    parser.add_argument("--train-steps", type=int, default=20,
                        help="Number of training steps; 0 = run until samples exhausted")
    parser.add_argument("--max-seq-len", type=int, default=1024,
                        help="Max tokens per sample")
    parser.add_argument("--micro-token-budget", type=int, default=16384,
                        help="Max padded tokens per micro-batch")
    parser.add_argument("--max-completion-tokens", type=int, default=512,
                        help="Max completion tokens per rollout")
    parser.add_argument("--train-temperature", type=float, default=1.0,
                        help="Sampling temperature for training rollouts")
    parser.add_argument("--concurrency", type=int, default=64,
                        help="Rollout concurrency")

    # OPD-specific configuration (hybrid GSPO + KL)
    parser.add_argument("--kl-coeff", type=float, default=1.0,
                        help="Coefficient for reverse KL loss term")

    # System prompt
    parser.add_argument("--system-prompt", type=str,
                        default="First, think step by step. Then put your final answer inside \\boxed{...}.",
                        help="System prompt for GSM8K env; set to '' to use the model default.")

    # Logging
    parser.add_argument("--rollout-log", type=str, default="opd_rollouts.jsonl")
    parser.add_argument("--logger", type=str, default="rich",
                        help="Comma-separated loggers: rich, print, wandb, none.")

    # Evaluation
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Eval every N train steps.")
    parser.add_argument("--eval-before-start", action="store_true", default=True,
                        help="Run eval once before training begins.")
    parser.add_argument("--eval-limit", type=int, default=1000,
                        help="Number of test samples for eval.")
    parser.add_argument("--eval-concurrency", type=int, default=64)
    parser.add_argument("--eval-temperature", type=float, default=0.0,
                        help="Sampling temperature for eval passes.")

    # Checkpointing
    parser.add_argument("--final-save", action="store_true",
                        help="Save a final checkpoint after training completes.")

    args = parser.parse_args()

    # Validation
    if args.rollouts_per_update <= 0:
        raise ValueError("--rollouts-per-update must be > 0.")
    if args.rollouts_per_update % args.group_size != 0:
        raise ValueError("--rollouts-per-update must be divisible by --group-size.")
    if args.max_completion_tokens > args.max_seq_len:
        raise ValueError("--max-completion-tokens must be <= --max-seq-len.")

    # Setup rollout log path
    rollout_log_path = os.path.abspath(args.rollout_log)
    os.makedirs(os.path.dirname(rollout_log_path) or ".", exist_ok=True)
    # Touch the file so tailing works even before the first rollout is written
    open(rollout_log_path, "a", encoding="utf-8").close()

    # Load training data
    print(f"Loading GSM8K {args.split} split...")
    train_samples = load_gsm8k(args.split, args.limit)
    if not train_samples:
        raise SystemExit("No GSM8K samples loaded.")
    print(f"Loaded {len(train_samples)} training samples")

    # Load eval data
    eval_samples = load_gsm8k("test", args.eval_limit) if args.eval_limit else []
    if eval_samples:
        print(f"Loaded {len(eval_samples)} eval samples")

    # Create sample queue
    samples_q: queue.Queue = queue.Queue()
    for idx, s in enumerate(train_samples):
        samples_q.put((idx, s))

    # Load tokenizer and model
    print(f"Loading student model: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    model = AutoModelForCausalLM.from_pretrained(args.student_model, dtype=torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Create vLLM client for student
    client = VLLMChatClient(
        host=args.student_host,
        port=args.student_port,
        enable_weight_updates=True,
    )
    publisher = create_vllm_publisher(client)
    chat_template = HFChatTemplate(tokenizer)

    # Teacher scorer - computes per-token logprobs during Agent.act()
    teacher_scorer = make_vllm_teacher_scorer(
        base_url=f"http://{args.teacher_host}:{args.teacher_port}",
        model=args.teacher_model,
    )

    # Registries
    env_registry = {
        "gsm8k": lambda sample: GSM8KEnv(sample=sample, system_prompt=args.system_prompt)
    }

    def protocol_factory():
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,
                model=args.student_model,
                ctx=FullDialog(),
                parser=boxed_parser,
                chat_template=chat_template,
                scorers=[teacher_scorer],  # OPD: teacher provides per-token scores
            )
        )

    protocol_registry = {"single_agent": protocol_factory}

    # Algorithm: GSPO (task rewards) + OPD (teacher KL) hybrid
    # Credit assignment from GSPO: group-normalized returns
    credit_assigner = GroupNormalizedReturn(
        group_size=args.group_size,
        normalize_adv=True,
        positive_only=False,
    )
    # Composite loss: PPO-style clipped surrogate + reverse KL
    loss = CompositeLoss(terms=[
        LossTerm(
            name="gspo",
            loss=ClippedSurrogateLoss(
                clip_eps_low=3e-4,
                clip_eps_high=4e-4,
                length_normalize=True,
            ),
            weight=1.0,
        ),
        LossTerm(
            name="kl",
            loss=ReverseKLLoss(
                coeff=1.0,
                length_normalize=True,
            ),
            weight=args.kl_coeff,
        ),
    ])
    algo = RLAlgorithm(
        name="gspo_opd",
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=validate_actor_logps,
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
        # Ask vLLM for token IDs + chosen-token logprobs for importance sampling
        return_=ReturnSpec.for_rl(top_logprobs_k=1),
    )
    base_requests = args.rollouts_per_update // args.group_size
    requests_fn = make_dataset_queue_requests_fn(
        samples_q,
        batch_size=base_requests,
        env_kind="gsm8k",
        protocol_kind="single_agent",
        inference=train_inference,
        protocol_kwargs={},
        request_meta_fn=lambda idx, sample: {
            "sample_index": idx,
            "question_id": sample.get("id", idx),
        },
        env_seed_fn=lambda idx, _sample: idx,
        sampling_seed_fn=lambda idx, _sample: idx,
        group_size=args.group_size,
    )
    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=algo.credit_assigner,
        requests_fn=requests_fn,
        max_steps=1,
        concurrency=args.concurrency,
    )

    # Trainer config
    cfg = TrainerConfig(
        model_device="cuda" if torch.cuda.is_available() else "cpu",
        max_seq_len=args.max_seq_len,
        micro_token_budget=args.micro_token_budget,
        max_grad_norm=0.5,
        pad_token_id=tokenizer,
        eval_at_start=bool(args.eval_before_start and eval_samples),
        eval_every_n_steps=(args.eval_every if args.eval_every and args.eval_every > 0 and eval_samples else None),
        eval_concurrency=args.eval_concurrency,
        eval_max_steps=1,
    )

    # Checkpoint config
    checkpoint_cfg = CheckpointConfig(
        output_dir="checkpoints_opd",
        every_n_steps=25,
        max_to_keep=2,
        save_optimizer=True,
    )

    # Reducers
    reducers = {
        "correct_rate": Reducer(
            kind="count_true",
            source="correct",
            normalize_by="rollouts",
        ),
        "parse_err_rate": Reducer(
            kind="count_true",
            source="parse_error",
            normalize_by="samples",
        ),
        "total_completion_tokens": Reducer(
            kind="sum",
            source="completion_length",
        ),
    }
    reducers = {**default_reducers(), **reducers}

    # Logger keys
    logger_keys = [
        "train/loss",
        "train/gspo/loss",
        "train/kl/loss",
        "train/kl/reverse_kl_mean",
        "train/avg_total_reward",
        "train/correct_rate",
        "train/parse_err_rate",
        "train/avg_completion_length",
        "train/total_completion_tokens",
        "eval/accuracy",
        "eval/parse_error_rate",
        "eval/avg_completion_tokens",
        "train/target_rollouts",
        "train/num_samples",
    ]

    train_logger = None
    raw_logger = args.logger or "rich"
    logger_tokens = [tok.strip().lower() for tok in raw_logger.replace("+", ",").split(",") if tok.strip()]
    valid_loggers = {"rich", "print", "wandb", "none"}
    unknown = [tok for tok in logger_tokens if tok not in valid_loggers]
    if unknown:
        raise SystemExit(f"Unknown logger(s): {unknown}. Valid: {sorted(valid_loggers)}")
    if "none" in logger_tokens:
        logger_tokens = ["none"]

    console_logger = None
    if "print" in logger_tokens:
        console_logger = PrintLogger(prefix="[opd]", keys=logger_keys, precision=4)
    elif "rich" in logger_tokens:
        if not sys.stdout.isatty():
            console_logger = PrintLogger(prefix="[opd]", keys=logger_keys, precision=4)
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

    # Eval reducers
    eval_reducers = {
        "accuracy": Reducer(kind="count_true", source="correct", normalize_by="samples", as_percent=True),
        "parse_error_rate": Reducer(kind="count_true", source="parse_error", normalize_by="samples", as_percent=True),
        "avg_completion_tokens": Reducer(kind="mean", source="completion_length"),
    }

    # Create trainer
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
            if not eval_samples
            else EngineEvaluator(
                engine=RolloutEngine(env_registry=env_registry, protocol_registry=protocol_registry),
                requests_fn=lambda: [
                    RolloutRequest(
                        env=EnvSpec(
                            kind="gsm8k",
                            kwargs={"sample": sample},
                        ),
                        protocol=ProtocolSpec(kind="single_agent"),
                        env_seed=idx,
                        sampling_seed=idx,
                        inference=InferenceSpec(
                            sampling=SamplingParams(
                                temperature=args.eval_temperature,
                                max_tokens=args.max_completion_tokens,
                            ),
                            return_=ReturnSpec.for_eval(return_token_ids=True),
                        ),
                        num_episodes=1,
                        meta={"eval_sample_index": idx, "question_id": sample.get("id", idx)},
                    )
                    for idx, sample in enumerate(eval_samples)
                ],
                reducers=eval_reducers,
                max_steps=1,
                timeout_s=cfg.eval_timeout_s,
                concurrency=cfg.eval_concurrency,
            )
        ),
    )

    # Train
    print(f"\nStarting OPD training for {args.train_steps} steps...")
    print(f"  Student: {args.student_model}")
    print(f"  Teacher: {args.teacher_model}")
    print(f"  KL coefficient: {args.kl_coeff}")
    print()

    try:
        trainer.train_sync(args.train_steps)
    except RequestsExhausted:
        print("No more training samples; stopping.")

    if args.final_save:
        try:
            ckpt_path = trainer.save_checkpoint(metadata={"final": True})
            print(f"Final checkpoint saved to: {ckpt_path}")
        except RuntimeError:
            pass  # No checkpointer configured

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
