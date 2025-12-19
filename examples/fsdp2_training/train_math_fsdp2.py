"""
FSDP2 MATH training scaffold (3 training GPUs + 1 vLLM GPU).

Assumptions:
  - GPU0 runs vLLM serving Qwen2.5-7B-Instruct.
  - GPUs 1-3 are reserved for training (set CUDA_VISIBLE_DEVICES=1,2,3).
  - Launch with torchrun: torchrun --nproc_per_node=3 examples/fsdp2_training/train_math_fsdp2.py

This is a skeleton to illustrate FSDP2 wrapping + Ludic trainer wiring.
Tune batch sizes, steps, and sampling to your hardware.
"""

from __future__ import annotations

import argparse
import logging
import json
import os
import queue
import sys
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.distributed import fsdp
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset  # type: ignore

from environments.math import MATHEnv
from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient, InferenceSpec, SamplingParams, ReturnSpec
from ludic.interaction import SingleAgentSyncProtocol
from ludic.distributed import create_vllm_publisher
from ludic.parsers import boxed_parser, extract_last_boxed_content
from ludic.eval import EngineEvaluator
from ludic.training import (
    RolloutEngine,
    RolloutBatchSource,
    Trainer,
    TrainerConfig,
    CheckpointConfig,
    make_dataset_queue_requests_fn,
    make_grpo,
    RequestsExhausted,
    RolloutRequest,
    EnvSpec,
    ProtocolSpec,
)
from ludic.training import Reducer, RichLiveLogger


def configure_logging(*, rank: int, level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format=f"%(asctime)s [rank{rank}] %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    for noisy in ("urllib3", "aiohttp", "httpx", "openai", "datasets", "transformers"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))


def silence_nonzero_ranks(*, rank: int) -> None:
    if rank == 0:
        return
    logging.getLogger().setLevel(logging.ERROR)
    for noisy in ("urllib3", "aiohttp", "httpx", "openai", "datasets", "transformers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    sys.stdout = open(os.devnull, "w")


def init_dist(*, local_rank: int) -> int:
    if dist.is_initialized():
        return dist.get_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    else:
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
        )
    return dist.get_rank()


def shard_samples(samples: List[Dict[str, Any]], rank: int, world_size: int) -> List[Dict[str, Any]]:
    return [s for i, s in enumerate(samples) if i % world_size == rank]


MATH_TRAIN_DATASET = "qwedsacf/competition_math"
MATH_EVAL_DATASET = "HuggingFaceH4/MATH-500"


def _load_dataset_split(dataset: str, split: str):
    """
    Load an HF dataset split without fallback.
    """
    try:
        return load_dataset(dataset, split=split)
    except Exception as e:
        raise SystemExit(f"Could not load dataset={dataset!r} split={split!r}: {e}") from e


def load_math_train(limit: int | None) -> List[Dict[str, Any]]:
    ds = load_dataset(MATH_TRAIN_DATASET, split="train")
    samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        problem = row.get("problem")
        solution = row.get("solution")
        if problem is None:
            raise SystemExit(f"{MATH_TRAIN_DATASET} sample is missing 'problem' at index={idx}")
        if solution is None:
            raise SystemExit(f"{MATH_TRAIN_DATASET} sample is missing 'solution' at index={idx}")
        samples.append(
            {
                "problem": str(problem),
                "solution": str(solution),
                "id": row.get("id", idx),
                "level": row.get("level"),
                "type": row.get("type"),
            }
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_math_eval(limit: int | None) -> List[Dict[str, Any]]:
    ds = _load_dataset_split(MATH_EVAL_DATASET, split="test")
    samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        solution = row.get("solution")
        problem = row.get("problem")
        sample: Dict[str, Any] = {
            "problem": str(problem) if problem is not None else "",
            "id": row.get("id", idx),
            "level": row.get("level"),
            "type": row.get("type"),
        }
        if problem is None:
            raise SystemExit(f"{MATH_EVAL_DATASET} sample is missing 'problem' at index={idx}")
        if solution is None:
            raise SystemExit(f"{MATH_EVAL_DATASET} sample is missing 'solution' at index={idx}")
        if extract_last_boxed_content(str(solution)) is None:
            raise SystemExit(
                f"{MATH_EVAL_DATASET} sample at index={idx} does not contain a final \\boxed{...} answer"
            )
        sample["solution"] = str(solution)
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=100, help="Number of trainer steps.")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=11)
    parser.add_argument("--train-temperature", type=float, default=1.0)
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "Think step by step and put your final answer into \\boxed{...}."
        ),
    )
    parser.add_argument("--rollout-log", type=str, default="fsdp2_math_rollouts.jsonl")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_math_fsdp2")
    parser.add_argument(
        "--eval-log",
        type=str,
        default="fsdp2_eval_metrics.jsonl",
        help="Append eval metrics here (jsonl). Set empty to disable.",
    )
    parser.add_argument(
        "--eval-every", type=int, default=10, help="Eval every N train steps (0 disables).")
    parser.add_argument("--eval-before-start", action="store_true", default=False, help="Run eval once at step 0.")
    parser.add_argument("--eval-limit", type=int, default=100, help="Number of test samples for eval (0 disables).")
    parser.add_argument("--eval-concurrency", type=int, default=32)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens per completion (train + eval).",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--logger", choices=["rich", "print", "none"], default="rich")
    parser.add_argument(
        "--rank0-only-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Silence stdout/logging on non-rank0 processes (recommended).",
    )
    parser.add_argument("--final-save", action="store_true", help="Save a final checkpoint after training completes.")
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = init_dist(local_rank=env_local_rank)
    world_size = dist.get_world_size()
    configure_logging(rank=rank, level=args.log_level)
    if args.rank0_only_output:
        silence_nonzero_ranks(rank=rank)

    device = torch.device(f"cuda:{env_local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    rollout_log_path = args.rollout_log.replace(".jsonl", f".rank{rank}.jsonl")
    eval_log_path = args.eval_log if args.eval_log else None
    if rank == 0:
        os.makedirs(os.path.dirname(rollout_log_path) or ".", exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        if eval_log_path:
            os.makedirs(os.path.dirname(eval_log_path) or ".", exist_ok=True)
    dist.barrier()
    logging.getLogger(__name__).info(
        "Initialized distributed training (world_size=%s, device=%s).", world_size, device
    )

    # Data
    all_train_samples = load_math_train(args.limit)
    train_samples = shard_samples(all_train_samples, rank, world_size)
    if not train_samples:
        raise SystemExit(f"Rank {rank}: no samples after sharding.")

    do_eval = bool(args.eval_limit and args.eval_limit > 0)
    eval_samples: List[Dict[str, Any]] = []
    if rank == 0 and do_eval:
        eval_samples = load_math_eval(args.eval_limit)

    samples_q: queue.Queue = queue.Queue()
    for idx, s in enumerate(train_samples):
        samples_q.put((idx, s))

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    mp_policy = fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    blocks = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers  # type: ignore[attr-defined]
    elif hasattr(model, "layers"):
        blocks = model.layers  # type: ignore[attr-defined]
    if blocks is not None:
        for layer in blocks:
            fsdp.fully_shard(layer, mp_policy=mp_policy)
    fsdp.fully_shard(model, mp_policy=mp_policy)

    action_parser = boxed_parser

    # Shared client for inference (rank0 does weight updates)
    client = VLLMChatClient(
        host=args.vllm_host,
        port=args.vllm_port,
        enable_weight_updates=(rank == 0),
        device=str(device),
    )
    publisher = create_vllm_publisher(client, rank0_only=True)

    env_registry = {"math": lambda sample: MATHEnv(sample=sample, system_prompt=args.system_prompt)}

    def protocol_factory():
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,
                model=args.model,
                ctx=FullDialog(system_prompt=args.system_prompt),
                parser=action_parser,
            )
        )

    protocol_registry = {"single_agent": protocol_factory}

    algo = make_grpo(
        name="grpo",
        group_size=args.group_size,
        group_normalize_adv=True,
        clip_eps=0.1,
        length_normalize=True,
    )

    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=rollout_log_path,
    )
    train_inference = InferenceSpec(
        sampling=SamplingParams(temperature=args.train_temperature, max_tokens=args.max_tokens),
        return_=ReturnSpec.for_rl(top_logprobs_k=1),
    )
    requests_fn = make_dataset_queue_requests_fn(
        samples_q,
        batch_size=args.batch_size,
        env_kind="math",
        protocol_kind="single_agent",
        inference=train_inference,
        protocol_kwargs={},
        request_meta_fn=lambda idx, sample: {
            "sample_index": idx,
            "problem_id": sample.get("id", idx),
            "level": sample.get("level"),
            "type": sample.get("type"),
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

    cfg = TrainerConfig(
        model_device=str(device),
        grad_accum_steps=2,
        max_grad_norm=0.5,
        pad_token_id=tokenizer.pad_token_id,
        reduce_stats_across_ranks=True,
        eval_at_start=bool(args.eval_before_start and do_eval),
        eval_every_n_steps=(int(args.eval_every) if args.eval_every and args.eval_every > 0 and do_eval else None),
        eval_concurrency=int(args.eval_concurrency),
        eval_max_steps=1,
        eval_timeout_s=None,
    )
    checkpoint_cfg = CheckpointConfig(
        output_dir=args.checkpoint_dir,
        every_n_steps=25,
        max_to_keep=2,
        save_optimizer=True,
    )
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
    }
    train_logger = None
    if rank == 0:
        if args.logger == "none":
            train_logger = None
        elif args.logger == "print" or not sys.stdout.isatty():
            from ludic.training import PrintLogger

            train_logger = PrintLogger(
                prefix="[trainer]",
                keys=[
                    "loss",
                    "avg_total_reward",
                    "correct_rate",
                    "parse_err_rate",
                    "eval_accuracy",
                    "eval_parse_error_rate",
                    "eval_avg_completion_tokens",
                    "num_rollouts",
                    "num_samples",
                ],
                precision=4,
            )
        else:
            train_logger = RichLiveLogger(
                keys=[
                    "loss",
                    "avg_total_reward",
                    "correct_rate",
                    "parse_err_rate",
                    "eval_accuracy",
                    "eval_parse_error_rate",
                    "eval_avg_completion_tokens",
                    "num_rollouts",
                    "num_samples",
                ],
                spark_key="avg_total_reward",
                history=100,
                precision=4,
            )

    trainer = Trainer(
        model=model,
        algo=algo,
        batch_source=batch_source,
        publisher=publisher,
        cfg=cfg,
        enable_gradient_checkpointing=True,
        checkpoint_config=checkpoint_cfg,
        train_logger=train_logger,
        reducers=reducers,
        evaluator=(
            None
            if not do_eval
            else EngineEvaluator(
                engine=RolloutEngine(env_registry=env_registry, protocol_registry=protocol_registry),
                requests_fn=lambda: [
                    RolloutRequest(
                        env=EnvSpec(kind="math", kwargs={"sample": sample}),
                        protocol=ProtocolSpec(kind="single_agent", kwargs={}),
                        num_episodes=1,
                        env_seed=int(idx),
                        sampling_seed=int(idx),
                        inference=InferenceSpec(
                            sampling=SamplingParams(
                                temperature=float(args.eval_temperature),
                                max_tokens=int(args.max_tokens),
                            ),
                            return_=ReturnSpec.for_eval(return_token_ids=True),
                        ),
                        meta={"eval_sample_index": idx, "problem_id": sample.get("id", idx)},
                    )
                    for idx, sample in enumerate(eval_samples)
                ],
                reducers={
                    "accuracy": Reducer(kind="count_true", source="correct", normalize_by="rollouts", as_percent=True),
                    "parse_error_rate": Reducer(kind="count_true", source="parse_error", normalize_by="samples", as_percent=True),
                    "avg_completion_tokens": Reducer(kind="mean", source="completion_length"),
                },
                max_steps=cfg.eval_max_steps,
                timeout_s=cfg.eval_timeout_s,
                concurrency=cfg.eval_concurrency,
            )
        ),
    )

    def log_eval_metrics(step: int, metrics: Dict[str, float]) -> None:
        if rank != 0 or not eval_log_path:
            return
        entry = {"step": int(step)}
        entry.update({k: float(v) for k, v in metrics.items()})
        try:
            with open(eval_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logging.getLogger(__name__).exception("Failed to write eval metrics")

    def train_log(stats: Dict[str, float]) -> None:
        if rank != 0:
            return
        step = int(stats.get("train_step", 0))
        print(
            f"[rank0 step {step}] loss={stats.get('loss'):.4f} reward={stats.get('avg_total_reward'):.4f}",
            flush=True,
        )

    try:
        trainer.train_sync(
            args.train_steps,
            log_every=1,
            log_fn=train_log,
            eval_log_fn=lambda metrics: log_eval_metrics(trainer.current_step, metrics),
        )
    except RequestsExhausted:
        if rank == 0:
            print("No more training samples; stopping.")

    if args.final_save:
        try:
            ckpt_path = trainer.save_checkpoint(metadata={"final": True})
            if rank == 0:
                print(f"Final checkpoint saved to: {ckpt_path}")
        except RuntimeError:
            pass  # No checkpointer configured

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
