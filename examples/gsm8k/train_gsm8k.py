"""
Minimal GSM8K training scaffold using DatasetQAEnv/GSM8KEnv and the built-in Trainer.

This wires together:
  - HF datasets for GSM8K samples
  - single-sample QA envs (GSM8KEnv)
  - SingleAgentSyncProtocol with a shared VLLMChatClient
  - RolloutBatchSource + MonteCarloReturn credit
  - Trainer with REINFORCE loss

This is a skeleton; adjust hyperparameters, batching, and model loading to your setup.
"""

from __future__ import annotations

import argparse
import os
import queue
from typing import List, Dict, Any

import torch
from datasets import load_dataset  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
from environments.gsm8k import GSM8KEnv

from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import VLLMChatClient, InferenceSpec, SamplingParams, ReturnSpec
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
    make_grpo,
    RequestsExhausted,
    RolloutRequest,
    EnvSpec,
    ProtocolSpec,
)
from ludic.training import Reducer, RichLiveLogger


def load_gsm8k(split: str, limit: int | None) -> List[Dict[str, Any]]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4, help="Rollout requests per batch source call")
    parser.add_argument("--train-steps", type=int, default=20, help="Number of trainer steps; 0 = run until samples are exhausted.")
    parser.add_argument("--group-size", type=int, default=8, help="Group size for grouped advantages.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="First, think step by step. Then put your final answer inside \\boxed{...}.",
        help="Optional system prompt for GSM8K env; set to '' to use the model default.",
    )
    parser.add_argument("--train-temperature", type=float, default=1.0, help="Sampling temperature for training rollouts.")
    parser.add_argument("--eval-every", type=int, default=10, help="Eval every N train steps.")
    parser.add_argument("--eval-before-start", action="store_true", default=True, help="Run eval once before training begins.")
    parser.add_argument("--eval-limit", type=int, default=750, help="Number of test samples for eval.")
    parser.add_argument("--eval-concurrency", type=int, default=64)
    parser.add_argument("--eval-temperature", type=float, default=0.0, help="Sampling temperature for eval passes.")
    parser.add_argument("--rollout-log", type=str, default="gsm8k_train_rollouts.jsonl")
    parser.add_argument("--final-save", action="store_true", help="Save a final checkpoint after training completes.")
    args = parser.parse_args()

    rollout_log_path = os.path.abspath(args.rollout_log)
    os.makedirs(os.path.dirname(rollout_log_path) or ".", exist_ok=True)
    # Touch the file so tailing works even before the first rollout is written
    open(rollout_log_path, "a", encoding="utf-8").close()

    # Data
    train_samples = load_gsm8k(args.split, args.limit)
    if not train_samples:
        raise SystemExit("No GSM8K samples loaded.")
    eval_samples = load_gsm8k("test", args.eval_limit) if args.eval_limit else []

    samples_q: queue.Queue = queue.Queue()
    for idx, s in enumerate(train_samples):
        samples_q.put((idx, s))

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Shared client for inference
    client = VLLMChatClient(host=args.host, port=args.port, enable_weight_updates=True)
    publisher = create_vllm_publisher(client)

    # Registries
    env_registry = {"gsm8k": lambda sample: GSM8KEnv(sample=sample, system_prompt=args.system_prompt)}

    def protocol_factory():
        return SingleAgentSyncProtocol(
            agent=Agent(
                client=client,
                model=args.model,
                ctx=FullDialog(),
                parser=boxed_parser,
            )
        )

    protocol_registry = {"single_agent": protocol_factory}

    # Algorithm (GRPO-style: group-relative advantages + PPO clipped objective)
    algo = make_grpo(
        name="grpo",
        group_size=args.group_size,
        group_normalize_adv=True,
        clip_eps=0.1,
        length_normalize=True,
    )

    # Engine + batch source
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=rollout_log_path,
    )
    train_inference = InferenceSpec(
        sampling=SamplingParams(temperature=args.train_temperature, max_tokens=512),
        # Ask vLLM for token IDs + chosen-token logprobs so GRPO can use rollout-time behavior logprobs.
        return_=ReturnSpec.for_rl(top_logprobs_k=1),
    )
    requests_fn = make_dataset_queue_requests_fn(
        samples_q,
        batch_size=args.batch_size,
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

    # Trainer
    cfg = TrainerConfig(
        model_device="cuda" if torch.cuda.is_available() else "cpu",
        grad_accum_steps=8,
        max_grad_norm=0.5,
        pad_token_id=tokenizer.pad_token_id,
        eval_at_start=bool(args.eval_before_start and eval_samples),
        eval_every_n_steps=(args.eval_every if args.eval_every and args.eval_every > 0 and eval_samples else None),
        eval_concurrency=args.eval_concurrency,
        eval_max_steps=1,
    )
    # Checkpoint every 25 steps into ./checkpoints_gsm8k
    checkpoint_cfg = CheckpointConfig(
        output_dir="checkpoints_gsm8k",
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
        "total_completion_tokens": Reducer(
            kind="sum",
            source="completion_length",
        ),
    }

    # Choose your logger: RichLiveLogger (with ASCII chart + metrics panel).
    train_logger = RichLiveLogger(
        keys=[
            "loss",
            "avg_total_reward",
            "correct_rate",
            "parse_err_rate",
            "avg_completion_length",
            "total_completion_tokens",
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

    eval_reducers = {
        "accuracy": Reducer(kind="count_true", source="correct", normalize_by="samples", as_percent=True),
        "parse_error_rate": Reducer(kind="count_true", source="parse_error", normalize_by="samples", as_percent=True),
        "avg_completion_tokens": Reducer(kind="mean", source="completion_length"),
    }

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
                            sampling=SamplingParams(temperature=args.eval_temperature, max_tokens=512),
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


if __name__ == "__main__":
    main()
