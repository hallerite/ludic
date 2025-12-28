"""
On-Policy Distillation (OPD) training on GSM8K using vLLM.

This example demonstrates on-policy distillation where:
  - A student model samples trajectories
  - A teacher model provides per-token logprobs as dense supervision
  - Training minimizes reverse KL divergence: KL(student || teacher)

This combines the benefits of:
  - On-policy learning (student samples from itself)
  - Dense supervision (per-token feedback, not sparse rewards)

The key insight: teacher logprobs are an *intrinsic scorer* attached to the Agent.
The scorer runs during Agent.act() and scores flow through to training.

Reference: https://thinkingmachines.ai/blog/on-policy-distillation

Usage:
    python train_opd_gsm8k.py \
        --student-model Qwen/Qwen3-8B-Base \
        --teacher-model Qwen/Qwen3-32B \
        --limit 1000

Requirements:
    - vLLM servers running for student (port 8000) and teacher (port 8001)
"""

from __future__ import annotations

import argparse
import queue
from typing import List, Dict, Any

import torch
from datasets import load_dataset  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer

from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import InferenceSpec, SamplingParams, ReturnSpec, HFChatTemplate
from ludic.interaction import SingleAgentSyncProtocol
from ludic.parsers import boxed_parser
from ludic.training import (
    RolloutEngine,
    RolloutBatchSource,
    Trainer,
    TrainerConfig,
    make_dataset_queue_requests_fn,
    make_opd,
    RequestsExhausted,
)
from ludic.training import Reducer, PrintLogger, default_reducers
from ludic.training.scoring import make_vllm_teacher_scorer

# Try to import environments
try:
    from environments.gsm8k import GSM8KEnv
except ImportError:
    # Fallback: define a minimal GSM8K env
    from ludic.envs import DatasetQAEnv

    class GSM8KEnv(DatasetQAEnv):
        def __init__(self, sample: Dict[str, Any], system_prompt: str = ""):
            super().__init__(
                question=sample["question"],
                ground_truth=self._extract_answer(sample["answer"]),
                system_prompt=system_prompt or "Solve the following problem step by step. Put your final answer in \\boxed{}.",
            )

        @staticmethod
        def _extract_answer(answer_text: str) -> str:
            # GSM8K answers have format "...\n#### answer"
            if "####" in answer_text:
                return answer_text.split("####")[-1].strip()
            return answer_text.strip()


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
    parser.add_argument("--rollouts-per-update", type=int, default=64,
                        help="Number of rollouts per training step")
    parser.add_argument("--train-steps", type=int, default=100,
                        help="Number of training steps")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Max sequence length")
    parser.add_argument("--micro-token-budget", type=int, default=32768,
                        help="Max padded tokens per micro-batch")
    parser.add_argument("--max-completion-tokens", type=int, default=1024,
                        help="Max completion tokens per rollout")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Rollout concurrency")

    # OPD-specific configuration
    parser.add_argument("--kl-coeff", type=float, default=1.0,
                        help="Coefficient for reverse KL loss")
    parser.add_argument("--length-normalize", action="store_true",
                        help="Normalize loss by sequence length")

    # System prompt
    parser.add_argument("--system-prompt", type=str,
                        default="First, think step by step. Then put your final answer inside \\boxed{...}.")

    args = parser.parse_args()

    # Load training data
    print(f"Loading GSM8K {args.split} split...")
    train_samples = load_gsm8k(args.split, args.limit)
    if not train_samples:
        raise SystemExit("No GSM8K samples loaded.")
    print(f"Loaded {len(train_samples)} training samples")

    # Create sample queue
    samples_q: queue.Queue = queue.Queue()
    for idx, s in enumerate(train_samples):
        samples_q.put((idx, s))

    # Load tokenizer and model
    print(f"Loading student model: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Create vLLM clients
    from ludic.inference import VLLMChatClient
    from ludic.distributed.adapters import create_vllm_publisher

    # Student client for sampling
    client = VLLMChatClient(
        host=args.student_host,
        port=args.student_port,
        enable_weight_updates=True,
    )
    publisher = create_vllm_publisher(client)

    # Teacher scorer - computes per-token logprobs during Agent.act()
    teacher_scorer = make_vllm_teacher_scorer(
        base_url=f"http://{args.teacher_host}:{args.teacher_port}",
        model=args.teacher_model,
    )

    chat_template = HFChatTemplate(tokenizer)

    # Environment and protocol registries
    env_registry = {
        "gsm8k": lambda sample: GSM8KEnv(
            sample=sample,
            system_prompt=args.system_prompt,
        )
    }

    # Agent with teacher scorer - scores flow through to training
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

    # Create OPD algorithm
    algo = make_opd(
        kl_coeff=args.kl_coeff,
        length_normalize=args.length_normalize,
        name="opd",
    )

    # Create rollout engine
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path="opd_rollouts.jsonl",
    )

    # Create inference spec
    train_inference = InferenceSpec(
        sampling=SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_completion_tokens,
        ),
        return_=ReturnSpec.for_rl(top_logprobs_k=1),
    )

    # Create requests function
    requests_fn = make_dataset_queue_requests_fn(
        samples_q,
        batch_size=args.rollouts_per_update,
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
    )

    # Create batch source (no teacher_client needed - it's in the Agent!)
    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=algo.credit_assigner,
        requests_fn=requests_fn,
        max_steps=1,
        concurrency=args.concurrency,
    )

    # Trainer config
    cfg = TrainerConfig(
        model_device=device,
        max_seq_len=args.max_seq_len,
        micro_token_budget=args.micro_token_budget,
        max_grad_norm=0.5,
        pad_token_id=tokenizer,
    )

    # Reducers for logging
    reducers = {
        **default_reducers(),
        "correct_rate": Reducer(
            kind="count_true",
            source="correct",
            normalize_by="rollouts",
        ),
    }

    # Logger
    logger_keys = [
        "train/loss",
        "train/reverse_kl_mean",
        "train/avg_total_reward",
        "train/correct_rate",
        "train/avg_completion_length",
        "train/num_samples",
    ]
    train_logger = PrintLogger(prefix="[opd]", keys=logger_keys, precision=4)

    # Create trainer
    trainer = Trainer(
        model=model,
        algo=algo,
        batch_source=batch_source,
        publisher=publisher,
        enable_gradient_checkpointing=True,
        cfg=cfg,
        train_logger=train_logger,
        reducers=reducers,
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

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
