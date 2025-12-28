"""
SFT training for Tic-Tac-Toe using offline data.

This script trains on pre-generated data (from generate_synth_data.py) to
bootstrap a model before RL training. The workflow is:

1. generate_synth_data.py → data/tictactoe_sft_data.jsonl
2. sft_tic_tac_toe.py (this script) → checkpoints/
3. train_tic_tac_toe.py with TruncatedThinkingContext (RL refinement)

The SFT phase teaches the model:
- The <think>...</think><move>X</move> format
- The [TRUNCATED THOUGHTS] convention in history
- Basic Tic-Tac-Toe strategy from winning examples
Use --ctx full to train on full (untruncated) history prompts.

This variant runs full-finetune FSDP2 over 2 GPUs (torchrun). No eval, no LoRA, no vLLM; it just prints loss.
Launch with:
  torchrun --nproc_per_node=2 examples/tic_tac_toe/sft_tic_tac_toe.py --data data/tictactoe_sft_data.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed import fsdp
from transformers import AutoModelForCausalLM, AutoTokenizer

from ludic.types import Step
from ludic.training import (
    OfflineBatchSource,
    Trainer,
    TrainerConfig,
    CheckpointConfig,
    make_sft,
    make_chat_template_step_to_item,
)


# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

STRICT_THINK_PATTERN = re.compile(
    r"^(\s*<think>)(.*?)(</think>\s*)(.+)$",
    flags=re.DOTALL | re.IGNORECASE,
)


def _truncate_assistant_text(text: str, placeholder: str) -> str:
    m = STRICT_THINK_PATTERN.match(text)
    if not m:
        return text
    return f"{m.group(1)}{placeholder}{m.group(3)}{m.group(4)}"


def _truncate_history_messages(
    messages: list[dict[str, str]],
    placeholder: str,
) -> list[dict[str, str]]:
    truncated: list[dict[str, str]] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            new_msg = dict(msg)
            new_msg["content"] = _truncate_assistant_text(content, placeholder)
            truncated.append(new_msg)
        else:
            truncated.append(msg)
    return truncated


def _with_truncated_history(step: Step, placeholder: str) -> Step:
    info = step.info or {}
    full_messages = info.get("chat_prompt_messages_full")
    if full_messages:
        truncated = _truncate_history_messages(full_messages, placeholder)
    else:
        chat_messages = info.get("chat_prompt_messages")
        if not chat_messages:
            return step
        truncated = _truncate_history_messages(chat_messages, placeholder)
    new_info = dict(info)
    new_info["chat_prompt_messages"] = truncated
    return Step(
        index=step.index,
        prev_obs=step.prev_obs,
        action=step.action,
        next_obs=step.next_obs,
        reward=step.reward,
        truncated=step.truncated,
        terminated=step.terminated,
        info=new_info,
        trace=step.trace,
        ts_ns=step.ts_ns,
    )


def _with_full_history(step: Step) -> Step:
    info = step.info or {}
    full_messages = info.get("chat_prompt_messages_full")
    if not full_messages:
        return step
    new_info = dict(info)
    new_info["chat_prompt_messages"] = full_messages
    return Step(
        index=step.index,
        prev_obs=step.prev_obs,
        action=step.action,
        next_obs=step.next_obs,
        reward=step.reward,
        truncated=step.truncated,
        terminated=step.terminated,
        info=new_info,
        trace=step.trace,
        ts_ns=step.ts_ns,
    )


def init_dist(local_rank: int) -> int:
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
        dist.init_process_group(backend="gloo", init_method="env://")
    return dist.get_rank()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="FSDP2 SFT for Tic-Tac-Toe (offline).")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="data/tictactoe_sft_train_data.jsonl")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs over the dataset.")
    parser.add_argument("--batch-size", type=int, default=8, help="Samples per batch per rank.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max tokens per sample.")
    parser.add_argument("--micro-token-budget", type=int, default=8192, help="Max padded tokens per micro-batch.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", default="checkpoints_tictactoe_fsdp2")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="0 disables checkpoints.")
    parser.add_argument("--max-to-keep", type=int, default=2, help="Max checkpoints to keep (None=keep all).")
    parser.add_argument("--final-save", action=argparse.BooleanOptionalAction, default=True,
                        help="Save a final checkpoint at the end of training.")
    parser.add_argument("--log-every", type=int, default=1, help="Print every N trainer steps.")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable HF gradient checkpointing.",
    )
    parser.add_argument(
        "--ctx",
        choices=["full", "truncated"],
        default="truncated",
        help="Prompt history format for SFT: full or TruncatedThinking-style.",
    )
    parser.add_argument(
        "--placeholder",
        default="[TRUNCATED]",
        help="Placeholder token for truncated <think> blocks.",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = init_dist(local_rank)
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    data_path = Path(args.data)
    if not data_path.exists():
        if rank == 0:
            print(f"Error: data file not found: {data_path}")
            print("Run generate_synth_data.py first.")
        dist.destroy_process_group()
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    mp_policy = fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
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

    algo = make_sft(length_normalize=True)

    # Create step_to_item function with chat template preprocessing
    base_step_to_item = make_chat_template_step_to_item(tokenizer)
    if args.ctx == "truncated":
        def step_to_item(rollout, step, weight):
            truncated_step = _with_truncated_history(step, args.placeholder)
            return base_step_to_item(rollout, truncated_step, weight)
    else:
        def step_to_item(rollout, step, weight):
            full_step = _with_full_history(step)
            return base_step_to_item(rollout, full_step, weight)

    batch_source = OfflineBatchSource(
        jsonl_paths=[data_path],
        step_to_item=step_to_item,
        credit_assigner=algo.credit_assigner,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Calculate training steps correctly:
    # - Each train_step() consumes one macro-batch
    # - We want exactly `epochs` passes through the data
    batches_per_epoch = batch_source.num_batches_per_epoch
    batches_needed = args.epochs * batches_per_epoch
    total_steps = batches_needed
    effective_samples_per_step = args.batch_size * world_size

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Loaded {len(batch_source)} samples from {data_path}")
        print(f"Batches per epoch (per rank): {batches_per_epoch}")
        print(f"Total batches needed: {batches_needed} ({args.epochs} epochs)")
        print(f"Total trainer steps: {total_steps} (macro-batches)")
        print(f"Effective samples per step: {effective_samples_per_step} (batch_size * world_size)")

    cfg = TrainerConfig(
        model_device=str(device),
        max_seq_len=args.max_seq_len,
        micro_token_budget=args.micro_token_budget,
        max_grad_norm=args.max_grad_norm,
        pad_token_id=tokenizer,
        lr=args.lr,
        reduce_stats_across_ranks=True,
        eval_at_start=False,
        eval_every_n_steps=None,
        sync_every_steps=0,  # disable publishing/sync; local-only training
    )

    checkpoint_cfg = None
    if args.checkpoint_every > 0:
        checkpoint_cfg = CheckpointConfig(
            output_dir=args.checkpoint_dir,
            every_n_steps=args.checkpoint_every,
            max_to_keep=args.max_to_keep,
            save_optimizer=True,
        )
        if rank == 0:
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        dist.barrier()

    trainer = Trainer(
        model=model,
        algo=algo,
        batch_source=batch_source,
        publisher=None,
        cfg=cfg,
        checkpoint_config=checkpoint_cfg,
        enable_gradient_checkpointing=bool(args.gradient_checkpointing),
        train_logger=None,
        evaluator=None,
    )

    async def train_loop() -> None:
        for step in range(total_steps):
            stats = await trainer.train_step()
            if rank == 0 and (step % args.log_every == 0):
                loss_val = stats.get("train/loss")
                loss_str = f"{loss_val:.4f}" if loss_val is not None else "n/a"
                logp_val = stats.get("train/logp_mean")
                logp_str = f"{logp_val:.4f}" if logp_val is not None else "n/a"
                print(
                    f"[step {step + 1}/{total_steps}] loss={loss_str} logp_mean={logp_str} "
                    f"(effective_samples={effective_samples_per_step})",
                    flush=True,
                )

    asyncio.run(train_loop())

    if rank == 0:
        print("Training complete.")
        if args.final_save:
            try:
                ckpt_path = trainer.save_checkpoint()
                print(f"Final checkpoint saved to: {ckpt_path}")
            except RuntimeError:
                pass  # No checkpointer configured
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
