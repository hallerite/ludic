#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import fsdp

from ludic.training.algorithm import make_sft
from ludic.training.batching.micro_batching import collate_saw_items, split_items_by_token_budget
from ludic.training.types import SAWBatch, SAWItem


@dataclass
class TrialResult:
    ok: bool
    elapsed_s: float
    peak_mem_mb: float
    micro_batches: int
    padded_tokens: int
    error: Optional[str] = None


def _parse_token_budgets(arg: Optional[str]) -> List[int]:
    if not arg:
        return []
    out: List[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _run_budget_trials(
    *,
    budget: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    algo,
    device: torch.device,
    pad_token_id: int,
    max_seq_len: int,
    saw_batch: SAWBatch,
    warmup_steps: int,
    steps: int,
    use_grad_scaler: bool,
    scaler: Optional[torch.amp.GradScaler],
) -> tuple[List[TrialResult], bool, Optional[TrialResult]]:
    results: List[TrialResult] = []
    micro_chunks = split_items_by_token_budget(
        saw_batch.items,
        micro_token_budget=budget,
        max_seq_len=max_seq_len,
    )
    last_res: Optional[TrialResult] = None
    for _ in range(warmup_steps):
        last_res = _run_step(
            model=model,
            optimizer=optimizer,
            algo=algo,
            device=device,
            pad_token_id=pad_token_id,
            micro_token_budget=budget,
            max_seq_len=max_seq_len,
            saw_batch=saw_batch,
            use_grad_scaler=use_grad_scaler,
            micro_chunks=micro_chunks,
            scaler=scaler,
        )
        results.append(last_res)
        if not last_res.ok:
            return results, False, last_res
    for _ in range(steps):
        last_res = _run_step(
            model=model,
            optimizer=optimizer,
            algo=algo,
            device=device,
            pad_token_id=pad_token_id,
            micro_token_budget=budget,
            max_seq_len=max_seq_len,
            saw_batch=saw_batch,
            use_grad_scaler=use_grad_scaler,
            micro_chunks=micro_chunks,
            scaler=scaler,
        )
        results.append(last_res)
        if not last_res.ok:
            return results, False, last_res
    return results, True, last_res


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def _init_dist(local_rank: int) -> int:
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" not in os.environ:
        raise ValueError("FSDP requires torchrun; missing RANK in env.")
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


def _maybe_enable_gradient_checkpointing(model: nn.Module) -> None:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def _load_model(
    *,
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    fsdp_enabled: bool,
    fsdp_param_dtype: torch.dtype,
    fsdp_reduce_dtype: torch.dtype,
    fsdp_per_layer: bool,
    gradient_checkpointing: bool,
    lora_path: Optional[str],
    lora_rank: int,
    lora_alpha_mult: float,
    lora_dropout: float,
    lora_target_modules: str,
) -> nn.Module:
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
    except ImportError as exc:
        raise ImportError("transformers is required for calibration.") from exc

    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if fsdp_enabled:
        load_kwargs["device_map"] = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    if lora_path or lora_rank > 0:
        try:
            from peft import PeftModel, LoraConfig, TaskType, get_peft_model  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA calibration. Install with: uv sync --extra examples"
            ) from exc
        if lora_path:
            model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=int(lora_rank * lora_alpha_mult),
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=lora_target_modules,
            )
            model = get_peft_model(model, lora_config)

    if gradient_checkpointing:
        _maybe_enable_gradient_checkpointing(model)

    if fsdp_enabled:
        mp_policy = fsdp.MixedPrecisionPolicy(
            param_dtype=fsdp_param_dtype,
            reduce_dtype=fsdp_reduce_dtype,
        )
        blocks = None
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            blocks = model.model.layers  # type: ignore[attr-defined]
        elif hasattr(model, "layers"):
            blocks = model.layers  # type: ignore[attr-defined]
        if fsdp_per_layer and blocks is not None:
            for layer in blocks:
                fsdp.fully_shard(layer, mp_policy=mp_policy)
        fsdp.fully_shard(model, mp_policy=mp_policy)
        return model

    return model.to(device)


def _build_batch(
    *,
    target_rollouts: int,
    steps_per_rollout: int,
    seq_len: int,
    min_seq_len: int,
    vocab_size: int,
    action_ratio: float,
    seed: int,
) -> SAWBatch:
    g = torch.Generator()
    g.manual_seed(seed)

    items: List[SAWItem] = []
    for rollout_idx in range(target_rollouts):
        for step_idx in range(steps_per_rollout):
            if min_seq_len >= seq_len:
                length = seq_len
            else:
                length = int(torch.randint(min_seq_len, seq_len + 1, (1,), generator=g).item())
            action_len = max(1, int(round(length * action_ratio)))
            prompt_len = max(0, length - action_len)
            input_ids = torch.randint(0, vocab_size, (length,), generator=g).tolist()
            attention_mask = [1] * length
            action_mask = [0] * prompt_len + [1] * action_len
            items.append(
                SAWItem(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    weight=1.0,
                    meta={
                        "rollout_id": f"r{rollout_idx}",
                        "step_index": step_idx,
                        "completion_length": action_len,
                        "prompt_length": prompt_len,
                    },
                )
            )

    avg_completion = sum(it.meta["completion_length"] for it in items) / len(items)
    avg_prompt = sum(it.meta["prompt_length"] for it in items) / len(items)
    meta = {
        "target_rollouts": target_rollouts,
        "num_samples": len(items),
        "avg_completion_length": avg_completion,
        "avg_prompt_length": avg_prompt,
    }
    return SAWBatch(items=items, meta=meta)


def _run_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    algo,
    device: torch.device,
    pad_token_id: int,
    micro_token_budget: int,
    max_seq_len: int,
    saw_batch: SAWBatch,
    use_grad_scaler: bool,
    micro_chunks: Optional[List[List[SAWItem]]] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> TrialResult:
    model.train()
    if micro_chunks is None:
        micro_chunks = split_items_by_token_budget(
            saw_batch.items,
            micro_token_budget=micro_token_budget,
            max_seq_len=max_seq_len,
        )
    total_items = sum(len(chunk) for chunk in micro_chunks)
    if total_items == 0:
        raise ValueError("Synthetic batch produced no items.")

    if use_grad_scaler and scaler is None:
        scaler = torch.amp.GradScaler(enabled=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    try:
        for idx, chunk in enumerate(micro_chunks):
            is_last = idx == len(micro_chunks) - 1
            grad_sync_disabled = False
            if isinstance(model, fsdp.FSDPModule) and not is_last:
                model.set_requires_gradient_sync(False)
                grad_sync_disabled = True
            try:
                batch = collate_saw_items(
                    chunk,
                    pad_token_id=pad_token_id,
                    device=device,
                )
                loss, _stats = algo.compute_loss(model, batch)
                scaled = loss * (len(chunk) / total_items)
                if use_grad_scaler:
                    assert scaler is not None
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
            finally:
                if grad_sync_disabled and isinstance(model, fsdp.FSDPModule):
                    model.set_requires_gradient_sync(True)

        if use_grad_scaler:
            assert scaler is not None
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return TrialResult(
                ok=False,
                elapsed_s=0.0,
                peak_mem_mb=0.0,
                micro_batches=len(micro_chunks),
                padded_tokens=micro_token_budget,
                error=str(exc),
            )
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return TrialResult(
        ok=True,
        elapsed_s=elapsed,
        peak_mem_mb=peak_mb,
        micro_batches=len(micro_chunks),
        padded_tokens=micro_token_budget,
    )


def _run_trials(
    budgets: Iterable[int],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    algo,
    device: torch.device,
    pad_token_id: int,
    max_seq_len: int,
    saw_batch: SAWBatch,
    warmup_steps: int,
    steps: int,
    use_grad_scaler: bool,
    scaler: Optional[torch.amp.GradScaler],
) -> List[TrialResult]:
    results: List[TrialResult] = []
    for budget in budgets:
        micro_chunks = split_items_by_token_budget(
            saw_batch.items,
            micro_token_budget=budget,
            max_seq_len=max_seq_len,
        )
        for _ in range(warmup_steps):
            _run_step(
                model=model,
                optimizer=optimizer,
                algo=algo,
                device=device,
                pad_token_id=pad_token_id,
                micro_token_budget=budget,
                max_seq_len=max_seq_len,
                saw_batch=saw_batch,
                use_grad_scaler=use_grad_scaler,
                micro_chunks=micro_chunks,
                scaler=scaler,
            )
        for _ in range(steps):
            results.append(
                _run_step(
                    model=model,
                    optimizer=optimizer,
                    algo=algo,
                    device=device,
                    pad_token_id=pad_token_id,
                    micro_token_budget=budget,
                    max_seq_len=max_seq_len,
                    saw_batch=saw_batch,
                    use_grad_scaler=use_grad_scaler,
                    micro_chunks=micro_chunks,
                    scaler=scaler,
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate micro-batch token budget with a real model/FSDP.",
    )
    parser.add_argument("--model", type=str, required=True, help="HF model path or id.")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g., cuda:0).")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max tokens per sample.")
    parser.add_argument("--micro-token-budget", type=int, default=None, help="Max padded tokens per micro-batch.")
    parser.add_argument("--token-budgets", type=str, default=None, help="Comma-separated token budgets.")
    parser.add_argument("--min-seq-len", type=int, default=None, help="Min tokens per sample.")
    parser.add_argument("--target-rollouts", type=int, default=8, help="Target rollouts per synthetic batch.")
    parser.add_argument("--steps-per-rollout", type=int, default=1)
    parser.add_argument("--action-ratio", type=float, default=0.5, help="Fraction of action tokens.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--pad-token-id", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--auto-token-budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-sweep micro-token budgets (starts at max_seq_len, doubles each step, "
            "then binary-searches the last OK/first OOM range)."
        ),
    )
    parser.add_argument("--lora-path", type=str, default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--lora-rank", type=int, default=0, help="LoRA rank (0 disables).")
    parser.add_argument(
        "--lora-alpha-mult",
        type=float,
        default=2.0,
        help="Multiplier applied to rank to set lora_alpha (alpha = rank * mult).",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout probability.")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="all-linear",
        help="Modules to target for LoRA (e.g., all-linear).",
    )
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--fsdp-param-dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--fsdp-reduce-dtype", choices=["bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument(
        "--fsdp-per-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard transformer blocks individually when available.",
    )
    parser.add_argument("--local-rank", type=int, default=None)
    args = parser.parse_args()

    token_budgets = _parse_token_budgets(args.token_budgets)

    if args.target_rollouts <= 0:
        raise ValueError("--target-rollouts must be > 0.")
    if args.steps_per_rollout <= 0:
        raise ValueError("--steps-per-rollout must be > 0.")
    if args.action_ratio <= 0 or args.action_ratio > 1:
        raise ValueError("--action-ratio must be in (0, 1].")
    if args.lora_path and args.lora_rank > 0:
        raise ValueError("Pass either --lora-path or --lora-rank, not both.")
    if args.lora_rank < 0:
        raise ValueError("--lora-rank must be >= 0.")

    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    device_str = args.device
    if device_str is None:
        device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    dtype = _resolve_dtype(args.dtype)
    fsdp_param_dtype = _resolve_dtype(args.fsdp_param_dtype) if args.fsdp_param_dtype else dtype
    fsdp_reduce_dtype = _resolve_dtype(args.fsdp_reduce_dtype)

    rank = 0
    if args.fsdp:
        rank = _init_dist(local_rank)

    model = _load_model(
        model_id=args.model,
        device=device,
        dtype=dtype,
        fsdp_enabled=args.fsdp,
        fsdp_param_dtype=fsdp_param_dtype,
        fsdp_reduce_dtype=fsdp_reduce_dtype,
        fsdp_per_layer=args.fsdp_per_layer,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        lora_alpha_mult=args.lora_alpha_mult,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    config = getattr(model, "config", None)
    vocab_size = int(getattr(config, "vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError("Model config missing vocab_size.")

    max_pos = getattr(config, "max_position_embeddings", None)
    if max_pos is not None and args.max_seq_len > int(max_pos):
        raise ValueError(f"max_seq_len {args.max_seq_len} exceeds model max_position_embeddings {max_pos}.")

    pad_token_id = args.pad_token_id
    if pad_token_id is None:
        pad_token_id = int(getattr(config, "pad_token_id", 0) or 0)

    min_seq_len = args.min_seq_len if args.min_seq_len is not None else args.max_seq_len
    if min_seq_len > args.max_seq_len:
        raise ValueError("min-seq-len cannot exceed max-seq-len.")

    algo = make_sft(length_normalize=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    use_grad_scaler = args.dtype == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_grad_scaler) if use_grad_scaler else None

    saw_batch = _build_batch(
        target_rollouts=args.target_rollouts,
        steps_per_rollout=args.steps_per_rollout,
        seq_len=args.max_seq_len,
        min_seq_len=min_seq_len,
        vocab_size=vocab_size,
        action_ratio=args.action_ratio,
        seed=args.seed,
    )

    auto_mode = False
    if token_budgets:
        budgets = token_budgets
    elif args.micro_token_budget is not None:
        if args.micro_token_budget <= 0:
            raise ValueError("--micro-token-budget must be > 0.")
        budgets = [args.micro_token_budget]
    elif args.auto_token_budget:
        auto_mode = True
        budgets = []
    else:
        raise ValueError("Pass --micro-token-budget, --token-budgets, or enable --auto-token-budget.")

    if auto_mode:
        results = []
        min_ok_pow2 = None
        max_ok = None
        first_oom_budget = None
        oom_error = None
        oom_micro_batches = None

        max_budget = args.max_seq_len * len(saw_batch.items)
        budget = args.max_seq_len
        last_ok_budget = None
        last_ok_micro_batches = None

        while budget <= max_budget:
            budget_results, ok, last_res = _run_budget_trials(
                budget=budget,
                model=model,
                optimizer=optimizer,
                algo=algo,
                device=device,
                pad_token_id=pad_token_id,
                max_seq_len=args.max_seq_len,
                saw_batch=saw_batch,
                warmup_steps=args.warmup_steps,
                steps=args.steps,
                use_grad_scaler=use_grad_scaler,
                scaler=scaler,
            )
            results.extend(budget_results)
            if ok:
                if min_ok_pow2 is None:
                    min_ok_pow2 = budget
                last_ok_budget = budget
                if last_res is not None:
                    last_ok_micro_batches = last_res.micro_batches
                    if last_ok_micro_batches == 1:
                        max_ok = budget
                        break
                if budget >= max_budget:
                    max_ok = budget
                    break
                next_budget = min(budget * 2, max_budget)
                if next_budget == budget:
                    max_ok = budget
                    break
                budget = next_budget
                continue

            first_oom_budget = budget
            if last_res is not None:
                oom_error = last_res.error
                oom_micro_batches = last_res.micro_batches
            break

        if last_ok_budget is not None and first_oom_budget is not None:
            if last_ok_micro_batches == 1 or (
                oom_micro_batches is not None and last_ok_micro_batches == oom_micro_batches
            ):
                max_ok = last_ok_budget
            else:
                low = last_ok_budget
                high = first_oom_budget
                while high - low > 1:
                    mid = (low + high) // 2
                    budget_results, ok, last_res = _run_budget_trials(
                        budget=mid,
                        model=model,
                        optimizer=optimizer,
                        algo=algo,
                        device=device,
                        pad_token_id=pad_token_id,
                        max_seq_len=args.max_seq_len,
                        saw_batch=saw_batch,
                        warmup_steps=args.warmup_steps,
                        steps=args.steps,
                        use_grad_scaler=use_grad_scaler,
                        scaler=scaler,
                    )
                    results.extend(budget_results)
                    if ok:
                        low = mid
                    else:
                        high = mid
                        if oom_error is None and last_res is not None:
                            oom_error = last_res.error
                max_ok = low
        elif last_ok_budget is not None and max_ok is None:
            max_ok = last_ok_budget

    else:
        results = _run_trials(
            budgets,
            model=model,
            optimizer=optimizer,
            algo=algo,
            device=device,
            pad_token_id=pad_token_id,
            max_seq_len=args.max_seq_len,
            saw_batch=saw_batch,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
            use_grad_scaler=use_grad_scaler,
            scaler=scaler,
        )

    if not args.fsdp or rank == 0:
        for res in results:
            status = "ok" if res.ok else "oom"
            msg = (
                f"status={status} micro_token_budget={res.padded_tokens} "
                f"micro_batches={res.micro_batches} time_s={res.elapsed_s:.4f} "
                f"peak_mb={res.peak_mem_mb:.1f}"
            )
            if res.error:
                msg += f" error={res.error}"
            print(msg)
        if auto_mode:
            if min_ok_pow2 is not None:
                print(f"min_micro_token_budget_ok_pow2={min_ok_pow2}")
            if max_ok is not None:
                print(f"max_micro_token_budget_ok={max_ok}")
            if first_oom_budget is not None:
                print(f"oom_micro_token_budget={first_oom_budget}")
            if oom_error:
                print(f"oom_error={oom_error}")

    if args.fsdp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
