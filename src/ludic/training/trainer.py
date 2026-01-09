from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
from torch import nn, optim, Tensor
import torch.distributed as dist
from torch.distributed import fsdp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)

from ludic.distributed.interfaces import PolicyPublisher
from ludic.training.checkpoint import CheckpointConfig, CheckpointManager
from ludic.training.algorithm import RLAlgorithm
from ludic.training.loggers import TrainingLogger
from ludic.training.config import TrainerConfig
from ludic.training.batching.micro_batching import collate_saw_items, split_items_by_token_budget
from ludic.training.stats import aggregate_stats, Reducer
from ludic.eval.evaluator import Evaluator
from ludic.training.types import SAWBatch, BatchSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Orchestrates the training loop with gradient accumulation:

        BatchSource.next_batch() → macro-batch
          ↓
        bucket + split (token budget) → micro-batches
          ↓
        collate (per micro) → tensors
          ↓
        RLAlgorithm.compute_loss(model, batch)
          ↓
        scaled_loss.backward() (with gradient sync disabled on non-final micros for FSDP)
      ↓
    optimizer.step()
      ↓
    optimizer.zero_grad()  # <-- Grads freed
      ↓
    publisher.publish(...)     # <-- Sync

    Trainer is agnostic to envs, contexts, rollouts, and tokenization.

    This variant is FSDP2-aware:

      - If `model` is wrapped in FSDP2 (fully_shard), it will:
          * disable gradient sync on non-final micro-batches
          * gather a full state dict on rank 0 via DCP APIs
          * push the full (unsharded) params to the runtime

      - On non-FSDP models, it just uses `named_parameters()` as before.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        algo: RLAlgorithm,
        batch_source: BatchSource,
        cfg: TrainerConfig,
        publisher: Optional[PolicyPublisher] = None,
        param_filter: Optional[Callable[[str, Tensor], bool]] = None,
        enable_gradient_checkpointing: bool = False,
        checkpointer: Optional[CheckpointManager] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        resume_from: Optional[int | str] = None,
        train_logger: Optional[TrainingLogger] = None,
        reducers: Optional[Mapping[str, Reducer]] = None,
        evaluator: Optional[Evaluator] = None,
    ) -> None:
        """
        Args:
            model:
                Trainable policy model (typically a HF CausalLM-like module).
                Must accept (input_ids, attention_mask) and expose .logits.
                May be wrapped in FSDP.

            algo:
                RLAlgorithm = (CreditAssigner + Loss).

            batch_source:
                Any object implementing BatchSource.next_batch() -> SAWBatch.
                The SAWBatch is treated as a macro-batch and split into
                micro-batches for gradient accumulation.

            cfg:
                TrainerConfig for device, optimizer hyperparams, pad_token_id,
                micro_token_budget, max_seq_len, and sync_every_steps.

            publisher:
                Abstract interface to push weights to inference workers. If None, weight
                syncing is disabled.

            param_filter:
                Optional predicate (name, Tensor) -> bool deciding which
                parameters get pushed into the runtime.

            enable_gradient_checkpointing:
                If True, enables activation checkpointing on the model to save VRAM.
                Also automatically disables KV cache and enables input gradients.

            checkpointer:
                Optional CheckpointManager for periodic HF-format saves. If not
                provided, one will be constructed from `checkpoint_config`.
                Prefer `checkpoint_config` for standard usage; `checkpointer`
                is an escape hatch to inject a custom implementation.

            checkpoint_config:
                Convenience config to build a default CheckpointManager.

            resume_from:
                Optional checkpoint to load on startup. Accepts a step number
                or an explicit checkpoint directory path. Requires a
                checkpointer (explicit or via checkpoint_config).

            train_logger:
                Optional stats logger (e.g., PrintLogger, WandbLogger). Called
                once per train_step with (step, stats).

            reducers:
                Optional aggregation reducers passed to aggregate_stats to
                compute custom metrics from SAWItem meta.

            evaluator:
                Optional evaluator object used for periodic evaluation runs.
        """
        self.cfg = cfg
        self.train_logger = train_logger
        self.reducers = reducers

        # Assume caller has already done any FSDP wrapping / device placement.
        # We do NOT unconditionally .to(device) for FSDP; that’s the caller’s job.
        self.model = model.to(cfg.model_device) if not isinstance(model, fsdp.FSDPModule) else model  # type: ignore[arg-type]

        self.algo = algo
        self.publisher = publisher
        self._batch_source = batch_source
        self.sync_every_steps = (
            self.cfg.sync_every_steps if self.cfg.sync_every_steps and self.cfg.sync_every_steps > 0 else None
        )
        self.param_filter = param_filter
        self._train_step_idx = 0
        self._last_train_stats: Dict[str, float] = {}
        self._last_eval_stats: Dict[str, float] = {}
        self._checkpointer = checkpointer or (
            CheckpointManager(checkpoint_config) if checkpoint_config is not None else None
        )
        self._resume_from = resume_from
        self.evaluator = evaluator
        self._validate_invariants()

        # ---- Gradient Checkpointing Setup ----------------------------
        if enable_gradient_checkpointing:
            logger.info("🛡️ Enabling Gradient Checkpointing (Activation Checkpointing)...")

            # 1. Enable on the model (HuggingFace standard API)
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                logger.warning("⚠️ Model does not have 'gradient_checkpointing_enable' method.")

            # 2. Disable KV Cache (incompatible with checkpointing + training)
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False

            # 3. Enable input gradients
            # This is often required so that gradients can flow through the checkpointed segments.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            else:
                # Fallback for generic torch modules if needed, though usually HF specific
                pass
        # --------------------------------------------------------------

        # Initialize optimizer
        self.optimizer = self.initialize_optimizer()

        # Assume gradients are zeroed at init
        self.optimizer.zero_grad(set_to_none=True)

        # Optionally resume from a checkpoint (weights + optimizer)
        if resume_from is not None:
            if self._checkpointer is None:
                raise ValueError("resume_from requires a CheckpointManager or checkpoint_config")
            load_kwargs: Dict[str, Optional[object]] = {}
            if isinstance(resume_from, int):
                load_kwargs["step"] = resume_from
            elif isinstance(resume_from, str):
                # Prefer treating a real path as explicit target; fall back to int-like strings
                if Path(resume_from).exists():
                    load_kwargs["path"] = resume_from
                elif resume_from.isdigit():
                    load_kwargs["step"] = int(resume_from)
            meta = self._checkpointer.load(
                self.model,
                optimizer=self.optimizer,
                step=load_kwargs.get("step"),  # type: ignore[arg-type]
                path=load_kwargs.get("path"),  # type: ignore[arg-type]
            )
            # Default to the saved step, else keep current.
            self._train_step_idx = int(meta.get("step", self._train_step_idx))
            logger.info("✅ Resumed from checkpoint at step %s", self._train_step_idx)

    # ------------------------------------------------------------------
    # Optimizer initialization
    # ------------------------------------------------------------------

    def initialize_optimizer(self) -> optim.Optimizer:
        """
        Initialize and return the optimizer.

        - We create AdamW over all model parameters with hyperparams
          from TrainerConfig:

                lr         = cfg.lr
                weight_decay= cfg.weight_decay
                betas       = cfg.betas
                eps         = cfg.eps
        """
        return optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
        )

    def _maybe_reduce_stats_across_ranks(
        self,
        stats: Dict[str, float],
        *,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        All-reduce stats across distributed ranks.

        Implementation notes:
        1. We batch all stats into a single tensor per reduction op (SUM or MAX)
           rather than calling all_reduce on individual scalars. NCCL has known
           issues with 0-dim tensors that can produce garbage values.

        2. We sort keys alphabetically before building the tensor to ensure
           consistent ordering across ranks. Different ranks may have different
           dict iteration orders, which would cause value misalignment in the
           all_reduce if not sorted.

        Stats handling:
        - gpu_* keys: reduced with MAX (peak memory stats)
        - sum_keys (num_samples, etc.): reduced with SUM (counts)
        - all other keys: reduced with SUM of (value * num_samples) then divided
          by total num_samples, falling back to world_size when num_samples is missing.
        """
        if not self.cfg.reduce_stats_across_ranks:
            return stats
        if not (dist.is_available() and dist.is_initialized()):
            return stats

        world_size = dist.get_world_size()
        sum_keys = {
            "num_samples",
            "target_rollouts",
            "effective_rollouts",
            "total_completion_tokens",
        }
        has_num_samples = "num_samples" in stats
        local_samples = float(stats.get("num_samples", 0.0)) if has_num_samples else 1.0
        reduced: Dict[str, float] = {}

        # Separate keys into gpu (MAX) and non-gpu (SUM) groups
        # Sort keys to ensure consistent ordering across ranks
        gpu_keys = sorted([k for k in stats if k.startswith("gpu_") and isinstance(stats[k], (int, float))])
        mean_reduce_keys = sorted(
            [
                k for k in stats
                if (
                    not k.startswith("gpu_")
                    and k not in sum_keys
                    and isinstance(stats[k], (int, float))
                )
            ]
        )
        sum_reduce_keys = sorted(
            [
                k for k in stats
                if (k in sum_keys and isinstance(stats[k], (int, float)))
            ]
        )

        gpu_vals = [float(stats[k]) for k in gpu_keys]
        sum_reduce_vals = [float(stats[k]) for k in sum_reduce_keys]
        mean_reduce_vals = [float(stats[k]) * local_samples for k in mean_reduce_keys]

        # Batch all_reduce for SUM keys
        if sum_reduce_vals:
            t_sum = torch.tensor(sum_reduce_vals, dtype=torch.float32, device=device)
            dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
            sum_vals = t_sum.cpu().tolist()
            for k, v in zip(sum_reduce_keys, sum_vals):
                reduced[k] = v

        # Batch all_reduce for mean keys using sample-weighted sums.
        if mean_reduce_vals:
            t_mean = torch.tensor(mean_reduce_vals + [local_samples], dtype=torch.float32, device=device)
            dist.all_reduce(t_mean, op=dist.ReduceOp.SUM)
            mean_vals = t_mean.cpu().tolist()
            total_samples = mean_vals[-1]
            if has_num_samples:
                denom = total_samples if total_samples > 0 else float(world_size)
            else:
                denom = float(world_size)
            for k, v in zip(mean_reduce_keys, mean_vals[:-1]):
                reduced[k] = v / denom

        # Batch all_reduce for MAX keys (gpu stats)
        if gpu_vals:
            t_max = torch.tensor(gpu_vals, dtype=torch.float32, device=device)
            dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
            max_vals = t_max.cpu().tolist()
            for k, v in zip(gpu_keys, max_vals):
                reduced[k] = v

        # Preserve the local step counter
        reduced["train_step"] = float(self._train_step_idx)
        return reduced

    @staticmethod
    def _group_log_stats(stats: Dict[str, float]) -> Dict[str, float]:
        grouped: Dict[str, float] = {}
        for key, value in stats.items():
            if key == "phase":
                continue
            if key == "train_step":
                grouped["train/step"] = value
                continue
            if key == "eval_step":
                grouped["eval/step"] = value
                continue
            if key.startswith(("train/", "eval/", "perf/")):
                grouped[key] = value
            elif key.startswith("eval_"):
                grouped[f"eval/{key[5:]}"] = value
            elif key.startswith("gpu_"):
                grouped[f"perf/{key}"] = value
            else:
                grouped[f"train/{key}"] = value
        return grouped

    # ------------------------------------------------------------------
    # Memory utilities (GPU-only)
    # ------------------------------------------------------------------

    def _reset_peak_memory(self, device: torch.device) -> Optional[int]:
        """
        Reset CUDA peak stats and return current allocated bytes.
        """
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.memory_allocated(device)

    def _capture_forward_memory_stats(
        self,
        device: torch.device,
        pre_forward_alloc: Optional[int],
    ) -> tuple[Dict[str, float], Optional[int], Optional[int]]:
        """
        Capture forward-pass peak memory and activation peak.

        Returns: (stats_dict, alloc_after_forward_bytes, forward_peak_bytes)
        """
        if device.type != "cuda" or not torch.cuda.is_available():
            return {}, None, None

        torch.cuda.synchronize(device)
        mb = 1024 ** 2

        forward_peak = torch.cuda.max_memory_allocated(device)
        forward_activation_peak = max(0, forward_peak - (pre_forward_alloc or 0))
        alloc_after_forward = torch.cuda.memory_allocated(device)

        gpu_forward_peak_mb = forward_peak / mb
        gpu_forward_activation_peak_mb = forward_activation_peak / mb
        stats = {
            "gpu_forward_peak_mb": torch.tensor(gpu_forward_peak_mb, device=device),
            "gpu_forward_activation_peak_mb": torch.tensor(gpu_forward_activation_peak_mb, device=device),
        }

        logger.info(
            "GPU forward memory (MB) peak/activation_peak: %.1f / %.1f",
            gpu_forward_peak_mb,
            gpu_forward_activation_peak_mb,
        )

        # Reset for backward measurement
        torch.cuda.reset_peak_memory_stats(device)

        return stats, alloc_after_forward, forward_peak

    def _collect_memory_stats(
        self,
        device: torch.device,
        *,
        baseline_alloc: Optional[int],
    ) -> tuple[Dict[str, float], Optional[int]]:
        """
        Capture current/peak GPU memory and backward activation footprint.

        Returns: (stats_dict, peak_bytes)
        """
        if device.type != "cuda" or not torch.cuda.is_available():
            return {}, None

        torch.cuda.synchronize(device)
        mb = 1024 ** 2

        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
        backward_activation_peak = max(0, peak - (baseline_alloc or 0))

        gpu_mem_alloc_mb = alloc / mb
        gpu_mem_reserved_mb = reserved / mb
        gpu_backward_peak_mb = peak / mb
        gpu_backward_activation_peak_mb = backward_activation_peak / mb
        stats = {
            "gpu_mem_alloc_mb": torch.tensor(gpu_mem_alloc_mb, device=device),
            "gpu_mem_reserved_mb": torch.tensor(gpu_mem_reserved_mb, device=device),
            "gpu_backward_peak_mb": torch.tensor(gpu_backward_peak_mb, device=device),
            "gpu_backward_activation_peak_mb": torch.tensor(gpu_backward_activation_peak_mb, device=device),
        }

        logger.info(
            "GPU backward memory (MB) alloc/reserved/peak/activation_peak: %.1f / %.1f / %.1f / %.1f",
            gpu_mem_alloc_mb,
            gpu_mem_reserved_mb,
            gpu_backward_peak_mb,
            gpu_backward_activation_peak_mb,
        )

        return stats, peak

    # ------------------------------------------------------------------
    # Core async train step (now a "macro-step")
    # ------------------------------------------------------------------

    async def train_step(self) -> Dict[str, float]:
        """
        One full "macro" RL step with gradient accumulation:

            - Split the macro-batch into micro-batches using `cfg.micro_token_budget`
            - Accumulate gradients over those micro-batches
            - Perform one optimizer step
            - Free gradients
            - Optionally push updated params into runtime

        Returns:
            Aggregated stats dict from all micro-batches, grouped as
            train/*, eval/*, and perf/* keys for logging.
        """
        device = torch.device(self.cfg.model_device)
        micro_token_budget = int(self.cfg.micro_token_budget)
        max_seq_len = int(self.cfg.max_seq_len)
        profile_memory = bool(self.cfg.profile_memory)

        all_micro_stats: List[Dict[str, Tensor]] = []
        all_saw_batches: List[SAWBatch] = []

        # ---- 1) Fetch Macro-Batch ---------------------------------------

        # Note: Gradients are *not* zeroed here. They are accumulated
        # from the previous state (which should be zero).
        self.model.train()

        # For PipelineRL, we might receive stale data from the queue.
        # We loop until we get a batch containing at least one fresh item.
        while True:
            saw_batch = await self._batch_source.next_batch()

            # If configured, filter out items that exceed max_lag.
            if self.cfg.max_lag is not None:
                current_time = self._train_step_idx
                limit = self.cfg.max_lag

                fresh_items = []
                for item in saw_batch.items:
                    # Default to current_time (0 lag) if tag is missing
                    item_ver = item.meta.get("policy_version", current_time)
                    if (current_time - item_ver) <= limit:
                        fresh_items.append(item)

                # Update the batch with only fresh items
                saw_batch.items = fresh_items

            # Algorithm-specific preprocessing (CPU-side) before collation.
            if self.algo.preprocess is not None:
                saw_batch = self.algo.preprocess(saw_batch)

            # If the batch is empty (e.g. all stale), drop it and fetch another.
            if not saw_batch.items:
                continue

            rollout_ids = {
                item.meta.get("rollout_id")
                for item in saw_batch.items
                if item.meta.get("rollout_id") is not None
            }
            if rollout_ids:
                saw_batch.meta["effective_rollouts"] = len(rollout_ids)
            else:
                saw_batch.meta["effective_rollouts"] = len(saw_batch.items)

            # Batch has valid items, proceed to collation
            break

        all_saw_batches.append(saw_batch)

        micro_chunks = split_items_by_token_budget(
            saw_batch.items,
            micro_token_budget=micro_token_budget,
            max_seq_len=max_seq_len,
        )
        # Use the post-truncation items for stats aggregation.
        processed_items = [item for chunk in micro_chunks for item in chunk]
        saw_batch.items = processed_items
        total_items = len(processed_items)
        if total_items == 0:
            raise ValueError("Macro-batch contains no items after preprocessing.")

        # ---- 2) Accumulation Loop (Micro-Steps) -------------------------
        for micro_step_idx, chunk in enumerate(micro_chunks):
            item_count = len(chunk)
            logger.debug(
                "[Micro-step %s/%s] Processing %s SAWItems.",
                micro_step_idx + 1,
                len(micro_chunks),
                item_count,
            )

            # ---- 2a) Collate into tensors ------------------------------
            batch_tensors = collate_saw_items(
                chunk,
                pad_token_id=self.cfg.pad_token_id,
                device=device,
            )

            input_shape = batch_tensors["input_ids"].shape
            logger.debug(
                "    -> Collated Tensor Shape: %s (Batch=%s, SeqLen=%s)",
                input_shape,
                input_shape[0],
                input_shape[1],
            )

            # ---- 2b) FSDP2 gradient sync control ----------------------
            # We only sync (all-reduce) gradients on the *last* micro-batch
            is_last_micro = (micro_step_idx == len(micro_chunks) - 1)
            grad_sync_disabled = False
            if isinstance(self.model, fsdp.FSDPModule) and not is_last_micro:
                self.model.set_requires_gradient_sync(False)
                grad_sync_disabled = True

            # ---- 2c) Loss + backward (scaled) --------------------------
            pre_forward_alloc = self._reset_peak_memory(device) if profile_memory else None
            try:
                loss, stats = self.algo.compute_loss(
                    self.model,
                    batch_tensors,
                    cast_logits_to_fp32=self.cfg.cast_logits_to_fp32,
                )

                # Scale loss by micro-batch size to preserve macro-batch mean.
                scaled_loss = loss * (item_count / total_items)
                if profile_memory:
                    # Forward memory stats before backward frees activations
                    forward_mem_stats, alloc_after_forward, forward_peak = (
                        self._capture_forward_memory_stats(device, pre_forward_alloc)
                    )
                else:
                    forward_mem_stats, alloc_after_forward, forward_peak = {}, None, None

                scaled_loss.backward()
            finally:
                if grad_sync_disabled and isinstance(self.model, fsdp.FSDPModule):
                    self.model.set_requires_gradient_sync(True)

            # Attach memory stats (if available) for logging/aggregation
            stats = dict(stats)
            if profile_memory:
                backward_mem_stats, backward_peak = self._collect_memory_stats(
                    device,
                    baseline_alloc=alloc_after_forward,
                )
            else:
                backward_mem_stats, backward_peak = {}, None

            # Compute overall peak/activation across forward+backward
            if profile_memory and (forward_peak is not None or backward_peak is not None):
                mb = 1024 ** 2
                total_peak = max(
                    forward_peak or 0,
                    backward_peak or 0,
                )
                activation_total_peak = max(0, total_peak - (pre_forward_alloc or 0))

                stats["gpu_mem_peak_mb"] = torch.tensor(total_peak / mb, device=device)
                stats["gpu_activation_peak_mb"] = torch.tensor(activation_total_peak / mb, device=device)

            stats.update(forward_mem_stats)
            stats.update(backward_mem_stats)

            all_micro_stats.append(stats)

        # ---- 3) Gradient Clipping (after loop) -------------------------
        max_norm = self.cfg.max_grad_norm
        grad_norm: Optional[float] = None

        if max_norm is not None:
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            grad_norm = float(gn)

        # ---- 4) Optimizer Step (one step for the macro-batch) ----------
        self.optimizer.step()

        # ---- 5) Free Gradients (as requested) --------------------------
        # Grads are freed *after* step and *before* weight sync
        self.optimizer.zero_grad(set_to_none=True)

        self._train_step_idx += 1

        # ---- 6) Push policy update into runtime ------------------------
        if self.sync_every_steps and self.publisher is not None and self._train_step_idx % self.sync_every_steps == 0:
            self._push_weights_to_runtime()

        # ---- 7) Optional Checkpoint ------------------------------------
        if self._checkpointer is not None:
            self._checkpointer.maybe_save(
                self.model,
                optimizer=self.optimizer,
                step=self._train_step_idx,
                metadata={"algorithm": self.algo.name},
            )

        # ---- 8) Stats aggregation (skip on non-logging steps) ----------
        log_every = self.cfg.log_every
        if log_every > 1 and self._train_step_idx % log_every != 0:
            # Skip expensive stats sync on non-logging steps
            return {"train/step": float(self._train_step_idx)}

        final_stats = aggregate_stats(
            all_micro_stats,
            all_saw_batches,
            reducers=self.reducers,
            micro_batch_sizes=[len(chunk) for chunk in micro_chunks],
        )
        if grad_norm is not None:
            final_stats["grad_norm"] = float(grad_norm)
        final_stats["train_step"] = float(self._train_step_idx)

        final_stats = self._maybe_reduce_stats_across_ranks(final_stats, device=device)

        # ---- 9) Logging ------------------------------------------------
        grouped = self._group_log_stats(final_stats)
        self._last_train_stats = dict(grouped)
        log_stats = dict(grouped)
        log_stats.update(self._last_eval_stats)
        if self.train_logger is not None:
            try:
                self.train_logger.log(self._train_step_idx, log_stats)
            except Exception:
                logger.exception("Stats logger failed at step %s", self._train_step_idx)

        return grouped

    # ------------------------------------------------------------------
    # Invariants / validation
    # ------------------------------------------------------------------

    def _validate_invariants(self) -> None:
        """
        Fail fast on unsupported configurations.
        """
        if self.sync_every_steps and self.publisher is None:
            raise ValueError(
                "Trainer requires a PolicyPublisher when sync_every_steps > 0. "
                "Set sync_every_steps=0 (or None) to disable runtime syncing."
            )
        if (self.cfg.eval_at_start or (self.cfg.eval_every_n_steps is not None and self.cfg.eval_every_n_steps > 0)) and self.evaluator is None:
            raise ValueError(
                "Trainer evaluation requested (eval_at_start or eval_every_n_steps) but no evaluator was provided."
            )
        if self.cfg.max_seq_len < 1:
            raise ValueError("TrainerConfig.max_seq_len must be >= 1.")
        if self.cfg.micro_token_budget <= 0:
            raise ValueError("TrainerConfig.micro_token_budget must be > 0.")
        if self.cfg.micro_token_budget < self.cfg.max_seq_len:
            raise ValueError("TrainerConfig.micro_token_budget must be >= max_seq_len.")
        if self._resume_from is not None and self._checkpointer is None:
            raise ValueError("resume_from requires a CheckpointManager or checkpoint_config.")

    # ------------------------------------------------------------------
    # Public checkpoint API
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Force an immediate checkpoint save.

        This is the public API for saving checkpoints outside the automatic
        periodic saves. Useful for final checkpoints or manual save points.

        Args:
            metadata: Optional dict of extra metadata to include in the checkpoint.

        Returns:
            Path to the saved checkpoint directory, or None if no checkpointer
            is configured or if not on the primary rank.

        Raises:
            RuntimeError: If no CheckpointManager is configured.
        """
        if self._checkpointer is None:
            raise RuntimeError(
                "Cannot save checkpoint: no CheckpointManager configured. "
                "Pass checkpoint_config to Trainer to enable checkpointing."
            )

        merged_meta = {"algorithm": self.algo.name}
        if metadata:
            merged_meta.update(metadata)

        return self._checkpointer.save(
            self.model,
            optimizer=self.optimizer,
            step=self._train_step_idx,
            metadata=merged_meta,
        )

    @property
    def current_step(self) -> int:
        """Current training step index (read-only)."""
        return self._train_step_idx

    # ------------------------------------------------------------------
    # Sync wrappers for non-async callers
    # ------------------------------------------------------------------

    def train_step_sync(self) -> Dict[str, float]:
        """
        Synchronous wrapper around `train_step()`.

        Use this from plain scripts/CLIs. Do NOT call inside an existing
        running event loop.
        """
        return asyncio.run(self.train_step())

    async def train(
        self,
        num_steps: int,
        *,
        log_every: int = 1,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
        eval_log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        """
        Run `num_steps` training iterations.

        Args:
            num_steps:
                Number of trainer steps (each step pulls a fresh SAWBatch).

            log_every:
                Frequency (in steps) at which to call `log_fn`.

            log_fn:
                Optional callback(stats_dict) for logging / progress reporting.

            eval_log_fn:
                Optional callback for eval metrics.

        """
        if self.cfg.eval_at_start:
            await self.eval(
                log_fn=eval_log_fn,
            )

        for _ in range(num_steps):
            stats = await self.train_step()
            if log_fn is not None and (self._train_step_idx % log_every == 0):
                log_fn(stats)

            if self.cfg.eval_every_n_steps and (self._train_step_idx % self.cfg.eval_every_n_steps == 0):
                await self.eval(
                    log_fn=eval_log_fn,
                )

    def train_sync(
        self,
        num_steps: int,
        *,
        log_every: int = 1,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
        eval_log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        """
        Synchronous wrapper around `train(...)`.
        """
        asyncio.run(
            self.train(
                num_steps,
                log_every=log_every,
                log_fn=log_fn,
                eval_log_fn=eval_log_fn,
            )
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def eval(
        self,
        *,
        reducers: Optional[Mapping[str, Reducer]] = None,
        max_steps: Optional[int] = None,
        timeout_s: Optional[float] = None,
        concurrency: Optional[int] = None,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation using the configured evaluator.
        """
        if self.evaluator is None:
            raise ValueError("Trainer.eval requires evaluator to be set.")

        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0

        if is_distributed:
            # Only rank 0 performs eval; other ranks wait at barriers.
            dist.barrier()
            if rank != 0:
                dist.barrier()
                return {"records": [], "metrics": {}}

        records, metrics = await self.evaluator.eval(
            reducers=reducers,
            max_steps=max_steps or self.cfg.eval_max_steps,
            timeout_s=timeout_s if timeout_s is not None else self.cfg.eval_timeout_s,
            concurrency=concurrency or self.cfg.eval_concurrency,
        )

        if log_fn is not None:
            try:
                log_fn(metrics)
            except Exception:
                logger.exception("Eval log_fn failed")

        raw_eval_stats = {"eval_step": float(self._train_step_idx)}
        raw_eval_stats.update({f"eval_{k}": float(v) for k, v in metrics.items()})
        self._last_eval_stats = self._group_log_stats(raw_eval_stats)
        if self.train_logger is not None and (not is_distributed or rank == 0):
            try:
                merged = dict(self._last_train_stats)
                merged.update(self._last_eval_stats)
                self.train_logger.log(self._train_step_idx, merged)
            except Exception:
                logger.exception("Train logger failed while logging eval metrics")

        if is_distributed:
            dist.barrier()

        return {"records": records, "metrics": metrics}

    def eval_sync(
        self,
        *,
        reducers: Optional[Mapping[str, Reducer]] = None,
        max_steps: Optional[int] = None,
        timeout_s: Optional[float] = None,
        concurrency: Optional[int] = None,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper around eval()."""
        return asyncio.run(
            self.eval(
                reducers=reducers,
                max_steps=max_steps,
                timeout_s=timeout_s,
                concurrency=concurrency,
                log_fn=log_fn,
            )
        )

    # ------------------------------------------------------------------
    # Weight sync into runtime via Agent (FSDP-aware + LoRA-aware)
    # ------------------------------------------------------------------

    def _push_weights_to_runtime(self) -> None:
        """
        Gather weights (handling FSDP if needed) and publish them.

        If the model is a PEFT/LoRA model, we strictly follow the
        Merge -> Publish -> Unmerge pattern so vLLM receives dense weights
        but training continues on adapters.
        """
        if self.publisher is None:
            return
        # Helper to get the underlying model if wrapped in FSDP2
        inner_model = self.model

        # 1. Check if this is a LoRA/PEFT model
        #    (We use getattr so we don't need to import peft here)
        merge_fn = getattr(inner_model, "merge_adapter", None)
        unmerge_fn = getattr(inner_model, "unmerge_adapter", None)
        is_peft = callable(merge_fn) and callable(unmerge_fn)

        # 2. If LoRA, merge weights before gathering
        if is_peft:
            merge_fn()

        try:
            rank = 0
            is_distributed = dist.is_available() and dist.is_initialized()
            if is_distributed:
                rank = dist.get_rank()

            runtime_device = torch.device(
                self.cfg.runtime_device or self.cfg.model_device
            )

            # We use the current training step as the authoritative version for PipelineRL
            current_version = self._train_step_idx

            raw_params: Dict[str, Tensor] = {}

            # --- Gather Raw Params (FSDP2 or Standard) ---
            if isinstance(self.model, fsdp.FSDPModule):
                # DCP full_state_dict gathering is collective; all ranks must participate.
                # Use cpu_offload=True to avoid materializing a full copy on every rank.
                options = StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                )
                full_state = get_model_state_dict(
                    model=self.model,
                    options=options,
                )
                if is_distributed and rank != 0:
                    return
                for k, v in full_state.items():
                    if self.param_filter is not None and not self.param_filter(k, v):
                        continue
                    raw_params[k] = v.detach().to(runtime_device)
            else:
                # Only rank 0 talks to the runtime in distributed mode
                if is_distributed and rank != 0:
                    return

                # Standard model
                for name, p in self.model.named_parameters():
                    # Optimization: In LoRA, only send what we touched (plus what needs fusion)
                    # If not PEFT, only send requires_grad.
                    # If PEFT (merged), we theoretically need to send the whole base layer
                    # because it changed.

                    # If standard training: send only requires_grad=True
                    # If LoRA (merged): send everything (because base weights need updating in vLLM)
                    if not is_peft and not p.requires_grad:
                        continue

                    if self.param_filter is not None:
                        if not self.param_filter(name, p):
                            continue

                    raw_params[name] = p.detach().to(runtime_device)

            if not raw_params:
                return

            # --- Publish ---
            # NOTE: We simply hand off the raw state dict and rely on the Publisher to handle fusion and renaming.
            self.publisher.publish(raw_params, version=current_version)

        finally:
            # 3. CRITICAL: Unmerge adapters immediately after publishing.
            #    If we don't do this, the optimizer state will become invalid
            #    (gradients calculated on merged weights vs adapter weights).
            if is_peft:
                unmerge_fn()
