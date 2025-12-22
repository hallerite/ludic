from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainerConfig:
    """
    High-level trainer configuration.

    This is *purely* about optimization / model device / collation.
    Rollout and batch-generation config live in BatchSource / Orchestrator.

    ==========================
    Model / Optimization
    ==========================

    - model_device:
          Device where the trainable model lives ("cuda:0", "cpu", ...).
          Trainer will move *batches* to this device, but assumes the model
          is already on the correct device and/or FSDP-wrapped.

    - runtime_device:
          Device to move tensors to before pushing them into the serving
          runtime. If None, defaults to `model_device`.

    - lr, weight_decay, betas, eps:
          AdamW hyperparameters.

    - max_grad_norm:
          Gradient clipping threshold; None disables clipping.

    - max_seq_len:
          Max token length for any single sample. Trainer raises if exceeded.
          
    - micro_token_budget:
          Max padded tokens per micro-batch (roughly batch_size * max_seq_len).
          Trainer splits macro-batches into micro-batches that fit this budget.
          Must be >= max_seq_len.
          
    - sync_every_steps:
          Frequency (in macro-steps) at which to push updated policy 
          weights to the Agent's runtime (e.g., vLLM). Set to 0 to disable
          syncing (e.g., pure offline/local training).

    - mixed_precision_dtype:
          Optional string to configure FSDP's mixed precision policy. 
          Use "bf16" or "fp16". If None, defaults to full precision (fp32).

    ==========================
    Collation
    ==========================

    - pad_token_id:
          Used when padding sequences during SAW collation.

    ==========================
    Distributed
    ==========================

    - reduce_stats_across_ranks:
          If True (and torch.distributed is initialized), Trainer will all-reduce
          the per-rank stats dict before logging/returning it.

    ==========================
    Evaluation
    ==========================

    - eval_at_start:
          If True, run eval before the first training step when using
          Trainer.train(). Requires Trainer(evaluator=...).

    - eval_every_n_steps:
          Run eval every N training steps. None disables periodic eval.

    - eval_concurrency:
          Number of concurrent episodes during eval rollout generation.

    - eval_max_steps:
          Maximum steps per eval episode.

    - eval_timeout_s:
          Optional per-call timeout for eval rollouts.
    """

    # ----- model / optimization -------------------
    model_device: str = "cuda"
    runtime_device: Optional[str] = None

    lr: float = 1e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    max_grad_norm: Optional[float] = 1.0

    # FSDP/RLHF specific settings
    max_seq_len: int = 1024
    micro_token_budget: int = 8192
    sync_every_steps: int = 1
    mixed_precision_dtype: Optional[str] = "bf16"

    # PipelineRL specific settings
    max_lag: Optional[int] = None  # Drop batches older than N steps
    reduce_stats_across_ranks: bool = False

    # ----- collation ------------------------------
    pad_token_id: int = 0

    # ----- evaluation -----------------------------
    eval_at_start: bool = False
    eval_every_n_steps: Optional[int] = None
    eval_concurrency: int = 32
    eval_max_steps: int = 1
    eval_timeout_s: Optional[float] = None
