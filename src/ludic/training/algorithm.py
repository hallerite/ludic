from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, List, Callable, Protocol

from torch import nn, Tensor
import torch

from ludic.training.types import CreditAssigner, SAWBatch
from ludic.training.loss import (
    Loss,
    ReinforceLoss,
    ReinforceBaselineLoss,
    ClippedSurrogateLoss,
    selective_log_softmax,
)
from ludic.training.credit_assignment import MonteCarloReturn, GroupNormalizedReturn, SPOReturn
from ludic.training.value_tracker import ValueTracker


Batch = Mapping[str, Tensor]

class PreprocessFn(Protocol):
    def __call__(
        self,
        saw_batch: SAWBatch,
        *,
        model: Optional[nn.Module] = None,
        pad_token_id: Optional[int] = None,
    ) -> SAWBatch: ...


@dataclass
class RLAlgorithm:
    """
    Full RL algorithm = credit assignment + loss.

    - credit_assigner: maps Rollouts -> per-step scalar credits
                 (e.g. discounted returns / advantages)
    - loss:      consumes a collated batch (built from SAWBatch) and produces
                 a scalar loss and stats.
    - name:      identifier for logging / checkpoints
    """

    name: str
    credit_assigner: CreditAssigner
    loss: Loss
    preprocess: Optional[PreprocessFn] = None

    def compute_loss(
        self,
        model: nn.Module,
        batch: Batch,
    ) -> tuple[Tensor, Dict[str, Any]]:
        """
        Runs the forward pass once and delegates to the Loss object.
        """
        # --- Run the forward pass ---
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits: Tensor = outputs.logits

        # Pass the resulting logits to the loss function
        return self.loss.compute(logits, batch)

    # ------------------------------------------------------------------
    # Optional preprocessing hook
    # ------------------------------------------------------------------
    def preprocess_batch(
        self,
        saw_batch: SAWBatch,
        *,
        model: Optional[nn.Module] = None,
        pad_token_id: Optional[int] = None,
    ) -> SAWBatch:
        """
        Optional algorithm-specific preprocessing on the SAWBatch before collation.

        This is mainly used to materialize additional metadata needed by the loss,
        e.g. behavior-policy logprobs for ratio-based objectives.
        """
        if self.preprocess is None:
            return saw_batch
        return self.preprocess(saw_batch, model=model, pad_token_id=pad_token_id)


# ---------------------------------------------------------------------------
# Behavior logprob plumbing (for ratio-based objectives)
# ---------------------------------------------------------------------------


def _ensure_old_token_logprobs(
    saw_batch: SAWBatch,
    *,
    model: Optional[nn.Module] = None,
    pad_token_id: Optional[int] = None,
    backfill_chunk_size: Optional[int] = None,
) -> SAWBatch:
    """
    Ensure SAWItems carry per-token behavior logprobs for the action region.

    Source priority:
      1) If an item has `meta["completion_logprobs"]` (rollout-time chosen-token
         logprobs), copy them to `meta["old_token_logprobs"]`.
      2) If still missing, optionally backfill via teacher forcing under `model`
         (no grad), but only for synchronous (on-policy) batches.

    - If all items already have old_token_logprobs, this is a no-op.
    - If some items are missing and a model is provided, it computes per-token
      logprobs of the sampled sequence under the current model (no grad) and
      stores them on the items.
    - Recompute is only allowed for synchronous (on-policy) batches; async /
      pipeline batches must ship behavior logprobs with the data.
    """
    items = saw_batch.items

    # 1) Copy rollout-time chosen-token logprobs if present.
    for it in items:
        if "completion_logprobs" in it.meta and isinstance(it.meta["completion_logprobs"], list):
            it.meta["old_token_logprobs"] = list(it.meta["completion_logprobs"])

    missing_indices: List[int] = [
        i for i, it in enumerate(items) if not isinstance(it.meta.get("old_token_logprobs"), list)
    ]
    if not missing_indices:
        return saw_batch

    # 2) Teacher-forced backfill is only allowed for synchronous (on-policy) batches.
    is_async = any("policy_version" in it.meta for it in items)
    if is_async:
        raise ValueError(
            "Missing old_token_logprobs on a batch tagged with policy_version; "
            "recomputation is only supported for synchronous on-policy batches."
        )

    if model is None or pad_token_id is None:
        raise ValueError("model and pad_token_id are required to backfill old_token_logprobs.")

    device = next(model.parameters()).device
    subset = [items[i] for i in missing_indices]
    chunk_size = backfill_chunk_size or len(subset)

    restore_train_state = model.training
    if restore_train_state:
        model.eval()

    try:
        for start in range(0, len(subset), chunk_size):
            chunk_items = subset[start : start + chunk_size]
            lengths = [len(it.input_ids) for it in chunk_items]
            max_len = max(lengths)

            input_ids_list = []
            attn_mask_list = []
            action_mask_list = []
            for it in chunk_items:
                L = len(it.input_ids)
                ids = torch.full((max_len,), pad_token_id, dtype=torch.long)
                am = torch.zeros((max_len,), dtype=torch.long)
                actm = torch.zeros((max_len,), dtype=torch.float32)
                ids[:L] = torch.tensor(it.input_ids, dtype=torch.long)
                am[:L] = torch.tensor(it.attention_mask, dtype=torch.long)
                actm[:L] = torch.tensor(it.action_mask, dtype=torch.float32)
                input_ids_list.append(ids)
                attn_mask_list.append(am)
                action_mask_list.append(actm)

            batch_input_ids = torch.stack(input_ids_list, dim=0).to(device)
            batch_attn = torch.stack(attn_mask_list, dim=0).to(device)
            batch_action_mask = torch.stack(action_mask_list, dim=0).to(device)

            with torch.inference_mode():
                logits = model(input_ids=batch_input_ids, attention_mask=batch_attn).logits
                logits_shifted = logits[:, :-1, :]
                target_ids = batch_input_ids[:, 1:]
                token_logp = selective_log_softmax(logits_shifted, target_ids)  # [B, T-1]
                action_mask_shifted = batch_action_mask[:, 1:]  # align with targets

            for idx, it in enumerate(chunk_items):
                mask = action_mask_shifted[idx].bool()
                per_token = token_logp[idx][mask].detach().cpu().tolist()
                it.meta["old_token_logprobs"] = [float(v) for v in per_token]
    finally:
        if restore_train_state:
            model.train()

    return saw_batch


def make_old_logprob_preprocessor(*, backfill_chunk_size: Optional[int] = None) -> PreprocessFn:
    """
    Return a PreprocessFn that ensures per-token behavior logprobs exist on items.

    This is a small functional wrapper so callers can configure backfill behavior
    without needing an algorithm subclass.
    """

    def _pre(
        saw_batch: SAWBatch,
        *,
        model: Optional[nn.Module] = None,
        pad_token_id: Optional[int] = None,
    ) -> SAWBatch:
        return _ensure_old_token_logprobs(
            saw_batch,
            model=model,
            pad_token_id=pad_token_id,
            backfill_chunk_size=backfill_chunk_size,
        )

    return _pre


# ---------------------------------------------------------------------------
# Presets: REINFORCE and REINFORCE+baseline
# ---------------------------------------------------------------------------


def make_reinforce(
    *,
    gamma: float = 1.0,
    name: str = "reinforce",
) -> RLAlgorithm:
    """
    REINFORCE without baseline.

    - Credit assignment: Monte Carlo discounted return-to-go with discount `gamma`
          G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    - Loss:              ReinforceLoss using `batch["weight"]` as the return

    The orchestrator will use this algorithm's `credit_assigner` (MonteCarloReturn)
    to compute G_t per step, store it in SAWItem.weight, and collate that
    into `batch["weight"]` for the loss.
    """
    credit_assigner: CreditAssigner = MonteCarloReturn(gamma=gamma)
    loss: Loss = ReinforceLoss()

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
    )


def make_reinforce_baseline(
    *,
    gamma: float = 1.0,
    name: str = "reinforce_baseline",
    normalize_adv: bool = False,
) -> RLAlgorithm:
    """
    REINFORCE with batch-mean baseline:

        G_t = discounted return-to-go from step t
        b   = mean(G_t) over the batch
        A_t = G_t - b
        loss = - E[ A_t * log π(a_t|s_t) ]

    Here:
      - MonteCarloReturn(gamma) computes G_t and feeds it into SAWItem.weight
      - the collated batch exposes this as `batch["weight"]`

    If `normalize_adv=True`, A_t is additionally normalized to zero mean /
    unit variance within the batch before being used in the loss.
    """
    credit_assigner: CreditAssigner = MonteCarloReturn(gamma=gamma)
    loss: Loss = ReinforceBaselineLoss(
        normalize=normalize_adv,
    )

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
    )


def make_grpo(
    *,
    group_size: int,
    group_normalize_adv: bool = True,
    clip_eps: float = 0.2,
    length_normalize: bool = False,
    name: str = "grpo",
    backfill_chunk_size: Optional[int] = None,
) -> RLAlgorithm:
    """
    GRPO-style preset (clipped surrogate):

      - Credit assignment: group-normalized returns (per-group baseline)
      - Loss: PPO-style clipped surrogate (policy term only)

    Rollouts must carry `group_id` in their metadata and each group must
    have exactly `group_size` members. Raises ValueError otherwise.

    Args:
        group_size: Number of rollouts per group.
        group_normalize_adv: Whether to normalize advantages within each group.
        clip_eps: PPO clipping epsilon for the surrogate objective.
        length_normalize: Whether to normalize log-probs by action length.
        name: Algorithm name for logging/metrics.
        backfill_chunk_size: Chunk size for teacher-forced logprob backfill.

    Note: For the clipped ratio objective, we need behavior-policy logprobs.
    This preset installs a preprocessor that:
      - copies rollout-time `completion_logprobs` into `old_token_logprobs`, or
      - backfills via teacher forcing for synchronous on-policy batches.
    """
    credit_assigner: CreditAssigner = GroupNormalizedReturn(
        group_size=group_size,
        normalize_adv=group_normalize_adv,
    )
    loss: Loss = ClippedSurrogateLoss(clip_eps=clip_eps, length_normalize=length_normalize)
    preprocess = make_old_logprob_preprocessor(backfill_chunk_size=backfill_chunk_size)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=preprocess,
    )


def make_spo(
    *,
    tracker: Optional[ValueTracker] = None,
    normalize_adv: bool = True,
    clip_eps: float = 0.2,
    length_normalize: bool = False,
    name: str = "spo",
    backfill_chunk_size: Optional[int] = None,
) -> tuple[RLAlgorithm, ValueTracker]:
    """
    Single-stream Policy Optimization (SPO) preset.

    SPO replaces GRPO's group-based baselines with a persistent per-prompt
    value tracker, eliminating:
    - Degenerate groups (all-correct or all-incorrect → zero gradient)
    - Synchronization barriers (no waiting for slowest group member)

    Reference: "Single-stream Policy Optimization" (Xu & Ding, 2025)

    Args:
        tracker: Optional ValueTracker instance. If None, creates a new one.
            Pass an existing tracker to maintain state across training runs.
        normalize_adv: Whether to normalize advantages globally across batch.
        clip_eps: PPO clipping epsilon for the surrogate objective.
        length_normalize: Whether to normalize log-probs by action length.
        name: Algorithm name for logging/metrics.
        backfill_chunk_size: Chunk size for teacher-forced logprob backfill.

    Returns:
        Tuple of (RLAlgorithm, ValueTracker). The tracker is returned so you
        can persist it across training sessions or inspect its state.

    Example:
        algo, tracker = make_spo()
        # tracker maintains per-prompt value estimates across batches
        # no grouping needed - each rollout is independent
    """
    if tracker is None:
        tracker = ValueTracker()

    credit_assigner = SPOReturn(
        tracker=tracker,
        normalize_adv=normalize_adv,
    )
    loss: Loss = ClippedSurrogateLoss(clip_eps=clip_eps, length_normalize=length_normalize)
    preprocess = make_old_logprob_preprocessor(backfill_chunk_size=backfill_chunk_size)

    algo = RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=preprocess,
    )

    return algo, tracker
