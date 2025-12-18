from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol, Tuple, List

import torch
from torch import Tensor
import torch.nn.functional as F
import math


Batch = Mapping[str, Tensor]


class Loss(Protocol):
    """
    Generic loss: given model outputs (logits) and a collated batch, return
    (scalar_loss, stats).
    """

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        ...

# We define this as a standalone helper so torch.compile can cache it cleanly.
# dynamic=True is critical for varying sequence lengths (preventing recompilation).
@torch.compile(dynamic=True)
def selective_log_softmax(logits: Tensor, index: Tensor) -> Tensor:
    """
    Fused kernel for log_softmax + gather.
    
    Inductor (torch.compile) generates a kernel that computes the log_softmax
    normalization term and selects the target token in a single pass.
    This avoids materializing the massive [B, T, V] probability tensor in VRAM.
    """
    # This looks naive, but the compiler fuses it into a single read/write op.
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

def compute_logp_action(
    logits: Tensor, 
    input_ids: Tensor, 
    action_mask: Tensor,
    *,
    length_normalize: bool = False,
) -> Tensor:
    """
    Compute log π(a|s) given token-level logits and an action mask.

    Args:
        logits: [B, T, V] float tensor of unnormalized logits.
        input_ids: [B, T] long tensor of token ids actually sampled.
        action_mask: [B, T] {0,1} mask; 1 where tokens belong to the "action".

    Returns:
        logp_action: [B] log-prob of the entire action sequence per sample.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B, T, V], got {tuple(logits.shape)}")
    
    if input_ids.shape != logits.shape[:2]:
        raise ValueError(f"Shape mismatch: input_ids {input_ids.shape} vs logits {logits.shape}")

    # Shift for causal LM: logits[t] predicts input_ids[t+1]
    if logits.size(1) < 2:
        raise ValueError("Sequence too short to compute next-token logprobs.")
    logits_shifted = logits[:, :-1, :]          # [B, T-1, V]
    target_ids = input_ids[:, 1:]               # [B, T-1]
    action_mask_shifted = action_mask[:, 1:]    # [B, T-1]

    # Use the compiled fused kernel on aligned targets
    token_logp = selective_log_softmax(logits_shifted, target_ids)

    # Sum log-probs over the action region only: [B]
    amask = action_mask_shifted.to(token_logp.dtype)
    logp_action = (token_logp * amask).sum(dim=-1)

    if length_normalize:
        lengths = amask.sum(dim=-1).clamp(min=1.0)
        logp_action = logp_action / lengths

    return logp_action


# ---------------------------------------------------------------------------
# REINFORCE family
# ---------------------------------------------------------------------------


@dataclass
class ReinforceLoss:
    """
    Vanilla REINFORCE:

        loss = - E[ A * log π(a|s) ]

    where A is taken from `batch["weight"]`.
    """
    length_normalize: bool = False

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]            # [B, T]
        action_mask = batch["action_mask"]        # [B, T]
        advantages = batch["weight"]              # [B]

        logp_action = compute_logp_action(
            logits, input_ids, action_mask, length_normalize=self.length_normalize
        )  # [B]

        loss = - (advantages * logp_action).mean()

        stats: Dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "adv_mean": float(advantages.mean().detach().cpu()),
            "adv_std": float(advantages.std(unbiased=False).detach().cpu()),
            "logp_mean": float(logp_action.mean().detach().cpu()),
        }
        return loss, stats


# ---------------------------------------------------------------------------
# Masked token-level CE (SFT-friendly)
# ---------------------------------------------------------------------------


@dataclass
class MaskedCausalLMCrossEntropyLoss:
    """
    Token-level masked cross entropy over the "action" region.

    This is the standard SFT objective when you have (prompt, completion)
    and want to train only on the completion tokens.

    Expects:
      - batch["input_ids"]:   [B, T]
      - batch["action_mask"]: [B, T] 0/1 mask where 1 marks completion tokens
      - batch["weight"]:      [B] optional per-sample weights (defaults to 1.0)
    """

    length_normalize: bool = True

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]  # [B, T]
        action_mask = batch["action_mask"]  # [B, T]
        weights = batch.get("weight")

        if logits.ndim != 3:
            raise ValueError(f"Expected logits [B, T, V], got {tuple(logits.shape)}")
        if input_ids.shape != logits.shape[:2]:
            raise ValueError(f"Shape mismatch: input_ids {input_ids.shape} vs logits {logits.shape}")

        if logits.size(1) < 2:
            raise ValueError("Sequence too short to compute next-token loss.")

        # Shift for causal LM: logits[t] predicts input_ids[t+1]
        logits_shifted = logits[:, :-1, :].float()  # [B, T-1, V]
        targets = input_ids[:, 1:]  # [B, T-1]
        mask = action_mask[:, 1:].to(dtype=torch.float32)  # [B, T-1]

        B, Tm1, V = logits_shifted.shape
        per_token_nll = F.cross_entropy(
            logits_shifted.reshape(B * Tm1, V),
            targets.reshape(B * Tm1),
            reduction="none",
        ).reshape(B, Tm1)

        token_counts = mask.sum(dim=-1).clamp(min=1.0)  # [B]
        per_sample_nll = (per_token_nll * mask).sum(dim=-1)  # [B]
        if self.length_normalize:
            per_sample_nll = per_sample_nll / token_counts

        if weights is not None:
            loss = (per_sample_nll * weights.to(per_sample_nll.dtype)).mean()
        else:
            loss = per_sample_nll.mean()

        # Stats for parity with ReinforceLoss
        per_sample_logp = -per_sample_nll
        stats: Dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "logp_mean": float(per_sample_logp.mean().detach().cpu()),
            "nll_mean": float(per_sample_nll.mean().detach().cpu()),
            "avg_action_tokens": float(token_counts.mean().detach().cpu()),
        }
        return loss, stats


@dataclass
class OnPolicyDistillationLoss:
    """
    On-Policy Distillation (OPD) using per-token reverse-KL advantages.

    Expects:
      - batch["input_ids"]: [B, T]
      - batch["action_mask"]: [B, T] (1 on action/completion tokens)
      - batch["teacher_token_logprobs"]: [B, T-1] teacher-forced chosen-token logprobs,
            aligned to targets input_ids[:, 1:] and zero elsewhere.
      - batch["old_token_logprobs"]: [B, T-1] behavior-policy chosen-token logprobs,
            aligned to targets input_ids[:, 1:] and zero elsewhere.

    Optional:
      - batch["weight"]: [B] scalar env-derived weight; used only when env_weight_coef != 0.
    """

    opd_coef: float = 1.0
    env_weight_coef: float = 0.0
    length_normalize: bool = False

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        if not math.isfinite(self.opd_coef):
            raise ValueError("opd_coef must be finite.")
        if not math.isfinite(self.env_weight_coef):
            raise ValueError("env_weight_coef must be finite.")

        input_ids = batch["input_ids"]  # [B, T]
        action_mask = batch["action_mask"].to(dtype=torch.float32)  # [B, T]

        if "teacher_token_logprobs" not in batch:
            raise KeyError("OnPolicyDistillationLoss requires 'teacher_token_logprobs' in batch.")
        if "old_token_logprobs" not in batch:
            raise KeyError("OnPolicyDistillationLoss requires 'old_token_logprobs' in batch.")

        teacher_lp = batch["teacher_token_logprobs"].to(dtype=torch.float32)  # [B, T-1]
        old_lp = batch["old_token_logprobs"].to(dtype=torch.float32)  # [B, T-1]

        if logits.ndim != 3:
            raise ValueError(f"Expected logits [B, T, V], got {tuple(logits.shape)}")
        if input_ids.shape != logits.shape[:2]:
            raise ValueError(f"Shape mismatch: input_ids {input_ids.shape} vs logits {logits.shape}")
        if logits.size(1) < 2:
            raise ValueError("Sequence too short to compute next-token OPD loss.")

        # Align with next-token prediction.
        logits_shifted = logits[:, :-1, :]  # [B, T-1, V]
        targets = input_ids[:, 1:]  # [B, T-1]
        token_logp = selective_log_softmax(logits_shifted, targets)  # [B, T-1]

        # Mask that selects action tokens in the target positions.
        action_mask_shifted = action_mask[:, 1:]  # [B, T-1]

        if teacher_lp.shape != token_logp.shape:
            raise ValueError(
                f"teacher_token_logprobs shape {tuple(teacher_lp.shape)} must match token_logp {tuple(token_logp.shape)}"
            )
        if old_lp.shape != token_logp.shape:
            raise ValueError(
                f"old_token_logprobs shape {tuple(old_lp.shape)} must match token_logp {tuple(token_logp.shape)}"
            )

        # Per-token reverse KL advantage (discount=0): A_t = logp_teacher - logp_old_student
        advantages = (teacher_lp - old_lp).detach() * self.opd_coef  # [B, T-1]

        # Optional hybrid term: broadcast env weight across action tokens.
        if self.env_weight_coef != 0.0:
            if "weight" not in batch:
                raise KeyError("env_weight_coef != 0 requires 'weight' in batch.")
            env_w = batch["weight"].to(dtype=torch.float32).unsqueeze(-1)  # [B, 1]
            advantages = advantages + self.env_weight_coef * env_w

        per_token_loss = -(advantages * token_logp)  # [B, T-1]
        masked = per_token_loss * action_mask_shifted

        if self.length_normalize:
            denom = action_mask_shifted.sum(dim=-1).clamp(min=1.0)  # [B]
            loss = (masked.sum(dim=-1) / denom).mean()
        else:
            denom = action_mask_shifted.sum().clamp(min=1.0)
            loss = masked.sum() / denom

        # Useful diagnostics.
        with torch.no_grad():
            denom_tokens = action_mask_shifted.sum().clamp(min=1.0)
            mean_reverse_kl = ((token_logp - teacher_lp) * action_mask_shifted).sum() / denom_tokens
            mean_adv = ((teacher_lp - old_lp) * action_mask_shifted).sum() / denom_tokens

        stats: Dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "reverse_kl_mean": float(mean_reverse_kl.detach().cpu()),
            "adv_mean": float(mean_adv.detach().cpu()),
            "opd_coef": float(self.opd_coef),
            "env_weight_coef": float(self.env_weight_coef),
        }
        return loss, stats


@dataclass
class ReinforceBaselineLoss:
    """
    REINFORCE with batch-mean baseline:

        A_i = adv_i - mean(adv)
        loss = - E[ A_i * log π(a_i|s_i) ]

    where adv_i is batch["weight"].
    """

    normalize: bool = False
    length_normalize: bool = False

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]
        action_mask = batch["action_mask"]
        adv_raw = batch["weight"]                # [B]

        logp_action = compute_logp_action(
            logits, input_ids, action_mask, length_normalize=self.length_normalize
        )  # [B]

        baseline = adv_raw.mean()
        advantages = adv_raw - baseline

        if self.normalize:
            std = advantages.std(unbiased=False)
            advantages = advantages / (std + 1e-8)

        loss = - (advantages * logp_action).mean()

        stats: Dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "baseline": float(baseline.detach().cpu()),
            "adv_mean": float(advantages.mean().detach().cpu()),
            "adv_std": float(advantages.std(unbiased=False).detach().cpu()),
            "logp_mean": float(logp_action.mean().detach().cpu()),
        }
        return loss, stats


# ---------------------------------------------------------------------------
# PPO clipped policy loss (no value term here)
# ---------------------------------------------------------------------------


@dataclass
class ClippedSurrogateLoss:
    """
    PPO-style clipped surrogate policy loss (actor part only):

        r = π_new(a|s) / π_old(a|s)
        L_clip = - E[ min(r * A, clip(r, 1 - eps, 1 + eps) * A) ]

    Expects:
        - batch["weight"]:       A  (advantages)      [B]
        - batch[old_logp_key]:   log π_old(a|s)      [B]
        - input_ids / attention_mask / action_mask for π_new.
    """

    clip_eps: float = 0.2
    old_logp_key: str = "old_logp_action"
    length_normalize: bool = False

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]
        action_mask = batch["action_mask"]
        advantages = batch["weight"]              # [B]
        if self.old_logp_key not in batch:
            raise KeyError(f"ClippedSurrogateLoss requires '{self.old_logp_key}' in batch.")

        logp_action = compute_logp_action(
            logits,
            input_ids,
            action_mask,
            length_normalize=self.length_normalize,
        )  # [B]
        old_logp = batch[self.old_logp_key]  # [B]
        if self.length_normalize:
            lengths = action_mask[:, 1:].to(old_logp.dtype).sum(dim=-1).clamp(min=1.0)
            old_logp = old_logp / lengths

        # ratio = π_new / π_old
        ratio = torch.exp(logp_action - old_logp)                          # [B]

        # unclipped and clipped objectives
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

        obj = torch.min(unclipped, clipped)
        loss = -obj.mean()

        clip_frac = ((ratio > 1.0 + self.clip_eps) | (ratio < 1.0 - self.clip_eps)).float().mean()

        stats: Dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "ratio_mean": float(ratio.mean().detach().cpu()),
            "ratio_std": float(ratio.std(unbiased=False).detach().cpu()),
            "clip_frac": float(clip_frac.detach().cpu()),
            "adv_mean": float(advantages.mean().detach().cpu()),
            "adv_std": float(advantages.std(unbiased=False).detach().cpu()),
            "logp_mean": float(logp_action.mean().detach().cpu()),
        }
        return loss, stats


# ---------------------------------------------------------------------------
# KL penalty and entropy bonus
# ---------------------------------------------------------------------------


@dataclass
class KLLoss:
    """
    KL penalty between π_new and a reference policy whose log-prob is stored as
    batch[old_logp_key].

    We use the standard policy-gradient surrogate estimate:

        KL(π_new || π_old) ≈ E_{a ~ π_new} [ log π_new(a|s) - log π_old(a|s) ]

    Loss is:

        loss = coeff * mean(kl)

    (You usually *add* this to the overall loss; coeff > 0 makes it a penalty.)
    """

    coeff: float = 1.0
    old_logp_key: str = "old_logp_action"
    length_normalize: bool = False

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]
        action_mask = batch["action_mask"]
        old_logp = batch[self.old_logp_key]       # [B]

        logp_new = compute_logp_action(
            logits,
            input_ids,
            action_mask,
            length_normalize=self.length_normalize,
        )  # [B]
        if self.length_normalize:
            lengths = action_mask[:, 1:].to(old_logp.dtype).sum(dim=-1).clamp(min=1.0)
            old_logp = old_logp / lengths

        kl = logp_new - old_logp                                           # [B]
        loss = self.coeff * kl.mean()

        stats: Dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "kl_mean": float(kl.mean().detach().cpu()),
            "kl_std": float(kl.std(unbiased=False).detach().cpu()),
        }
        return loss, stats


@dataclass
class EntropyBonus:
    """
    Entropy bonus over the action region.

    Computes token-level entropy H(π(·|token)) and averages over tokens where
    action_mask == 1. Loss is:

        loss = - coeff * mean_entropy

    So with coeff > 0, this *reduces* the total loss (encourages exploration).
    """

    coeff: float = 0.01

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        action_mask = batch["action_mask"]

        logprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)

        # token entropy: [B, T]
        token_entropy = -(probs * logprobs).sum(dim=-1)

        mask = action_mask.to(token_entropy.dtype)

        masked_entropy = token_entropy * mask   # [B, T]
        # avoid divide-by-zero if mask is all zeros
        denom = mask.sum()
        if denom.item() == 0:
            mean_entropy = torch.zeros((), device=logits.device, dtype=logits.dtype)
        else:
            mean_entropy = masked_entropy.sum() / denom

        loss = -self.coeff * mean_entropy

        stats: Dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "entropy_mean": float(mean_entropy.detach().cpu()),
        }
        return loss, stats


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------


@dataclass
class LossTerm:
    """
    Single term inside a CompositeLoss.

    - name:   short identifier for logging
    - loss:   loss object implementing Loss protocol
    - weight: scalar multiplier applied to that loss
    """
    name: str
    loss: Loss
    weight: float = 1.0


@dataclass
class CompositeLoss:
    """
    Combine multiple Loss terms into a single scalar loss:

        total_loss = sum_i weight_i * loss_i

    Stats are merged with hierarchical keys:

        "{name}/loss", "{name}/<stat_key>", ...

    and a top-level "loss" key for the final combined loss.
    
    This class expects logits to be passed in, and it passes them
    down to all child terms.
    """

    terms: List[LossTerm]

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        if not self.terms:
            raise ValueError("CompositeLoss.terms must be non-empty")

        total_loss: Tensor | None = None
        stats: Dict[str, Any] = {}

        for term in self.terms:
            # Pass the pre-computed logits down to the child term
            raw_loss, term_stats = term.loss.compute(logits, batch)
            scaled_loss = term.weight * raw_loss

            if total_loss is None:
                total_loss = scaled_loss
            else:
                total_loss = total_loss + scaled_loss

            # per-term stats
            stats[f"{term.name}/loss"] = float(raw_loss.detach().cpu())
            stats[f"{term.name}/weight"] = term.weight
            for k, v in term_stats.items():
                stats[f"{term.name}/{k}"] = v

        assert total_loss is not None
        stats["loss"] = float(total_loss.detach().cpu())

        return total_loss, stats
