from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from beartype.typing import Any, Dict, Mapping, Protocol, Tuple, List, Optional

from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
import torch
from torch import Tensor
import torch.nn.functional as F


Batch = Mapping[str, Tensor]
Logits = Float[Tensor, "B T V"]
TokenIds = Int[Tensor, "B T"]
Mask = Float[Tensor, "B T"]
Weights = Float[Tensor, "B"]

logger = logging.getLogger(__name__)

def _no_op(fn):
    return fn

_TYPECHECK_ENABLED = os.getenv("LUDIC_TYPECHECK", "0") == "1"
typechecker = beartype if _TYPECHECK_ENABLED else _no_op
logger.info(
    "Jaxtyping runtime checks: %s",
    "enabled (beartype)" if _TYPECHECK_ENABLED else "disabled",
)


class Loss(Protocol):
    """
    Generic loss: given model outputs (logits) and a collated batch, return
    (scalar_loss, stats).

    Stats values must be scalar tensors (on the same device as logits) to enable
    batched D2H transfer during aggregation. Use `tensor.detach()` not `float(...)`.
    """

    def compute(self, logits: Logits, batch: Batch) -> Tuple[Tensor, Dict[str, Tensor]]:
        ...

# We define this as a standalone helper so torch.compile can cache it cleanly.
# dynamic=True is critical for varying sequence lengths (preventing recompilation).
@jaxtyped(typechecker=typechecker)
@torch.compile(dynamic=True)
def selective_log_softmax(logits: Logits, index: TokenIds) -> Float[Tensor, "B T"]:
    """
    Fused kernel for log_softmax + gather.
    
    Inductor (torch.compile) generates a kernel that computes the log_softmax
    normalization term and selects the target token in a single pass.
    This avoids materializing the massive [B, T, V] probability tensor in VRAM.
    """
    # This looks naive, but the compiler fuses it into a single read/write op.
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

@jaxtyped(typechecker=typechecker)
def compute_logp_action(
    logits: Logits,
    input_ids: TokenIds,
    action_mask: Mask,
    *,
    length_normalize: bool = False,
) -> Weights:
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


@jaxtyped(typechecker=typechecker)
def compute_token_logp(
    logits: Logits,
    input_ids: TokenIds,
) -> Float[Tensor, "B T-1"]:
    """
    Compute per-token log π(a_t|s_t) for each next-token prediction.

    Returns:
        token_logp: [B, T-1] log-prob of each next token.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B, T, V], got {tuple(logits.shape)}")
    if input_ids.shape != logits.shape[:2]:
        raise ValueError(f"Shape mismatch: input_ids {input_ids.shape} vs logits {logits.shape}")
    if logits.size(1) < 2:
        raise ValueError("Sequence too short to compute next-token logprobs.")

    logits_shifted = logits[:, :-1, :]          # [B, T-1, V]
    target_ids = input_ids[:, 1:]               # [B, T-1]
    return selective_log_softmax(logits_shifted, target_ids)


# ---------------------------------------------------------------------------
# REINFORCE family
# ---------------------------------------------------------------------------


@dataclass
class ReinforceLoss:
    """
    REINFORCE with importance sampling correction:

        loss = - E[ sg(r) * A * log π(a|s) ]

    where:
        - A is taken from `batch["weight"]`
        - r = π_new(a|s) / π_old(a|s) is the IS ratio
        - sg(·) is stop-gradient

    The IS correction accounts for the fact that in LLM-RL we are always
    slightly off-policy due to different inference vs training kernels.

    Expects batch[old_logp_key] containing log π_old(a|s).
    """

    length_normalize: bool = False
    old_logp_key: str = "old_logp_action"

    @jaxtyped(typechecker=typechecker)
    def compute(self, logits: Logits, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]            # [B, T]
        action_mask = batch["action_mask"]        # [B, T]
        advantages = batch["weight"]              # [B]

        if self.old_logp_key not in batch:
            raise KeyError(f"ReinforceLoss requires '{self.old_logp_key}' in batch.")

        logp_action = compute_logp_action(
            logits, input_ids, action_mask, length_normalize=self.length_normalize
        )  # [B]

        old_logp = batch[self.old_logp_key]  # [B]
        if self.length_normalize:
            lengths = action_mask[:, 1:].to(old_logp.dtype).sum(dim=-1).clamp(min=1.0)
            old_logp = old_logp / lengths

        log_ratio = logp_action - old_logp
        ratio = torch.exp(log_ratio)
        # KL approximation for stats: r - log(r) - 1
        mismatch_kl = ratio - log_ratio - 1.0

        # Stop-gradient on IS weight: gradient flows only through log π
        loss = - (ratio.detach() * advantages * logp_action).mean()

        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "adv_mean": advantages.mean().detach(),
            "adv_std": advantages.std(unbiased=False).detach(),
            "logp_mean": logp_action.mean().detach(),
            "ratio_mean": ratio.mean().detach(),
            "ratio_std": ratio.std(unbiased=False).detach(),
            "kl_actor_policy": mismatch_kl.mean().detach(),
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

    @jaxtyped(typechecker=typechecker)
    def compute(self, logits: Logits, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
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
            "loss": loss.detach(),
            "logp_mean": per_sample_logp.mean().detach(),
            "nll_mean": per_sample_nll.mean().detach(),
            "avg_action_tokens": token_counts.mean().detach(),
        }
        return loss, stats


@dataclass
class ReinforceBaselineLoss:
    """
    REINFORCE with batch-mean baseline and importance sampling correction:

        A_i = adv_i - mean(adv)
        loss = - E[ sg(r) * A_i * log π(a_i|s_i) ]

    where:
        - adv_i is batch["weight"]
        - r = π_new(a|s) / π_old(a|s) is the IS ratio
        - sg(·) is stop-gradient

    Expects batch[old_logp_key] containing log π_old(a|s).
    """

    normalize: bool = False
    length_normalize: bool = False
    old_logp_key: str = "old_logp_action"

    @jaxtyped(typechecker=typechecker)
    def compute(self, logits: Logits, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]
        action_mask = batch["action_mask"]
        adv_raw = batch["weight"]                # [B]

        if self.old_logp_key not in batch:
            raise KeyError(f"ReinforceBaselineLoss requires '{self.old_logp_key}' in batch.")

        logp_action = compute_logp_action(
            logits, input_ids, action_mask, length_normalize=self.length_normalize
        )  # [B]

        old_logp = batch[self.old_logp_key]  # [B]
        if self.length_normalize:
            lengths = action_mask[:, 1:].to(old_logp.dtype).sum(dim=-1).clamp(min=1.0)
            old_logp = old_logp / lengths

        log_ratio = logp_action - old_logp
        ratio = torch.exp(log_ratio)
        mismatch_kl = ratio - log_ratio - 1.0

        baseline = adv_raw.mean()
        advantages = adv_raw - baseline

        if self.normalize:
            std = advantages.std(unbiased=False)
            advantages = advantages / (std + 1e-8)

        # Stop-gradient on IS weight
        loss = - (ratio.detach() * advantages * logp_action).mean()

        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "baseline": baseline.detach(),
            "adv_mean": advantages.mean().detach(),
            "adv_std": advantages.std(unbiased=False).detach(),
            "logp_mean": logp_action.mean().detach(),
            "ratio_mean": ratio.mean().detach(),
            "ratio_std": ratio.std(unbiased=False).detach(),
            "kl_actor_policy": mismatch_kl.mean().detach(),
        }
        return loss, stats


# ---------------------------------------------------------------------------
# PPO clipped policy loss (no value term here)
# ---------------------------------------------------------------------------


@dataclass
class ClippedSurrogateLoss:
    """
    PPO-style clipped surrogate policy loss (actor part only).

        r = π_new(a|s) / π_old(a|s)
        L_clip = - E[ min(r * A, clip(r, 1 - eps_low, 1 + eps_high) * A) ]

    Expects:
        - batch["weight"]:       A  (advantages)      [B]
        - batch[old_logp_key]:   log π_old(a|s)      [B]
        - input_ids / attention_mask / action_mask for π_new.

    Defaults follow the GSPO paper clip ranges:
    https://arxiv.org/abs/2507.18071
    """

    clip_eps_low: float = 3e-4
    clip_eps_high: float = 4e-4
    old_logp_key: str = "old_logp_action"
    length_normalize: bool = False
    ratio_clip: Optional[float] = None

    def __post_init__(self) -> None:
        if self.clip_eps_low < 0 or self.clip_eps_high < 0:
            raise ValueError(
                f"clip_eps_low/high must be non-negative, got {self.clip_eps_low}, {self.clip_eps_high}"
            )
        if self.ratio_clip is not None and self.ratio_clip <= 0:
            raise ValueError(f"ratio_clip must be positive, got {self.ratio_clip}")

    @jaxtyped(typechecker=typechecker)
    def compute(self, logits: Logits, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
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

        log_ratio = logp_action - old_logp
        ratio_raw = torch.exp(log_ratio)
        mismatch_kl = ratio_raw - log_ratio - 1.0
        ratio = ratio_raw
        if self.ratio_clip is not None:
            ratio = torch.clamp(ratio, max=self.ratio_clip)

        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio, 1.0 - self.clip_eps_low, 1.0 + self.clip_eps_high
        ) * advantages

        obj = torch.min(unclipped, clipped)
        loss = -obj.mean()

        ppo_clip_frac = (
            (ratio > 1.0 + self.clip_eps_high) | (ratio < 1.0 - self.clip_eps_low)
        ).float().mean()
        if self.ratio_clip is not None:
            ratio_clip_frac = (ratio >= self.ratio_clip).float().mean()
        else:
            ratio_clip_frac = torch.zeros((), device=ratio.device, dtype=ratio.dtype)

        stats = {
            "loss": loss.detach(),
            "ratio_mean": ratio.mean().detach(),
            "ratio_std": ratio.std(unbiased=False).detach(),
            "clip_frac": ppo_clip_frac.detach(),
            "ratio_clip_frac": ratio_clip_frac.detach(),
            "kl_actor_policy": mismatch_kl.mean().detach(),
            "adv_mean": advantages.mean().detach(),
            "adv_std": advantages.std(unbiased=False).detach(),
            "logp_mean": logp_action.mean().detach(),
        }
        return loss, stats


@dataclass
class TokenClippedSurrogateLoss:
    """
    Token-level PPO-style clipped surrogate loss (Token-TIS-style ratios).

    Uses asymmetric clipping: clip(r, 1 - eps_low, 1 + eps_high).

    Expects:
        - batch["weight"]:       A  (advantages)      [B]
        - batch["actor_logps"]:  token logps under behavior policy [B, T]
        - input_ids / attention_mask / action_mask for π_new.

    Defaults follow the GSPO paper clip ranges for the GRPO baseline:
    https://arxiv.org/abs/2507.18071
    """

    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.27
    length_normalize: bool = False
    ratio_clip: Optional[float] = None

    def __post_init__(self) -> None:
        if self.clip_eps_low < 0 or self.clip_eps_high < 0:
            raise ValueError(
                f"clip_eps_low/high must be non-negative, got {self.clip_eps_low}, {self.clip_eps_high}"
            )
        if self.ratio_clip is not None and self.ratio_clip <= 0:
            raise ValueError(f"ratio_clip must be positive, got {self.ratio_clip}")

    @jaxtyped(typechecker=typechecker)
    def compute(self, logits: Logits, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]
        action_mask = batch["action_mask"]
        advantages = batch["weight"]
        if "actor_logps" not in batch:
            raise KeyError("TokenClippedSurrogateLoss requires batch['actor_logps'] for token IS.")

        actor_logps = batch["actor_logps"]
        if actor_logps.shape != input_ids.shape:
            raise ValueError(
                f"actor_logps shape {tuple(actor_logps.shape)} does not match input_ids "
                f"{tuple(input_ids.shape)}."
            )

        token_logp = compute_token_logp(logits, input_ids)  # [B, T-1]
        token_mask = action_mask[:, 1:].to(token_logp.dtype)
        token_counts = token_mask.sum(dim=-1).clamp(min=1.0)
        actor_logps_shifted = actor_logps[:, 1:]

        log_ratio = token_logp - actor_logps_shifted
        ratio_raw = torch.exp(log_ratio)
        token_mismatch_kl = ratio_raw - log_ratio - 1.0
        ratio = ratio_raw
        if self.ratio_clip is not None:
            ratio = torch.clamp(ratio, max=self.ratio_clip)

        ratio_clipped = torch.clamp(
            ratio, 1.0 - self.clip_eps_low, 1.0 + self.clip_eps_high
        )
        adv = advantages.unsqueeze(-1)
        unclipped = ratio * adv
        clipped = ratio_clipped * adv
        obj = torch.min(unclipped, clipped) * token_mask
        per_sample_obj = obj.sum(dim=-1)
        if self.length_normalize:
            per_sample_obj = per_sample_obj / token_counts

        loss = -per_sample_obj.mean()

        mask = token_mask > 0
        if mask.any():
            ratio_vals = ratio.masked_select(mask)
            ppo_clip_frac = (
                (ratio_vals > 1.0 + self.clip_eps_high) | (ratio_vals < 1.0 - self.clip_eps_low)
            ).float().mean()
            ratio_mean = ratio_vals.mean()
            ratio_std = ratio_vals.std(unbiased=False)
            mismatch_kl = token_mismatch_kl.masked_select(mask).mean()
            if self.ratio_clip is not None:
                ratio_clip_frac = (ratio_vals >= self.ratio_clip).float().mean()
            else:
                ratio_clip_frac = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
        else:
            ratio_mean = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            ratio_std = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            ppo_clip_frac = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            ratio_clip_frac = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            mismatch_kl = torch.zeros((), device=ratio.device, dtype=ratio.dtype)

        logp_action = (token_logp * token_mask).sum(dim=-1)
        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "ratio_mean": ratio_mean.detach(),
            "ratio_std": ratio_std.detach(),
            "clip_frac": ppo_clip_frac.detach(),
            "ratio_clip_frac": ratio_clip_frac.detach(),
            "kl_actor_policy": mismatch_kl.detach(),
            "adv_mean": advantages.mean().detach(),
            "adv_std": advantages.std(unbiased=False).detach(),
            "logp_mean": logp_action.mean().detach(),
            "avg_action_tokens": token_counts.mean().detach(),
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

    @jaxtyped(typechecker=typechecker)
    def compute(self, logits: Logits, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
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
            "loss": loss.detach(),
            "kl_mean": kl.mean().detach(),
            "kl_std": kl.std(unbiased=False).detach(),
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

    @jaxtyped(typechecker=typechecker)
    def compute(self, logits: Logits, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
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
            "loss": loss.detach(),
            "entropy_mean": mean_entropy.detach(),
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

    @jaxtyped(typechecker=typechecker)
    def compute(self, logits: Logits, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
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
            stats[f"{term.name}/loss"] = raw_loss.detach()
            stats[f"{term.name}/weight"] = term.weight
            for k, v in term_stats.items():
                stats[f"{term.name}/{k}"] = v

        assert total_loss is not None
        stats["loss"] = total_loss.detach()

        return total_loss, stats
