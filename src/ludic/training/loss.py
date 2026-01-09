from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
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


# ---------------------------------------------------------------------------
# Shared context for memory-efficient loss composition
# ---------------------------------------------------------------------------


_shared_context_var: ContextVar[Optional["SharedContext"]] = ContextVar(
    "ludic_shared_context",
    default=None,
)


def _get_shared_context(
    logits: Logits,
    *,
    batch: Optional[Batch] = None,
    input_ids: Optional[TokenIds] = None,
    action_mask: Optional[Mask] = None,
) -> Optional["SharedContext"]:
    shared = _shared_context_var.get()
    if shared is None:
        return None
    if shared.logits is not logits:
        return None
    if batch is not None and shared.batch is not batch:
        return None
    if input_ids is not None and shared.batch.get("input_ids") is not input_ids:
        return None
    if action_mask is not None and shared.batch.get("action_mask") is not action_mask:
        return None
    return shared


@contextmanager
def _use_shared_context(shared: "SharedContext"):
    token = _shared_context_var.set(shared)
    try:
        yield
    finally:
        _shared_context_var.reset(token)


class SharedContext:
    """
    Lazy-computed shared tensors for memory-efficient loss composition.

    When multiple losses are combined via CompositeLoss, each typically needs
    the same expensive intermediate tensors (e.g., token_logp from log_softmax).
    Without sharing, each loss computes these independently, creating separate
    autograd graphs that store duplicate [B, T, V] activations for backward.

    SharedContext solves this by computing expensive tensors ONCE on first access
    and caching them for subsequent uses. All losses receive the same tensor
    objects, sharing a single autograd graph.

    Memory savings example (7B model, V=32K, B=8, T=4096):
        - Without sharing (2 losses): 2× [B, T, V] ≈ 4GB activations
        - With sharing (2 losses):    1× [B, T, V] ≈ 2GB activations

    Usage:
        # CompositeLoss installs a SharedContext so helpers can reuse cached tensors.
        with _use_shared_context(SharedContext(logits, batch)):
            token_logp = compute_token_logp(logits, batch["input_ids"])

    Note: Properties that depend on batch["actor_logps"] will raise KeyError
    if that key is missing. This is intentional - not all loss combinations
    need actor logprobs.
    """

    __slots__ = ("logits", "batch", "_cache")

    def __init__(self, logits: Logits, batch: Batch) -> None:
        self.logits = logits
        self.batch = batch
        self._cache: Dict[str, Tensor] = {}

    @property
    def input_ids(self) -> TokenIds:
        """Token IDs from batch (not cached, just a convenience accessor)."""
        return self.batch["input_ids"]

    @property
    def action_mask(self) -> Mask:
        """Action mask from batch (not cached, just a convenience accessor)."""
        return self.batch["action_mask"]

    @property
    def token_logp(self) -> Float[Tensor, "B T-1"]:
        """
        Per-token log probabilities: log π(a_t|s_t) for each position.

        THIS IS THE EXPENSIVE OPERATION - calls selective_log_softmax which
        requires storing [B, T, V] activations for backward. Caching this
        is the primary memory optimization.
        """
        if "token_logp" not in self._cache:
            self._cache["token_logp"] = _compute_token_logp_raw(self.logits, self.input_ids)
        return self._cache["token_logp"]

    @property
    def token_mask(self) -> Float[Tensor, "B T-1"]:
        """Action mask aligned with token_logp (shifted by 1 for next-token prediction)."""
        if "token_mask" not in self._cache:
            self._cache["token_mask"] = self.action_mask[:, 1:].to(
                self.token_logp.dtype
            )
        return self._cache["token_mask"]

    @property
    def token_counts(self) -> Float[Tensor, "B"]:
        """Number of action tokens per sample (for length normalization)."""
        if "token_counts" not in self._cache:
            self._cache["token_counts"] = self.token_mask.sum(dim=-1).clamp(min=1.0)
        return self._cache["token_counts"]

    @property
    def actor_logps_shifted(self) -> Float[Tensor, "B T-1"]:
        """
        Behavior policy log probs aligned with token_logp.

        Raises:
            KeyError: If batch["actor_logps"] is not present.
        """
        if "actor_logps_shifted" not in self._cache:
            if "actor_logps" not in self.batch:
                raise KeyError(
                    "SharedContext.actor_logps_shifted requires batch['actor_logps']. "
                    "Ensure your rollouts include actor_logps for ratio-based objectives."
                )
            actor_logps = self.batch["actor_logps"]
            if actor_logps.shape != self.input_ids.shape:
                raise ValueError(
                    f"actor_logps shape {tuple(actor_logps.shape)} does not match input_ids "
                    f"{tuple(self.input_ids.shape)}."
                )
            self._cache["actor_logps_shifted"] = actor_logps[:, 1:]
        return self._cache["actor_logps_shifted"]

    @property
    def log_ratio(self) -> Float[Tensor, "B T-1"]:
        """Log importance ratio: log(π_new/π_old) per token."""
        if "log_ratio" not in self._cache:
            self._cache["log_ratio"] = self.token_logp - self.actor_logps_shifted
        return self._cache["log_ratio"]

    @property
    def ratio(self) -> Float[Tensor, "B T-1"]:
        """Importance ratio: π_new/π_old per token."""
        if "ratio" not in self._cache:
            self._cache["ratio"] = torch.exp(self.log_ratio)
        return self._cache["ratio"]

    def logp_action(self, *, length_normalize: bool = False) -> Float[Tensor, "B"]:
        """
        Sequence-level log probability (sum over action tokens).

        Unlike token_logp, this is a cheap derivation that doesn't require
        additional [B, T, V] storage. The length_normalize flag controls
        whether to divide by number of action tokens.

        Args:
            length_normalize: If True, return mean log prob instead of sum.

        Returns:
            [B] tensor of per-sample log probabilities.
        """
        masked_logp = (self.token_logp * self.token_mask).sum(dim=-1)
        if length_normalize:
            return masked_logp / self.token_counts
        return masked_logp


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
def _selective_log_softmax_impl(
    logits: Logits,
    index: TokenIds,
) -> Float[Tensor, "B T"]:
    """
    Fused kernel for log_softmax + gather.

    Inductor (torch.compile) generates a kernel that computes the log_softmax
    normalization term and selects the target token in a single pass.
    This avoids materializing the massive [B, T, V] probability tensor in VRAM.
    """
    # This looks naive, but the compiler fuses it into a single read/write op.
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)


_USE_TORCH_COMPILE = os.getenv("LUDIC_DISABLE_TORCH_COMPILE", "0") != "1"
_USE_COMPILED_SELECTIVE_LOG_SOFTMAX = _USE_TORCH_COMPILE
if _USE_TORCH_COMPILE:
    _selective_log_softmax_compiled = torch.compile(
        _selective_log_softmax_impl, dynamic=True
    )
else:
    _selective_log_softmax_compiled = _selective_log_softmax_impl


def selective_log_softmax(logits: Logits, index: TokenIds) -> Float[Tensor, "B T"]:
    global _USE_COMPILED_SELECTIVE_LOG_SOFTMAX
    if _USE_COMPILED_SELECTIVE_LOG_SOFTMAX:
        try:
            return _selective_log_softmax_compiled(logits, index)
        except Exception as exc:
            logger.warning(
                "torch.compile failed for selective_log_softmax, falling back to eager: %s",
                exc,
            )
            _USE_COMPILED_SELECTIVE_LOG_SOFTMAX = False
    return _selective_log_softmax_impl(logits, index)

@jaxtyped(typechecker=typechecker)
def _compute_logp_action_raw(
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
        raise ValueError(
            f"Shape mismatch: input_ids {input_ids.shape} vs logits {logits.shape}"
        )

    # Shift for causal LM: logits[t] predicts input_ids[t+1]
    if logits.size(1) < 2:
        raise ValueError("Sequence too short to compute next-token logprobs.")
    logits_shifted = logits[:, :-1, :]  # [B, T-1, V]
    target_ids = input_ids[:, 1:]  # [B, T-1]
    action_mask_shifted = action_mask[:, 1:]  # [B, T-1]

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
def compute_logp_action(
    logits: Logits,
    input_ids: TokenIds,
    action_mask: Mask,
    *,
    length_normalize: bool = False,
) -> Weights:
    shared = _get_shared_context(
        logits,
        input_ids=input_ids,
        action_mask=action_mask,
    )
    if shared is not None:
        return shared.logp_action(length_normalize=length_normalize)
    return _compute_logp_action_raw(
        logits,
        input_ids,
        action_mask,
        length_normalize=length_normalize,
    )


@jaxtyped(typechecker=typechecker)
def _compute_token_logp_raw(
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


@jaxtyped(typechecker=typechecker)
def compute_token_logp(
    logits: Logits,
    input_ids: TokenIds,
) -> Float[Tensor, "B T-1"]:
    shared = _get_shared_context(logits, input_ids=input_ids)
    if shared is not None:
        if "token_logp" not in shared._cache:
            shared._cache["token_logp"] = _compute_token_logp_raw(logits, input_ids)
        return shared._cache["token_logp"]
    return _compute_token_logp_raw(logits, input_ids)


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
    def compute(
        self,
        logits: Logits,
        batch: Batch,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]            # [B, T]
        action_mask = batch["action_mask"]        # [B, T]
        advantages = batch["weight"]              # [B]

        if self.old_logp_key not in batch:
            raise KeyError(f"ReinforceLoss requires '{self.old_logp_key}' in batch.")

        logp_action = compute_logp_action(
            logits, input_ids, action_mask, length_normalize=self.length_normalize
        )  # [B]
        token_counts = action_mask[:, 1:].sum(dim=-1).clamp(min=1.0)

        old_logp = batch[self.old_logp_key]  # [B]
        if self.length_normalize:
            lengths = token_counts.to(old_logp.dtype)
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
    def compute(
        self,
        logits: Logits,
        batch: Batch,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]  # [B, T]
        action_mask = batch["action_mask"]  # [B, T]
        weights = batch.get("weight")

        if logits.ndim != 3:
            raise ValueError(f"Expected logits [B, T, V], got {tuple(logits.shape)}")
        if input_ids.shape != logits.shape[:2]:
            raise ValueError(
                f"Shape mismatch: input_ids {input_ids.shape} vs logits {logits.shape}"
            )

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
    def compute(
        self,
        logits: Logits,
        batch: Batch,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        action_mask = batch["action_mask"]
        adv_raw = batch["weight"]                # [B]

        if self.old_logp_key not in batch:
            raise KeyError(f"ReinforceBaselineLoss requires '{self.old_logp_key}' in batch.")

        input_ids = batch["input_ids"]
        logp_action = compute_logp_action(
            logits, input_ids, action_mask, length_normalize=self.length_normalize
        )  # [B]
        token_counts = action_mask[:, 1:].sum(dim=-1).clamp(min=1.0)

        old_logp = batch[self.old_logp_key]  # [B]
        if self.length_normalize:
            lengths = token_counts.to(old_logp.dtype)
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
    def compute(
        self,
        logits: Logits,
        batch: Batch,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        action_mask = batch["action_mask"]
        advantages = batch["weight"]              # [B]
        if self.old_logp_key not in batch:
            raise KeyError(
                f"ClippedSurrogateLoss requires '{self.old_logp_key}' in batch."
            )

        input_ids = batch["input_ids"]
        logp_action = compute_logp_action(
            logits,
            input_ids,
            action_mask,
            length_normalize=self.length_normalize,
        )  # [B]
        token_counts = action_mask[:, 1:].sum(dim=-1).clamp(min=1.0)

        old_logp = batch[self.old_logp_key]  # [B]
        if self.length_normalize:
            lengths = token_counts.to(old_logp.dtype)
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

        # Token-weighted clip fraction: counts tokens in sequences where the
        # clipped branch is active (sequence-level GSPO-style metric).
        token_mask = action_mask[:, 1:].to(dtype=ratio.dtype)
        token_counts = token_mask.sum(dim=-1).clamp(min=1.0)
        adv_pos = advantages >= 0
        seq_clipped = torch.where(
            adv_pos,
            ratio > 1.0 + self.clip_eps_high,
            ratio < 1.0 - self.clip_eps_low,
        )
        total_tokens = token_counts.sum()
        if total_tokens > 0:
            ppo_clip_frac = (seq_clipped.to(token_counts.dtype) * token_counts).sum() / total_tokens
        else:
            ppo_clip_frac = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
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
class CISPOLoss:
    """
    CISPO (Clipped IS-weight Policy Optimization) loss.

    Unlike PPO/GRPO which clip the policy update via the min operation,
    CISPO clips the importance sampling weight itself and uses it as a
    stop-gradient multiplier on the REINFORCE objective. This preserves
    gradient contributions from all tokens, especially low-probability
    tokens crucial for reflective reasoning (e.g., "However", "Recheck").

    Loss:
        L = - E[ sg(clip(r_t, 1-ε_low, 1+ε_high)) * A * log π(a_t|s_t) ]

    Where:
        - r_t = π_new(a_t|s_t) / π_old(a_t|s_t) is the importance sampling weight
        - sg(·) is stop-gradient (weight does not receive gradients)
        - A is the advantage (from batch["weight"])

    Expects:
        - batch["weight"]:       A (advantages)       [B]
        - batch["actor_logps"]:  token logps under behavior policy [B, T]
        - input_ids / attention_mask / action_mask for π_new.

    Reference: MiniMax-M1 paper (arXiv:2506.13585)
    """

    clip_eps_low: float = 1e6  # Effectively no lower bound (paper setting)
    clip_eps_high: float = 0.2
    length_normalize: bool = True

    def __post_init__(self) -> None:
        if self.clip_eps_high < 0:
            raise ValueError(f"clip_eps_high must be non-negative, got {self.clip_eps_high}")

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Logits,
        batch: Batch,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        action_mask = batch["action_mask"]
        advantages = batch["weight"]  # [B]

        if "actor_logps" not in batch:
            raise KeyError("CISPOLoss requires batch['actor_logps'] for importance sampling.")

        input_ids = batch["input_ids"]
        actor_logps = batch["actor_logps"]  # [B, T]
        if actor_logps.shape != input_ids.shape:
            raise ValueError(
                f"actor_logps shape {tuple(actor_logps.shape)} does not match input_ids "
                f"{tuple(input_ids.shape)}."
            )

        # Compute token log probs under current policy
        token_logp = compute_token_logp(logits, input_ids)  # [B, T-1]
        token_mask = action_mask[:, 1:].to(token_logp.dtype)  # [B, T-1]
        token_counts = token_mask.sum(dim=-1).clamp(min=1.0)  # [B]
        actor_logps_shifted = actor_logps[:, 1:]  # [B, T-1]

        # Compute importance sampling ratios
        log_ratio = token_logp - actor_logps_shifted
        ratio = torch.exp(log_ratio)

        # CISPO: Clip the IS weight, not the update
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_eps_low,
            1.0 + self.clip_eps_high,
        )

        # Stop-gradient on the clipped IS weight
        # The gradient only flows through log π, not through the IS weight
        is_weight = clipped_ratio.detach()

        # REINFORCE loss with IS weight correction
        # L = - E[is_weight * A * log π]
        adv = advantages.unsqueeze(-1)  # [B, 1]
        weighted_logp = is_weight * adv * token_logp * token_mask  # [B, T-1]

        per_sample_obj = weighted_logp.sum(dim=-1)  # [B]
        if self.length_normalize:
            per_sample_obj = per_sample_obj / token_counts

        loss = -per_sample_obj.mean()

        # Stats
        mask = token_mask > 0
        if mask.any():
            ratio_vals = ratio.masked_select(mask)
            ratio_mean = ratio_vals.mean()
            ratio_std = ratio_vals.std(unbiased=False)
            # Fraction of tokens where IS weight was clipped
            clip_frac = (
                (ratio_vals > 1.0 + self.clip_eps_high) |
                (ratio_vals < 1.0 - self.clip_eps_low)
            ).float().mean()
            # KL approximation: r - log(r) - 1
            token_kl = ratio_vals - log_ratio.masked_select(mask) - 1.0
            mismatch_kl = token_kl.mean()
        else:
            ratio_mean = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            ratio_std = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            clip_frac = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            mismatch_kl = torch.zeros((), device=ratio.device, dtype=ratio.dtype)

        logp_action = (token_logp * token_mask).sum(dim=-1)
        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "ratio_mean": ratio_mean.detach(),
            "ratio_std": ratio_std.detach(),
            "clip_frac": clip_frac.detach(),
            "kl_actor_policy": mismatch_kl.detach(),
            "adv_mean": advantages.mean().detach(),
            "adv_std": advantages.std(unbiased=False).detach(),
            "logp_mean": logp_action.mean().detach(),
            "avg_action_tokens": token_counts.mean().detach(),
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
    def compute(
        self,
        logits: Logits,
        batch: Batch,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        action_mask = batch["action_mask"]
        advantages = batch["weight"]
        if "actor_logps" not in batch:
            raise KeyError("TokenClippedSurrogateLoss requires batch['actor_logps'] for token IS.")

        input_ids = batch["input_ids"]
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


@dataclass
class SAPOLoss:
    """
    SAPO (Soft Adaptive Policy Optimization) loss.

    SAPO replaces hard clipping with a smooth, temperature-controlled sigmoid gate
    that adaptively attenuates off-policy updates while preserving useful learning
    signals. Unlike hard clipping (GRPO) or sequence-level gates (GSPO), SAPO
    applies a soft trust region at the token level that naturally yields sequence-level
    coherence under mild conditions.

    Core idea:
        Instead of hard clipping: min(r * A, clip(r, 1-ε, 1+ε) * A)
        SAPO uses soft gate:   f(r) * A, where f(r) = (4/τ) * σ(τ(r - 1))

    The sigmoid gate σ(τ(r - 1)) peaks at r=1 and decays smoothly as r deviates,
    implementing a continuous trust region. The temperature τ controls decay rate:
    larger τ → faster decay → more conservative updates.

    Asymmetric temperatures:
        - τ_pos for positive advantages (token logit should increase)
        - τ_neg for negative advantages (token logit should decrease)

    Setting τ_neg > τ_pos makes negative gradients decay faster, improving stability.
    Rationale: Negative updates diffuse to many unsampled tokens in a large vocabulary,
    introducing more noise than positive updates which focus on the sampled token.

    Objective:
        J_SAPO = E[ (1/|o|) Σ_t f(ρ_t) * A ]
        where f(r) = (4/τ) * σ(τ(r - 1))
        and τ = τ_pos if A > 0 else τ_neg

    Gradient weight (from differentiating f):
        w(r) = 4 * p(r) * (1 - p(r)), where p(r) = σ(τ(r - 1))
        This peaks at r=1 with value 1 and decays smoothly.

    Connection to other methods:
        - Under mild conditions (small steps, low token variance), SAPO reduces to
          sequence-level optimization like GSPO but with smooth gating
        - Compared to GRPO's hard token clipping, SAPO provides smooth scaling
        - Compared to GSPO's sequence-level hard clipping, SAPO is token-adaptive

    Expects:
        - batch["weight"]:       A (advantages)       [B]
        - batch["actor_logps"]:  token logps under behavior policy [B, T]
        - input_ids / attention_mask / action_mask for π_new

    Reference: "Soft Adaptive Policy Optimization" (arXiv:2511.20347v2)
    https://arxiv.org/abs/2511.20347
    """

    tau_pos: float = 1.0    # Temperature for positive advantages
    tau_neg: float = 1.05   # Temperature for negative advantages (higher for stability)
    length_normalize: bool = False  # Normalize by sequence length

    def __post_init__(self) -> None:
        if self.tau_pos <= 0 or self.tau_neg <= 0:
            raise ValueError(
                f"tau_pos/tau_neg must be positive, got {self.tau_pos}, {self.tau_neg}"
            )

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Logits,
        batch: Batch,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        action_mask = batch["action_mask"]
        advantages = batch["weight"]  # [B]

        if "actor_logps" not in batch:
            raise KeyError("SAPOLoss requires batch['actor_logps'] for token IS.")

        input_ids = batch["input_ids"]
        actor_logps = batch["actor_logps"]
        if actor_logps.shape != input_ids.shape:
            raise ValueError(
                f"actor_logps shape {tuple(actor_logps.shape)} does not match input_ids "
                f"{tuple(input_ids.shape)}."
            )

        # Compute token-level log probabilities
        token_logp = compute_token_logp(logits, input_ids)  # [B, T-1]
        token_mask = action_mask[:, 1:].to(token_logp.dtype)  # [B, T-1]
        token_counts = token_mask.sum(dim=-1).clamp(min=1.0)  # [B]
        actor_logps_shifted = actor_logps[:, 1:]  # [B, T-1]

        # Compute importance ratios
        log_ratio = token_logp - actor_logps_shifted  # [B, T-1]
        ratio = torch.exp(log_ratio)  # [B, T-1]

        # Select temperature based on advantage sign
        # Use where to select between tau_pos and tau_neg without creating new tensors
        adv_positive = advantages > 0  # [B]
        tau_pos_val = self.tau_pos
        tau_neg_val = self.tau_neg

        # Compute soft gate for positive and negative advantages separately
        # This allows kernel fusion since we're not creating tensors in the graph
        ratio_minus_1 = ratio - 1.0  # [B, T-1]

        # For positive advantages: f(r) = (4/τ_pos) * σ(τ_pos * (r - 1))
        # For negative advantages: f(r) = (4/τ_neg) * σ(τ_neg * (r - 1))
        sigmoid_arg_pos = tau_pos_val * ratio_minus_1  # [B, T-1]
        sigmoid_arg_neg = tau_neg_val * ratio_minus_1  # [B, T-1]

        gate_pos = torch.sigmoid(sigmoid_arg_pos)  # [B, T-1]
        gate_neg = torch.sigmoid(sigmoid_arg_neg)  # [B, T-1]

        soft_gate_pos = (4.0 / tau_pos_val) * gate_pos  # [B, T-1]
        soft_gate_neg = (4.0 / tau_neg_val) * gate_neg  # [B, T-1]

        # Select based on advantage sign (broadcast over tokens)
        adv_positive_expanded = adv_positive.unsqueeze(-1)  # [B, 1]
        soft_gate = torch.where(adv_positive_expanded, soft_gate_pos, soft_gate_neg)  # [B, T-1]

        # Apply gate to advantages (broadcast advantages over tokens)
        adv_expanded = advantages.unsqueeze(-1)  # [B, 1]
        gated_obj = soft_gate * adv_expanded * token_mask  # [B, T-1]

        # Aggregate over tokens
        per_sample_obj = gated_obj.sum(dim=-1)  # [B]
        if self.length_normalize:
            per_sample_obj = per_sample_obj / token_counts

        loss = -per_sample_obj.mean()

        # --- Stats computation ---
        # Gradient weight: w(r) = 4 * p(r) * (1 - p(r))
        # Select the correct gate based on advantage sign
        gate_selected = torch.where(adv_positive_expanded, gate_pos, gate_neg)  # [B, T-1]
        grad_weight = 4.0 * gate_selected * (1.0 - gate_selected)  # [B, T-1]

        # Compute KL for monitoring
        token_mismatch_kl = ratio - log_ratio - 1.0  # [B, T-1]

        mask = token_mask > 0
        if mask.any():
            ratio_vals = ratio.masked_select(mask)
            ratio_mean = ratio_vals.mean()
            ratio_std = ratio_vals.std(unbiased=False)
            mismatch_kl = token_mismatch_kl.masked_select(mask).mean()

            # Average gradient weight (for monitoring soft gating)
            grad_weight_vals = grad_weight.masked_select(mask)
            grad_weight_mean = grad_weight_vals.mean()
            grad_weight_std = grad_weight_vals.std(unbiased=False)
        else:
            ratio_mean = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            ratio_std = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            mismatch_kl = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            grad_weight_mean = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
            grad_weight_std = torch.zeros((), device=ratio.device, dtype=ratio.dtype)

        logp_action = (token_logp * token_mask).sum(dim=-1)
        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "ratio_mean": ratio_mean.detach(),
            "ratio_std": ratio_std.detach(),
            "grad_weight_mean": grad_weight_mean.detach(),
            "grad_weight_std": grad_weight_std.detach(),
            "kl_actor_policy": mismatch_kl.detach(),
            "adv_mean": advantages.mean().detach(),
            "adv_std": advantages.std(unbiased=False).detach(),
            "logp_mean": logp_action.mean().detach(),
            "avg_action_tokens": token_counts.mean().detach(),
        }
        return loss, stats


@dataclass
class GMPOLoss:
    """
    GMPO (Geometric-Mean Policy Optimization) loss.

    GMPO stabilizes GRPO by using the geometric mean of token-level importance ratios
    instead of the arithmetic mean. This makes the objective less sensitive to outliers
    and results in more stable policy updates.

    Objective:
        J_GMPO = E[ (∏_t min(ρ_t * A, clip(ρ_t, 1-ε_low, 1+ε_high) * A))^(1/|o|) * sgn(A) ]

    where:
        - ρ_t = π_new(a_t|s_t) / π_old(a_t|s_t) is the token-level importance ratio
        - A is the advantage (from batch["weight"])
        - |o| is the sequence length (normalization factor)
        - sgn(A) ensures correct optimization direction

    Key differences from GRPO (TokenClippedSurrogateLoss):
        1. Uses geometric mean instead of arithmetic mean (more robust to outliers)
        2. Applies token-level clipping (not sequence-level)
        3. Supports wider clipping ranges (e.g., (e^-0.4, e^0.4) instead of (0.8, 1.2))
        4. Results in more stable importance sampling ratios during training

    Implementation details:
        - All operations performed in log-space for numerical stability
        - Clipping applied at token level before geometric mean computation
        - Normalization by sequence length (1/|o|) is critical for stability

    Expects:
        - batch["weight"]:       A (advantages)       [B]
        - batch["actor_logps"]:  token logps under behavior policy [B, T]
        - input_ids / attention_mask / action_mask for π_new.

    Reference: "GMPO: Geometric-Mean Policy Optimization" (arXiv:2507.20673v3)
    Defaults follow the GMPO paper settings with wider clipping (e^-0.4, e^0.4).
    """

    clip_eps_low: float = 0.4      # In log-space: clip to e^-0.4 ≈ 0.67
    clip_eps_high: float = 0.4     # In log-space: clip to e^0.4 ≈ 1.49
    length_normalize: bool = True  # 1/|o| normalization (critical for GMPO)
    ratio_clip: Optional[float] = None

    def __post_init__(self) -> None:
        if self.clip_eps_low < 0 or self.clip_eps_high < 0:
            raise ValueError(
                f"clip_eps_low/high must be non-negative, got {self.clip_eps_low}, {self.clip_eps_high}"
            )
        if self.ratio_clip is not None and self.ratio_clip <= 0:
            raise ValueError(f"ratio_clip must be positive, got {self.ratio_clip}")

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Logits,
        batch: Batch,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        action_mask = batch["action_mask"]
        advantages = batch["weight"]  # [B]

        if "actor_logps" not in batch:
            raise KeyError("GMPOLoss requires batch['actor_logps'] for token IS.")

        shared = _get_shared_context(logits, batch=batch)
        if shared is not None:
            token_logp = shared.token_logp
            token_mask = shared.token_mask
            token_counts = shared.token_counts
            actor_logps_shifted = shared.actor_logps_shifted
        else:
            input_ids = batch["input_ids"]
            actor_logps = batch["actor_logps"]
            if actor_logps.shape != input_ids.shape:
                raise ValueError(
                    f"actor_logps shape {tuple(actor_logps.shape)} does not match input_ids "
                    f"{tuple(input_ids.shape)}."
                )

            # Compute token-level log probabilities
            token_logp = compute_token_logp(logits, input_ids)  # [B, T-1]
            token_mask = action_mask[:, 1:].to(token_logp.dtype)  # [B, T-1]
            token_counts = token_mask.sum(dim=-1).clamp(min=1.0)  # [B]
            actor_logps_shifted = actor_logps[:, 1:]  # [B, T-1]

        # Compute log importance ratios (in log-space for numerical stability)
        log_ratio = token_logp - actor_logps_shifted  # [B, T-1]

        # Sign of advantage (for correct optimization direction)
        sgn_adv = torch.sign(advantages).unsqueeze(-1)  # [B, 1]

        # Apply advantage sign to log ratios: sgn(A) * log(ρ_t)
        sgn_log_ratio = sgn_adv * log_ratio  # [B, T-1]

        # Token-level clipping in log-space
        # clip(sgn(A) * log(ρ_t), -ε_low, ε_high)
        sgn_log_ratio_clipped = torch.clamp(
            sgn_log_ratio,
            -self.clip_eps_low,
            self.clip_eps_high
        )  # [B, T-1]

        # Take min of unclipped and clipped (still in log-space, signed)
        sgn_log_ratio_min = torch.min(sgn_log_ratio, sgn_log_ratio_clipped)  # [B, T-1]

        # Remove sign to get actual log ratios for geometric mean
        log_ratio_min = sgn_adv * sgn_log_ratio_min  # [B, T-1]

        # Geometric mean: exp(sum(log(ρ_t)) / |o|) = exp(mean(log(ρ_t)))
        # Only sum over valid tokens (token_mask == 1)
        sum_log_ratio = (log_ratio_min * token_mask).sum(dim=-1)  # [B]

        if self.length_normalize:
            # Normalize by sequence length: 1/|o| * sum(log(ρ_t))
            geom_mean_log_ratio = sum_log_ratio / token_counts  # [B]
        else:
            geom_mean_log_ratio = sum_log_ratio  # [B]

        # Convert back from log-space: ∏_t ρ_t^(1/|o|)
        geom_mean_ratio = torch.exp(geom_mean_log_ratio)  # [B]

        # Optional ratio clipping (after geometric mean)
        if self.ratio_clip is not None:
            geom_mean_ratio = torch.clamp(geom_mean_ratio, max=self.ratio_clip)

        # Objective: geom_mean_ratio * A (advantage sign already handled in clipping)
        obj = geom_mean_ratio * advantages  # [B]
        loss = -obj.mean()

        # --- Stats computation ---
        # Compute raw ratios for monitoring (not used in loss)
        ratio_raw = torch.exp(log_ratio)  # [B, T-1]
        token_mismatch_kl = ratio_raw - log_ratio - 1.0  # [B, T-1]

        mask = token_mask > 0
        if mask.any():
            ratio_vals = ratio_raw.masked_select(mask)

            # Clip fraction in original ratio space (for comparison with GRPO)
            # Note: GMPO clips in log-space, so we convert bounds
            lower_bound = torch.exp(torch.tensor(-self.clip_eps_low, device=ratio_vals.device))
            upper_bound = torch.exp(torch.tensor(self.clip_eps_high, device=ratio_vals.device))
            ppo_clip_frac = (
                (ratio_vals > upper_bound) | (ratio_vals < lower_bound)
            ).float().mean()

            ratio_mean = ratio_vals.mean()
            ratio_std = ratio_vals.std(unbiased=False)
            mismatch_kl = token_mismatch_kl.masked_select(mask).mean()

            if self.ratio_clip is not None:
                ratio_clip_frac = (geom_mean_ratio >= self.ratio_clip).float().mean()
            else:
                ratio_clip_frac = torch.zeros((), device=ratio_vals.device, dtype=ratio_vals.dtype)
        else:
            ratio_mean = torch.zeros((), device=log_ratio.device, dtype=log_ratio.dtype)
            ratio_std = torch.zeros((), device=log_ratio.device, dtype=log_ratio.dtype)
            ppo_clip_frac = torch.zeros((), device=log_ratio.device, dtype=log_ratio.dtype)
            ratio_clip_frac = torch.zeros((), device=log_ratio.device, dtype=log_ratio.dtype)
            mismatch_kl = torch.zeros((), device=log_ratio.device, dtype=log_ratio.dtype)

        logp_action = (token_logp * token_mask).sum(dim=-1)
        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "ratio_mean": ratio_mean.detach(),
            "ratio_std": ratio_std.detach(),
            "geom_mean_ratio_mean": geom_mean_ratio.mean().detach(),
            "geom_mean_ratio_std": geom_mean_ratio.std(unbiased=False).detach(),
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
class TokenKLLoss:
    """
    Token-level KL penalty between π_new and a reference policy.

    Uses the standard policy-gradient surrogate estimate:

        KL(π_new || π_old) ≈ E_{a ~ π_new} [ log π_new(a|s) - log π_old(a|s) ]

    Computed over action tokens and averaged per sequence if length_normalize=True.
    """

    coeff: float = 1.0
    old_logp_key: str = "actor_logps"
    length_normalize: bool = True

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Logits,
        batch: Batch,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        action_mask = batch["action_mask"]

        if self.old_logp_key not in batch:
            raise KeyError(f"TokenKLLoss requires batch['{self.old_logp_key}'].")

        shared = _get_shared_context(logits, batch=batch)
        if shared is not None:
            token_logp = shared.token_logp
            token_mask = shared.token_mask
            token_counts = shared.token_counts
            if self.old_logp_key == "actor_logps":
                old_logps_shifted = shared.actor_logps_shifted
            else:
                input_ids = batch["input_ids"]
                old_logps = batch[self.old_logp_key]
                if old_logps.shape != input_ids.shape:
                    raise ValueError(
                        f"{self.old_logp_key} shape {tuple(old_logps.shape)} does not match input_ids "
                        f"{tuple(input_ids.shape)}."
                    )
                old_logps_shifted = old_logps[:, 1:]
        else:
            input_ids = batch["input_ids"]
            old_logps = batch[self.old_logp_key]
            if old_logps.shape != input_ids.shape:
                raise ValueError(
                    f"{self.old_logp_key} shape {tuple(old_logps.shape)} does not match input_ids "
                    f"{tuple(input_ids.shape)}."
                )
            token_logp = compute_token_logp(logits, input_ids)
            token_mask = action_mask[:, 1:].to(token_logp.dtype)
            token_counts = token_mask.sum(dim=-1).clamp(min=1.0)
            old_logps_shifted = old_logps[:, 1:]

        token_kl = (token_logp - old_logps_shifted) * token_mask
        per_sample_kl = token_kl.sum(dim=-1)
        if self.length_normalize:
            per_sample_kl = per_sample_kl / token_counts

        loss = self.coeff * per_sample_kl.mean()

        stats: Dict[str, Any] = {
            "loss": loss.detach(),
            "kl_mean": per_sample_kl.mean().detach(),
            "kl_std": per_sample_kl.std(unbiased=False).detach(),
            "avg_action_tokens": token_counts.mean().detach(),
        }
        return loss, stats


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
    def compute(
        self,
        logits: Logits,
        batch: Batch,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        action_mask = batch["action_mask"]
        old_logp = batch[self.old_logp_key]       # [B]

        input_ids = batch["input_ids"]
        logp_new = compute_logp_action(
            logits,
            input_ids,
            action_mask,
            length_normalize=self.length_normalize,
        )  # [B]
        shared = _get_shared_context(logits, batch=batch)
        if shared is not None:
            token_counts = shared.token_counts
        else:
            token_counts = action_mask[:, 1:].sum(dim=-1).clamp(min=1.0)

        if self.length_normalize:
            lengths = token_counts.to(old_logp.dtype)
            old_logp = old_logp / lengths

        kl = logp_new - old_logp  # [B]
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
    def compute(
        self,
        logits: Logits,
        batch: Batch,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        action_mask = batch["action_mask"]

        logprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)

        # token entropy: [B, T]
        token_entropy = -(probs * logprobs).sum(dim=-1)

        mask = action_mask.to(token_entropy.dtype)

        masked_entropy = token_entropy * mask  # [B, T]
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

    Memory Efficiency:
        CompositeLoss automatically creates a SharedContext to cache expensive
        intermediate tensors (like token_logp from log_softmax). All child losses
        can access the same SharedContext implicitly, sharing cached tensors and
        avoiding duplicate autograd graphs.

        Without SharedContext: N losses → N× [B, T, V] autograd activations
        With SharedContext:    N losses → 1× [B, T, V] autograd activations

    Note:
        SharedContext is made available implicitly during computation so
        compute_logp_action/compute_token_logp can reuse cached tensors.
    """

    terms: List[LossTerm]

    @jaxtyped(typechecker=typechecker)
    def compute(
        self,
        logits: Logits,
        batch: Batch,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        if not self.terms:
            raise ValueError("CompositeLoss.terms must be non-empty")

        # Create shared context for memory-efficient tensor sharing.
        shared = SharedContext(logits, batch)

        total_loss: Tensor | None = None
        stats: Dict[str, Any] = {}

        with _use_shared_context(shared):
            for term in self.terms:
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
