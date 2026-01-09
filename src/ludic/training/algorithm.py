from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol

from jaxtyping import Float
from torch import nn, Tensor

from ludic.training.types import CreditAssigner, SAWBatch
from ludic.training.loss import (
    Loss,
    ReinforceLoss,
    ReinforceBaselineLoss,
    ClippedSurrogateLoss,
    TokenClippedSurrogateLoss,
    CISPOLoss,
    SAPOLoss,
    GMPOLoss,
    MaskedCausalLMCrossEntropyLoss,
    CompositeLoss,
    LossTerm,
    TokenKLLoss,
)
from ludic.training.credit_assignment import (
    MonteCarloReturn,
    GroupNormalizedReturn,
    HybridNormalizedReturn,
    ConstantCredit,
)


Batch = Mapping[str, Tensor]
Logits = Float[Tensor, "B T V"]

class PreprocessFn(Protocol):
    def __call__(self, saw_batch: SAWBatch) -> SAWBatch: ...


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
        *,
        cast_logits_to_fp32: bool = False,
    ) -> tuple[Tensor, Dict[str, Any]]:
        """
        Runs the forward pass once and delegates to the Loss object.

        Args:
            model: The trainable model.
            batch: Collated batch tensors (input_ids, attention_mask, etc.).
            cast_logits_to_fp32: If True, cast logits to FP32 before loss computation.
                This improves importance sampling ratio stability for ratio-based
                objectives (GRPO, CISPO, etc.) by reducing precision errors in
                exp(log_ratio). Recommended by ScaleRL paper (arXiv:2510.13786).
        """
        # --- Run the forward pass ---
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits: Logits = outputs.logits

        # ScaleRL: FP32 logits prevent IS ratio precision issues in exp(logp_new - logp_old)
        if cast_logits_to_fp32:
            logits = logits.float()

        # Pass the resulting logits to the loss function
        return self.loss.compute(logits, batch)


# ---------------------------------------------------------------------------
# Behavior logprob requirements (ratio-based objectives)
# ---------------------------------------------------------------------------


def validate_actor_logps(
    saw_batch: SAWBatch,
) -> SAWBatch:
    """
    Validate SAWItems carry per-token logprobs under the behavior policy (actor).

    Contract:
      - For ratio objectives (PPO/GRPO/KL-to-behavior), each SAWItem must carry
        `item.actor_logps` (backed by `item.attachments.actor_logps`), computed at rollout time by the inference
        client and propagated through batching/collation.

    This function validates the contract; it does not backfill or recompute logprobs.
    """
    items = saw_batch.items

    missing = []
    for i, it in enumerate(items):
        actor = it.actor_logps
        if actor is None:
            missing.append(i)
            continue
        expected_len = int(sum(int(x) for x in it.action_mask))
        if len(actor.token_logps) != expected_len:
            raise ValueError(
                "ActorTokenLogps length mismatch for item "
                f"(index={i}, rollout_id={it.meta.get('rollout_id')!r}, step_index={it.meta.get('step_index')!r}): "
                f"expected {expected_len}, got {len(actor.token_logps)}."
            )
        if not isinstance(actor.token_logps, list) or not all(
            isinstance(v, (int, float)) for v in actor.token_logps
        ):
            raise TypeError("ActorTokenLogps.token_logps must be a List[float].")

    if missing:
        raise ValueError(
            "Missing SampleAttachments.actor_logps for a ratio-based objective. "
            "Ensure your inference client returns chosen-token logprobs and your batch collation "
            "populates SAWItem.attachments.actor_logps (e.g., via ReturnSpec.for_rl()). "
            f"Missing indices: {missing}."
        )

    return saw_batch


def drop_zero_weight_samples(
    saw_batch: SAWBatch,
    *,
    eps: float = 1e-4,
) -> SAWBatch:
    """
    Drop samples with near-zero credit weight before collation.
    """
    if eps < 0:
        raise ValueError("eps must be >= 0.")
    items = [it for it in saw_batch.items if abs(float(it.weight)) > eps]
    saw_batch.items = items
    return saw_batch


def compose_preprocess(*fns: PreprocessFn) -> PreprocessFn:
    def _composed(batch: SAWBatch) -> SAWBatch:
        for fn in fns:
            batch = fn(batch)
        return batch
    return _composed


# ---------------------------------------------------------------------------
# Presets: REINFORCE and REINFORCE+baseline
# ---------------------------------------------------------------------------


def make_reinforce(
    *,
    gamma: float = 1.0,
    drop_zero_weight: bool = False,
    drop_zero_weight_eps: float = 1e-4,
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

    Set drop_zero_weight=True to drop zero-advantage samples before collation.
    """
    credit_assigner: CreditAssigner = MonteCarloReturn(gamma=gamma)
    loss: Loss = ReinforceLoss()

    preprocess = None
    if drop_zero_weight:
        preprocess = lambda batch: drop_zero_weight_samples(batch, eps=drop_zero_weight_eps)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=preprocess,
    )


def make_reinforce_baseline(
    *,
    gamma: float = 1.0,
    name: str = "reinforce_baseline",
    normalize_adv: bool = False,
    drop_zero_weight: bool = False,
    drop_zero_weight_eps: float = 1e-4,
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

    Set drop_zero_weight=True to drop zero-advantage samples before collation.
    """
    credit_assigner: CreditAssigner = MonteCarloReturn(gamma=gamma)
    loss: Loss = ReinforceBaselineLoss(
        normalize=normalize_adv,
    )

    preprocess = None
    if drop_zero_weight:
        preprocess = lambda batch: drop_zero_weight_samples(batch, eps=drop_zero_weight_eps)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=preprocess,
    )


def make_grpo(
    *,
    group_size: int,
    group_normalize_adv: bool = True,
    positive_only: bool = False,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.27,
    length_normalize: bool = False,
    ratio_clip: Optional[float] = None,
    drop_zero_weight: bool = False,
    drop_zero_weight_eps: float = 1e-4,
    name: str = "grpo",
) -> RLAlgorithm:
    """
    GRPO-style preset (clipped surrogate, token-level ratios by default):

      - Credit assignment: group-normalized returns (per-group baseline)
      - Loss: PPO-style clipped surrogate (policy term only)

    Rollouts must carry `group_id` in their metadata and each group must
    have exactly `group_size` members. Raises ValueError otherwise.

    Args:
        group_size: Number of rollouts per group.
        group_normalize_adv: Whether to normalize advantages within each group.
        positive_only: If True, clip negative advantages to zero (reinforce-only).
        clip_eps_low: Lower PPO clipping epsilon for the surrogate objective.
            Defaults follow GSPO paper settings (https://arxiv.org/abs/2507.18071).
        clip_eps_high: Upper PPO clipping epsilon for the surrogate objective.
            Defaults follow GSPO paper settings (https://arxiv.org/abs/2507.18071).
        length_normalize: Whether to normalize log-probs by action length.
        ratio_clip: Optional upper bound C for truncation (min(r, C)).
        name: Algorithm name for logging/metrics.
    Note: For the clipped ratio objective, we need behavior-policy logprobs.
    This preset installs a preprocessor that validates
    `item.actor_logps` is present.

    drop_zero_weight defaults to True to skip zero-advantage samples.
    """
    credit_assigner: CreditAssigner = GroupNormalizedReturn(
        group_size=group_size,
        normalize_adv=group_normalize_adv,
        positive_only=positive_only,
    )
    loss: Loss = TokenClippedSurrogateLoss(
        clip_eps_low=clip_eps_low,
        clip_eps_high=clip_eps_high,
        length_normalize=length_normalize,
        ratio_clip=ratio_clip,
    )
    preprocess_fns = []
    if drop_zero_weight:
        preprocess_fns.append(lambda batch: drop_zero_weight_samples(batch, eps=drop_zero_weight_eps))
    preprocess_fns.append(validate_actor_logps)
    preprocess = compose_preprocess(*preprocess_fns)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=preprocess,
    )


def make_dr_grpo(
    *,
    group_size: int,
    positive_only: bool = False,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.27,
    length_normalize: bool = False,
    ratio_clip: Optional[float] = None,
    drop_zero_weight: bool = False,
    drop_zero_weight_eps: float = 1e-4,
    name: str = "dr_grpo",
) -> RLAlgorithm:
    """
    Dr. GRPO (GRPO Done Right): removes per-response length normalization and
    per-group std normalization while keeping the GRPO-style clipped surrogate.

      - Credit assignment: group-mean baseline only (no std normalization)
      - Loss: token-level PPO-style clipped surrogate (Token-TIS)

    This corresponds to the unbiased GRPO variant described in
    "Understanding R1-Zero-Like Training: A Critical Perspective".

    Args:
        group_size: Number of rollouts per group.
        positive_only: If True, clip negative advantages to zero.
        clip_eps_low: Lower PPO clipping epsilon for the surrogate objective.
        clip_eps_high: Upper PPO clipping epsilon for the surrogate objective.
        length_normalize: If True, normalizes by number of action tokens.
            This reintroduces length normalization and deviates from Dr. GRPO.
        ratio_clip: Optional upper bound C for truncation (min(r, C)).
        name: Algorithm name for logging/metrics.
    """
    credit_assigner: CreditAssigner = GroupNormalizedReturn(
        group_size=group_size,
        normalize_adv=False,
        positive_only=positive_only,
    )
    loss: Loss = TokenClippedSurrogateLoss(
        clip_eps_low=clip_eps_low,
        clip_eps_high=clip_eps_high,
        length_normalize=length_normalize,
        ratio_clip=ratio_clip,
    )
    preprocess_fns = []
    if drop_zero_weight:
        preprocess_fns.append(lambda batch: drop_zero_weight_samples(batch, eps=drop_zero_weight_eps))
    preprocess_fns.append(validate_actor_logps)
    preprocess = compose_preprocess(*preprocess_fns)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=preprocess,
    )


def make_gspo(
    *,
    group_size: int,
    group_normalize_adv: bool = True,
    positive_only: bool = False,
    clip_eps_low: float = 3e-4,
    clip_eps_high: float = 4e-4,
    length_normalize: bool = True,
    ratio_clip: Optional[float] = None,
    drop_zero_weight: bool = False,
    drop_zero_weight_eps: float = 1e-4,
    name: str = "gspo",
) -> RLAlgorithm:
    """
    GSPO-style preset (sequence-level ratios with sequence-level clipping).

    This mirrors GRPO's group-normalized advantages but uses a sequence-level
    importance ratio. With length_normalize=True, this matches the geometric
    mean ratio used in the GSPO paper (https://arxiv.org/abs/2507.18071).

    drop_zero_weight defaults to True to skip zero-advantage samples.
    """
    credit_assigner: CreditAssigner = GroupNormalizedReturn(
        group_size=group_size,
        normalize_adv=group_normalize_adv,
        positive_only=positive_only,
    )
    loss: Loss = ClippedSurrogateLoss(
        clip_eps_low=clip_eps_low,
        clip_eps_high=clip_eps_high,
        length_normalize=length_normalize,
        ratio_clip=ratio_clip,
    )
    preprocess_fns = []
    if drop_zero_weight:
        preprocess_fns.append(lambda batch: drop_zero_weight_samples(batch, eps=drop_zero_weight_eps))
    preprocess_fns.append(validate_actor_logps)
    preprocess = compose_preprocess(*preprocess_fns)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=preprocess,
    )


def make_cispo(
    *,
    group_size: int,
    group_normalize_adv: bool = True,
    positive_only: bool = False,
    clip_eps_low: float = 1e6,
    clip_eps_high: float = 0.2,
    length_normalize: bool = True,
    drop_zero_weight: bool = False,
    drop_zero_weight_eps: float = 1e-4,
    name: str = "cispo",
) -> RLAlgorithm:
    """
    CISPO (Clipped IS-weight Policy Optimization) preset.

    Unlike PPO/GRPO which clip token updates via the min operation, CISPO
    clips the importance sampling weights themselves and uses them as
    stop-gradient multipliers on the REINFORCE objective. This preserves
    gradient contributions from all tokens, especially low-probability
    tokens crucial for reflective reasoning behaviors.

    Key insight from MiniMax-M1: Tokens like "However", "Recheck", "Wait"
    are rare but crucial for learning reflective CoT behaviors. PPO/GRPO
    clip these out after the first update, preventing them from contributing
    to subsequent off-policy gradient updates. CISPO keeps all tokens.

    Loss:
        L = - E[ sg(clip(r_t, 1-ε_low, 1+ε_high)) * A * log π(a_t|s_t) ]

    Args:
        group_size: Number of rollouts per group for advantage normalization.
        group_normalize_adv: Whether to normalize advantages within each group.
        positive_only: If True, clip negative advantages to zero.
        clip_eps_low: Lower bound for IS weight clipping. Paper uses large
            value (effectively no lower bound).
        clip_eps_high: Upper bound for IS weight clipping.
        length_normalize: Whether to normalize by number of action tokens.
        drop_zero_weight: Whether to drop zero-advantage samples.
        drop_zero_weight_eps: Epsilon for zero-weight detection.
        name: Algorithm name for logging.

    Note: Rollouts must carry `group_id` in their metadata and each group
    must have exactly `group_size` members. Use GRPORequestStrategy for
    request expansion.

    Reference: MiniMax-M1 paper (arXiv:2506.13585)
    """
    credit_assigner: CreditAssigner = GroupNormalizedReturn(
        group_size=group_size,
        normalize_adv=group_normalize_adv,
        positive_only=positive_only,
    )
    loss: Loss = CISPOLoss(
        clip_eps_low=clip_eps_low,
        clip_eps_high=clip_eps_high,
        length_normalize=length_normalize,
    )
    preprocess_fns = []
    if drop_zero_weight:
        preprocess_fns.append(lambda batch: drop_zero_weight_samples(batch, eps=drop_zero_weight_eps))
    preprocess_fns.append(validate_actor_logps)
    preprocess = compose_preprocess(*preprocess_fns)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=preprocess,
    )


def make_sapo(
    *,
    group_size: int,
    group_normalize_adv: bool = True,
    positive_only: bool = False,
    tau_pos: float = 1.0,
    tau_neg: float = 1.05,
    length_normalize: bool = False,
    drop_zero_weight: bool = False,
    drop_zero_weight_eps: float = 1e-4,
    name: str = "sapo",
) -> RLAlgorithm:
    """
    SAPO (Soft Adaptive Policy Optimization) preset.

    SAPO replaces hard clipping with a smooth, temperature-controlled sigmoid gate
    that adaptively attenuates off-policy updates while preserving learning signals.
    The soft gate implements a continuous trust region that is both sequence-coherent
    and token-adaptive.

    Core mechanism:
        Instead of hard clipping: min(r * A, clip(r, 1-ε, 1+ε) * A)
        SAPO uses soft gate:      f(r) * A, where f(r) = (4/τ) * σ(τ(r - 1))

    The sigmoid gate σ(τ(r - 1)) peaks at r=1 (on-policy) and decays smoothly as
    r deviates, providing gradual attenuation rather than abrupt cutoff.

    Asymmetric temperatures:
        - τ_pos: temperature for positive advantages (increase token logit)
        - τ_neg: temperature for negative advantages (decrease token logit)

    Setting τ_neg > τ_pos makes negative gradients decay faster, improving stability.
    This is motivated by the observation that negative updates diffuse to many
    unsampled tokens in the vocabulary, introducing more noise than positive updates.

    Advantages over hard clipping methods:
        - vs GRPO: smooth token-level scaling instead of hard cutoff
        - vs GSPO: token-adaptive (preserves signal from near-on-policy tokens even
          when sequence has outliers)
        - Maintains sequence-level coherence under mild conditions (small steps,
          low token variance)

    Args:
        group_size: Number of rollouts per group for advantage normalization.
        group_normalize_adv: Whether to normalize advantages within each group.
        positive_only: If True, clip negative advantages to zero.
        tau_pos: Temperature for positive advantages. Default: 1.0 (paper setting).
        tau_neg: Temperature for negative advantages. Default: 1.05 (paper setting).
            Higher values → faster decay → more conservative.
        length_normalize: Whether to normalize by sequence length.
        drop_zero_weight: Whether to drop zero-advantage samples.
        drop_zero_weight_eps: Epsilon for zero-weight detection.
        name: Algorithm name for logging.

    Note: Rollouts must carry `group_id` in their metadata and each group
    must have exactly `group_size` members. Use GRPORequestStrategy for
    request expansion.

    Reference: "Soft Adaptive Policy Optimization" (arXiv:2511.20347v2)
    https://arxiv.org/abs/2511.20347

    Usage example:
        ```python
        from ludic.training import make_sapo, GRPORequestStrategy

        # Create SAPO algorithm
        algo = make_sapo(group_size=4)

        # Use with GRPO request expansion
        request_strategy = GRPORequestStrategy(group_size=4)
        ```
    """
    credit_assigner: CreditAssigner = GroupNormalizedReturn(
        group_size=group_size,
        normalize_adv=group_normalize_adv,
        positive_only=positive_only,
    )
    loss: Loss = SAPOLoss(
        tau_pos=tau_pos,
        tau_neg=tau_neg,
        length_normalize=length_normalize,
    )
    preprocess_fns = []
    if drop_zero_weight:
        preprocess_fns.append(lambda batch: drop_zero_weight_samples(batch, eps=drop_zero_weight_eps))
    preprocess_fns.append(validate_actor_logps)
    preprocess = compose_preprocess(*preprocess_fns)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=preprocess,
    )


def make_gmpo(
    *,
    group_size: int,
    group_normalize_adv: bool = True,
    positive_only: bool = False,
    clip_eps_low: float = 0.4,
    clip_eps_high: float = 0.4,
    length_normalize: bool = True,
    ratio_clip: Optional[float] = None,
    drop_zero_weight: bool = False,
    drop_zero_weight_eps: float = 1e-4,
    name: str = "gmpo",
) -> RLAlgorithm:
    """
    GMPO (Geometric-Mean Policy Optimization) preset.

    GMPO stabilizes GRPO by using the geometric mean of token-level importance
    ratios instead of the arithmetic mean. This makes the objective less sensitive
    to outliers and results in more stable policy updates with fewer extreme
    importance sampling ratios.

    Key advantages over GRPO:
        1. More robust to outlier tokens (geometric mean vs arithmetic mean)
        2. More stable importance sampling ratios during training
        3. Supports wider clipping ranges (e.g., (e^-0.4, e^0.4) vs (0.8, 1.2))
        4. Better exploration due to higher entropy maintenance
        5. More stable gradients and lower KL divergence from reference policy

    Objective:
        J_GMPO = E[ (∏_t min(ρ_t * A, clip(ρ_t, e^-ε_low, e^ε_high) * A))^(1/|o|) * sgn(A) ]

    where:
        - ρ_t = π_new(a_t|s_t) / π_old(a_t|s_t) is the token-level importance ratio
        - A is the advantage (group-normalized)
        - |o| is the sequence length
        - Clipping is performed at the token level in log-space

    Implementation differences from GRPO:
        - Uses geometric mean: (∏_t ρ_t)^(1/|o|) instead of (1/|o|) Σ_t ρ_t
        - All operations performed in log-space for numerical stability
        - Token-level clipping (not sequence-level as in DeepSeek-R1)
        - Wider default clipping range: (e^-0.4, e^0.4) ≈ (0.67, 1.49)

    Args:
        group_size: Number of rollouts per group for advantage normalization.
        group_normalize_adv: Whether to normalize advantages within each group.
            Recommended: True (follows GRPO and paper experiments).
        positive_only: If True, clip negative advantages to zero.
        clip_eps_low: Lower clipping epsilon in log-space. Default 0.4 means
            clipping to e^-0.4 ≈ 0.67. Paper uses (e^-0.4, e^0.4).
        clip_eps_high: Upper clipping epsilon in log-space. Default 0.4 means
            clipping to e^0.4 ≈ 1.49.
        length_normalize: Whether to normalize by sequence length (1/|o|).
            This is critical for GMPO stability. Default: True.
        ratio_clip: Optional upper bound for geometric mean ratio truncation.
        drop_zero_weight: Whether to drop zero-advantage samples before training.
        drop_zero_weight_eps: Epsilon for zero-weight detection.
        name: Algorithm name for logging/metrics.

    Note: Rollouts must carry `group_id` in their metadata and each group
    must have exactly `group_size` members. Use GRPORequestStrategy for
    request expansion.

    Reference: "GMPO: Geometric-Mean Policy Optimization" (arXiv:2507.20673v3)
    https://arxiv.org/abs/2507.20673

    Usage example:
        ```python
        from ludic.training import make_gmpo, GRPORequestStrategy

        # Create GMPO algorithm
        algo = make_gmpo(group_size=4)

        # Use with GRPO request expansion
        request_strategy = GRPORequestStrategy(group_size=4)
        ```
    """
    credit_assigner: CreditAssigner = GroupNormalizedReturn(
        group_size=group_size,
        normalize_adv=group_normalize_adv,
        positive_only=positive_only,
    )
    loss: Loss = GMPOLoss(
        clip_eps_low=clip_eps_low,
        clip_eps_high=clip_eps_high,
        length_normalize=length_normalize,
        ratio_clip=ratio_clip,
    )
    preprocess_fns = []
    if drop_zero_weight:
        preprocess_fns.append(lambda batch: drop_zero_weight_samples(batch, eps=drop_zero_weight_eps))
    preprocess_fns.append(validate_actor_logps)
    preprocess = compose_preprocess(*preprocess_fns)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=preprocess,
    )


# ---------------------------------------------------------------------------
# SFT (Supervised Fine-Tuning / Behavioral Cloning)
# ---------------------------------------------------------------------------


def make_sft(
    *,
    length_normalize: bool = False,
    name: str = "sft",
) -> RLAlgorithm:
    """
    Supervised Fine-Tuning (SFT) / Behavioral Cloning.

    This is offline RL with trivial credit assignment:
      - Credit assignment: constant weight=1.0 for all steps
      - Loss: ReinforceLoss (which with uniform weights is just NLL)

    SFT treats all actions in the dataset equally, making it suitable for:
      - Cold-start training on rejection-sampled successful trajectories
      - Behavioral cloning from expert demonstrations
      - Pre-training before RL fine-tuning

    Args:
        length_normalize: If True, normalize log-probs by action length.
            This prevents the loss from being dominated by long sequences.
        name: Algorithm name for logging/metrics.

    Usage with OfflineBatchSource:
        ```python
        from ludic.training import OfflineBatchSource, Trainer, make_sft

        algo = make_sft()
        batch_source = OfflineBatchSource(
            jsonl_paths=[Path("data/winners.jsonl")],
            tokenize=tokenizer.encode,
            credit_assigner=algo.credit_assigner,
            batch_size=32,
        )
        trainer = Trainer(model=model, algorithm=algo, ...)
        ```
    """
    credit_assigner: CreditAssigner = ConstantCredit(value=1.0)
    # Use standard token-level CE over the action region for stability.
    loss: Loss = MaskedCausalLMCrossEntropyLoss(length_normalize=length_normalize)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
    )


# ---------------------------------------------------------------------------
# ScaleRL (CISPO + Hybrid Normalization)
# ---------------------------------------------------------------------------


def make_scalerl(
    *,
    group_size: int,
    positive_only: bool = False,
    clip_eps_low: float = 0.20,
    clip_eps_high: float = 0.28,
    length_normalize: bool = True,
    kl_coeff: float = 0.0,
    drop_zero_weight_eps: float = 1e-4,
    name: str = "scalerl",
) -> RLAlgorithm:
    """
    ScaleRL recipe: CISPO loss + hybrid advantage normalization + zero-weight filtering.

    This combines the key sample-efficiency improvements from the ScaleRL paper:

    1. **HybridNormalizedReturn**: Group-mean centering + batch-std scaling.
       More robust than pure group-level normalization because it avoids
       std=0 explosions in low-variance groups (easy prompts).

    2. **CISPOLoss**: Truncated IS-weight policy gradient that preserves
       gradient contributions from rare tokens (crucial for reflective
       reasoning behaviors like "Wait", "However", "Recheck").

    3. **Drop zero-weight samples**: After credit assignment, drop samples with
       near-zero weight to reduce no-op updates.

    4. **FP32 logits** (via TrainerConfig.cast_logits_to_fp32):
       Recommended for IS ratio stability. Not controlled by this preset—
       set in TrainerConfig.

    Args:
        group_size: Number of rollouts per group (required for credit assignment).
        positive_only: If True, clip negative advantages to zero (REINFORCE-only).
        clip_eps_low: Lower CISPO clipping bound. Default 0.20 per context-notes.md.
        clip_eps_high: Upper CISPO clipping bound. Default 0.28 per context-notes.md.
        length_normalize: Whether to normalize by number of action tokens.
        kl_coeff: Coefficient for optional token-level KL penalty.
            Set > 0 for additional stability. Typical: 0.01-0.1. Default 0.0.
        drop_zero_weight_eps: Epsilon for zero-weight sample detection.
        name: Algorithm name for logging/metrics.

    Note: Rollouts must carry `group_id` in their metadata and each group
    must have exactly `group_size` members. Use GRPORequestStrategy for
    request expansion.

    References:
        - ScaleRL: arXiv:2510.13786
        - DAPO (zero-weight filtering): arXiv:2503.14476
        - MiniMax-M1 (CISPO): arXiv:2506.13585
    """
    # HybridNormalizedReturn: group-mean baseline + batch-std scaling
    credit_assigner: CreditAssigner = HybridNormalizedReturn(
        group_size=group_size,
        positive_only=positive_only,
    )

    # CISPO loss with asymmetric clipping
    cispo_loss: Loss = CISPOLoss(
        clip_eps_low=clip_eps_low,
        clip_eps_high=clip_eps_high,
        length_normalize=length_normalize,
    )

    # Optionally add token-level KL penalty for stability
    if kl_coeff > 0:
        kl_loss = TokenKLLoss(coeff=kl_coeff, length_normalize=length_normalize)
        loss: Loss = CompositeLoss(
            terms=[
                LossTerm(name="cispo", loss=cispo_loss, weight=1.0),
                LossTerm(name="kl", loss=kl_loss, weight=1.0),
            ]
        )
    else:
        loss = cispo_loss

    # Build preprocessing pipeline (order matters)
    preprocess_fns = []

    # 1. Drop individual zero-weight samples (after credit assignment)
    preprocess_fns.append(
        lambda batch: drop_zero_weight_samples(batch, eps=drop_zero_weight_eps)
    )

    # 2. Validate actor logprobs (required for CISPO ratio computation)
    preprocess_fns.append(validate_actor_logps)

    preprocess = compose_preprocess(*preprocess_fns)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
        preprocess=preprocess,
    )
