from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol

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
    MaskedCausalLMCrossEntropyLoss,
    ReverseKLLoss,
)
from ludic.training.credit_assignment import (
    MonteCarloReturn,
    GroupNormalizedReturn,
    ConstantCredit,
    CreditModifier,
    KLCreditModifier,
)


Batch = Mapping[str, Tensor]
Logits = Float[Tensor, "B T V"]

class PreprocessFn(Protocol):
    def __call__(self, saw_batch: SAWBatch) -> SAWBatch: ...


@dataclass
class RLAlgorithm:
    """
    Full RL algorithm = credit assignment + credit modifiers + loss.

    Pipeline:
        Rollouts → CreditAssigner → SAWBatch → Collator → Batch
                                                            ↓
                                                    CreditModifiers
                                                            ↓
                                                      Modified Batch
                                                            ↓
                                                          Loss

    Components:
        - credit_assigner: Rollouts → per-step scalar credits (advantages)
        - credit_modifiers: Batch → Modified Batch (add per-token signals to advantages)
        - loss: Batch + Logits → scalar loss

    CreditModifiers (Level 2: Advantage Modification):
        Modify advantages AFTER collation, BEFORE loss. Signals added here:
        - Go through importance sampling (multiplied by ratio)
        - Use rollout-time values (old policy)
        - Interact with task rewards through the same loss

        Example: KLCreditModifier adds teacher KL penalty to advantages.

    See docs/composition.md for the full composition level documentation.
    """

    name: str
    credit_assigner: CreditAssigner
    loss: Loss
    credit_modifiers: List[CreditModifier] = field(default_factory=list)
    preprocess: Optional[PreprocessFn] = None

    def compute_loss(
        self,
        model: nn.Module,
        batch: Batch,
    ) -> tuple[Tensor, Dict[str, Any]]:
        """
        Apply credit modifiers, run forward pass, compute loss.

        Returns:
            Tuple of (loss, stats_dict) where stats_dict includes:
            - Modifier metrics namespaced as "{modifier.name}/{metric}"
            - Loss metrics from self.loss.compute()
        """
        all_stats: Dict[str, Any] = {}

        # --- Apply credit modifiers (Level 2: Advantage Modification) ---
        for modifier in self.credit_modifiers:
            batch, modifier_stats = modifier.modify(batch)
            # Namespace modifier metrics
            for key, value in modifier_stats.items():
                all_stats[f"{modifier.name}/{key}"] = value

        # --- Run the forward pass ---
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits: Logits = outputs.logits

        # --- Compute loss ---
        loss, loss_stats = self.loss.compute(logits, batch)
        all_stats.update(loss_stats)

        return loss, all_stats


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
# On-Policy Distillation (OPD)
# ---------------------------------------------------------------------------


def make_opd(
    *,
    kl_coeff: float = 1.0,
    length_normalize: bool = True,
    name: str = "opd",
) -> RLAlgorithm:
    """
    On-Policy Distillation (OPD).

    Dense per-token supervision from a teacher model. The student samples
    trajectories, teacher provides logprobs, and training minimizes reverse KL.

    Core idea: Sample from student, grade each token with teacher logprobs.
    This combines on-policy learning (samples from student) with dense
    supervision (per-token teacher signal), achieving better compute
    efficiency than sparse RL rewards.

    Credit assignment: ConstantCredit(1.0) - the per-step "advantage" is
    implicit in the loss (negative reverse KL per token).

    Loss: ReverseKLLoss - minimizes KL(student || teacher) per token.

    Prerequisites:
        - Agent must have a TokenLevelScorer with name="teacher_logps"
        - Use make_vllm_teacher_scorer() to create the scorer
        - Scorer runs during Agent.act() and scores flow through to training

    Args:
        kl_coeff: Coefficient for KL loss. Higher values push the student
            harder towards the teacher's distribution. Default 1.0.
        length_normalize: If True (default), divide per-sample loss by number
            of action tokens. This keeps gradients stable across sequence lengths.
        name: Algorithm name for logging/metrics.

    Example:
        ```python
        from ludic.training import Trainer, make_opd
        from ludic.training.scoring import make_vllm_teacher_scorer
        from ludic.agent import Agent

        # Create teacher scorer
        teacher_scorer = make_vllm_teacher_scorer(
            base_url="http://localhost:8001",
            model="Qwen/Qwen3-32B",
        )

        # Attach to agent - scores flow through automatically
        agent = Agent(
            client=client,
            ...,
            scorers=[teacher_scorer],
        )

        # Train with OPD
        trainer = Trainer(model=model, algo=make_opd(), ...)
        ```

    Reference: https://thinkingmachines.ai/blog/on-policy-distillation

    Note:
        This uses Loss Composition (Level 3). For OPD where KL goes through
        importance sampling (Level 2: Advantage Modification), use make_gspo_opd() instead.
    """
    credit_assigner: CreditAssigner = ConstantCredit(value=1.0)
    loss: Loss = ReverseKLLoss(coeff=kl_coeff, length_normalize=length_normalize)

    return RLAlgorithm(
        name=name,
        credit_assigner=credit_assigner,
        loss=loss,
    )


def make_gspo_opd(
    *,
    group_size: int,
    kl_coeff: float = 1.0,
    kl_per_token: bool = True,
    group_normalize_adv: bool = True,
    positive_only: bool = False,
    clip_eps_low: float = 3e-4,
    clip_eps_high: float = 4e-4,
    length_normalize: bool = True,
    ratio_clip: Optional[float] = None,
    drop_zero_weight: bool = False,
    drop_zero_weight_eps: float = 1e-4,
    name: str = "gspo_opd",
) -> RLAlgorithm:
    """
    GSPO + OPD hybrid using advantage composition.

    Combines:
    - GSPO: Task rewards with group-normalized advantages
    - OPD: Teacher KL penalty added to advantages (Level 2: Advantage Modification)

    KL is computed at rollout time and added to advantages BEFORE the loss.
    This means:
    - KL goes through importance sampling (multiplied by ratio)
    - KL uses old policy logprobs (from rollout), not current policy
    - All signals interact through the same loss function

    Pipeline:
        1. GroupNormalizedReturn computes task-based advantages
        2. KLCreditModifier adds negative KL to advantages (per-token if kl_per_token=True)
        3. ClippedSurrogateLoss computes policy gradient with modified advantages

    Args:
        group_size: Number of rollouts per group for advantage normalization.
        kl_coeff: Coefficient for KL penalty. Higher = stronger teacher matching.
        kl_per_token: If True, apply KL penalty per action token by broadcasting
            scalar advantages to token-level weights.
        group_normalize_adv: Normalize advantages within each group.
        positive_only: Clip negative advantages to zero.
        clip_eps_low: Lower PPO clipping epsilon.
        clip_eps_high: Upper PPO clipping epsilon.
        length_normalize: Normalize loss by number of action tokens.
        ratio_clip: Optional upper bound for ratio truncation.
        drop_zero_weight: Drop zero-advantage samples before collation.
        drop_zero_weight_eps: Epsilon for zero-weight detection.
        name: Algorithm name for logging.

    Prerequisites:
        - Rollouts must return actor logprobs (ReturnSpec.for_rl())
        - Agent must have a teacher scorer (make_vllm_teacher_scorer())
        - Rollouts must have group_id for GSPO advantage normalization

    Example:
        ```python
        from ludic.training import make_gspo_opd
        from ludic.training.scoring import make_vllm_teacher_scorer

        teacher_scorer = make_vllm_teacher_scorer(
            base_url="http://localhost:8001",
            model="Qwen/Qwen3-32B",
        )

        agent = Agent(client=client, ..., scorers=[teacher_scorer])
        algo = make_gspo_opd(group_size=8, kl_coeff=1.0)
        trainer = Trainer(model=model, algo=algo, ...)
        ```

    Reference: https://thinkingmachines.ai/blog/on-policy-distillation
    """
    credit_assigner: CreditAssigner = GroupNormalizedReturn(
        group_size=group_size,
        normalize_adv=group_normalize_adv,
        positive_only=positive_only,
    )

    credit_modifiers = [KLCreditModifier(coeff=kl_coeff, broadcast_advantage=kl_per_token)]

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
        credit_modifiers=credit_modifiers,
        loss=loss,
        preprocess=preprocess,
    )
