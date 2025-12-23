from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple, TypeGuard

from ludic.types import JSON, Rollout, Step
from ludic.inference.request import InferenceSpec


@dataclass
class EnvSpec:
    """
    Serializable description of an environment to instantiate.

    - kind: string key into an env registry
    - kwargs: JSON-serializable constructor/config kwargs
    """
    kind: str
    kwargs: Dict[str, JSON] = field(default_factory=dict)


@dataclass
class CtxSpec:
    """
    Serializable description of a context strategy.

    - kind: string key into a ctx registry
    - kwargs: JSON-serializable constructor/config kwargs
    """
    kind: str
    kwargs: Dict[str, JSON] = field(default_factory=dict)


@dataclass
class ProtocolSpec:
    """
    Serializable description of a protocol to instantiate.

    - kind: string key into a protocol registry
    - kwargs: JSON-serializable constructor/config kwargs
    """
    kind: str
    kwargs: Dict[str, JSON] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rollout-level configuration / identification
# ---------------------------------------------------------------------------

# (rollout_id, step_index)
RolloutStepKey = Tuple[str, int]

@dataclass
class RolloutRequest:
    """
    Template for one or more rollouts.

    This is *pure data*; RolloutEngine will:

        - resolve env via registry from (env.kind)
        - resolve protocol via registry from (protocol.kind)
        - call the factories with env.kwargs / protocol.kwargs
        - run `num_episodes` independent episodes using the
          instantiated InteractionProtocol.

    Fields:
      - env:
            EnvSpec, resolved via env_registry.
      - protocol:
            ProtocolSpec, resolved via protocol_registry.
            
      - inference:
            Passed directly to Agent via protocol.run().

      - num_episodes:
            How many episodes to run with this configuration.

      - meta:
            Arbitrary JSON metadata that gets merged into Rollout.meta["request_meta"].
    """
    env: EnvSpec
    protocol: ProtocolSpec
    env_seed: Optional[int] = None
    sampling_seed: Optional[int] = None
    inference: Optional[InferenceSpec] = None
    num_episodes: int = 1
    meta: Dict[str, JSON] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Credit assignment
# ---------------------------------------------------------------------------


class CreditAssigner(Protocol):
    """
    Computes a scalar weight for each (rollout, step) in a batch.
    """

    def compute(
        self,
        rollouts: List[Rollout],
    ) -> Dict[RolloutStepKey, float]:
        ...


# ---------------------------------------------------------------------------
# State–Action–Weight representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActorTokenLogps:
    """
    Per-token logprobs under the behavior policy (the actor), aligned to the
    sampled completion tokens.

    `token_logps[i]` corresponds to the chosen-token logprob for
    `completion_token_ids[i]`.
    """

    token_logps: List[float]

    def __post_init__(self) -> None:
        if not isinstance(self.token_logps, list) or not all(
            isinstance(v, (int, float)) for v in self.token_logps
        ):
            raise TypeError("ActorTokenLogps.token_logps must be a List[float].")


@dataclass
class SampleAttachments:
    """
    Optional typed attachments carried alongside a training sample.

    These should be populated by construction by BatchSources / annotators /
    rollout collation, not discovered ad-hoc from `meta`.
    """

    actor_logps: Optional[ActorTokenLogps] = None


class HasActorLogps(Protocol):
    """
    Structural “extension”: a sample that is guaranteed to carry per-token actor logps.

    This is meant for typing algorithms/losses that require token-level behavior
    logprobs, and composes naturally with other Protocol extensions.
    """

    actor_logps: ActorTokenLogps


def has_actor_logps(item: "SAWItem") -> TypeGuard[HasActorLogps]:
    """
    Type guard for `HasActorLogps`.

    Use this to narrow a SAWItem to something that is guaranteed (by runtime check)
    to have non-None `actor_logps`.
    """

    return item.actor_logps is not None


@dataclass
class SAWItem:
    """
    State–Action–Weight sample with masks.

    - input_ids: tokenized [state || action]
    - attention_mask: 1/0 attention mask to tell tokens from padding
    - action_mask: 1 on action tokens, 0 on state tokens
    - weight: scalar credit for this sample
    - meta: arbitrary rollout/step metadata (JSON-serializable; for logging,
      debugging, filtering, etc.)
    - attachments: typed, non-JSON training attachments (e.g. actor logps needed
      for PPO/GRPO ratios)
    """
    input_ids: List[int]
    attention_mask: List[int]
    action_mask: List[int]
    weight: float
    meta: Dict[str, JSON]
    attachments: SampleAttachments = field(default_factory=SampleAttachments)

    @property
    def actor_logps(self) -> Optional[ActorTokenLogps]:
        return self.attachments.actor_logps

@dataclass
class SAWBatch:
    """
    Logical batch of State–Action–Weight samples.

    - items: the SAWItem samples
    - meta: batch-level metadata (reward stats, timing, env info, etc.)
    """
    items: list[SAWItem]
    meta: dict[str, JSON] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Batch source abstraction
# ---------------------------------------------------------------------------


class BatchSource(Protocol):
    """
    Abstract source of SAWBatch samples.

    Trainer only depends on this interface and does not care where the
    data comes from (online rollouts, replay buffer, branching search, etc.).
    The returned SAWBatch is treated as a macro-batch and split into
    micro-batches for gradient accumulation.
    """

    async def next_batch(self) -> SAWBatch:
        ...

# ---------------------------------------------------------------------------
# Helper aliases
# ---------------------------------------------------------------------------

TokenizeFn = Callable[[str], List[int]]
StateFromStepFn = Callable[[Rollout, int, Step], str]
StepSelector = Callable[[Step], bool]

# Filter function: returns True to KEEP a sample, False to DROP it
SampleFilter = Callable[["SAWItem"], bool]
