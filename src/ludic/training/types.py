from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple, TypeVar, cast

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


class SampleExtra:
    """
    Marker base class for typed extras that algorithms may need.
    """


@dataclass(frozen=True)
class ActorTokenLogps(SampleExtra):
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


@dataclass(frozen=True)
class TeacherTokenLogps(SampleExtra):
    """
    Per-token logprobs under the teacher policy, aligned to the sampled completion tokens.

    `token_logps[i]` corresponds to the chosen-token logprob for
    `completion_token_ids[i]`.
    """

    token_logps: List[float]

    def __post_init__(self) -> None:
        if not isinstance(self.token_logps, list) or not all(
            isinstance(v, (int, float)) for v in self.token_logps
        ):
            raise TypeError("TeacherTokenLogps.token_logps must be a List[float].")


TExtra = TypeVar("TExtra", bound=SampleExtra)


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
    - extras: typed extras that algorithms may need (e.g. actor logps for PPO/GRPO
      ratios, teacher logps for OPD).
    """
    input_ids: List[int]
    attention_mask: List[int]
    action_mask: List[int]
    weight: float
    meta: Dict[str, JSON]
    extras: List[SampleExtra] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: set[type[SampleExtra]] = set()
        for extra in self.extras:
            if not isinstance(extra, SampleExtra):
                raise TypeError("SAWItem.extras must contain SampleExtra instances.")
            extra_type = type(extra)
            if extra_type in seen:
                raise ValueError(f"Duplicate extra type: {extra_type.__name__}")
            seen.add(extra_type)

    def get_extra(self, extra_type: type[TExtra]) -> Optional[TExtra]:
        for extra in self.extras:
            if isinstance(extra, extra_type):
                return cast(TExtra, extra)
        return None

    def add_extra(self, extra: SampleExtra) -> None:
        if self.get_extra(type(extra)) is not None:
            raise ValueError(f"Duplicate extra type: {type(extra).__name__}")
        self.extras.append(extra)

    @property
    def actor_logps(self) -> Optional[ActorTokenLogps]:
        return self.get_extra(ActorTokenLogps)

    @property
    def teacher_logps(self) -> Optional[TeacherTokenLogps]:
        return self.get_extra(TeacherTokenLogps)

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
    """

    async def next_batch(self) -> SAWBatch:
        ...

# ---------------------------------------------------------------------------
# Helper aliases
# ---------------------------------------------------------------------------

TokenizeFn = Callable[[str], List[int]]
StateFromStepFn = Callable[[Rollout, int, Step], str]

# Filter function: returns True to KEEP a sample, False to DROP it
SampleFilter = Callable[["SAWItem"], bool]
