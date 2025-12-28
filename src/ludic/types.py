from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Union, Optional, Literal, Mapping
import logging
import time
import uuid
import json

log = logging.getLogger(__name__)

JSON = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
StepKind = Literal["agent", "env"]


# Chat-style message schema
Message = Dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


@dataclass(frozen=True)
class TokenTrace:
    """
    Canonical token-in/token-out trace for a single model call.

    This is the single source of truth for prompt/completion token IDs
    and (optional) chosen-token logprobs. It is kept separate from Step.info
    to avoid duplicating large token arrays in JSON metadata.
    """
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    completion_logprobs: Optional[List[float]] = None
    finish_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_token_ids, list) or not all(
            isinstance(v, int) for v in self.prompt_token_ids
        ):
            raise TypeError("TokenTrace.prompt_token_ids must be a List[int].")
        if not isinstance(self.completion_token_ids, list) or not all(
            isinstance(v, int) for v in self.completion_token_ids
        ):
            raise TypeError("TokenTrace.completion_token_ids must be a List[int].")
        if self.completion_logprobs is not None:
            if not isinstance(self.completion_logprobs, list) or not all(
                isinstance(v, (int, float)) for v in self.completion_logprobs
            ):
                raise TypeError("TokenTrace.completion_logprobs must be a List[float].")
            if len(self.completion_logprobs) != len(self.completion_token_ids):
                raise ValueError(
                    "TokenTrace completion_logprobs length mismatch "
                    f"({len(self.completion_logprobs)} vs {len(self.completion_token_ids)})."
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_token_ids": list(self.prompt_token_ids),
            "completion_token_ids": list(self.completion_token_ids),
            "completion_logprobs": (
                list(self.completion_logprobs) if self.completion_logprobs is not None else None
            ),
            "finish_reason": self.finish_reason,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "TokenTrace":
        if "prompt_token_ids" not in data or "completion_token_ids" not in data:
            raise ValueError("TokenTrace missing prompt_token_ids or completion_token_ids.")
        return TokenTrace(
            prompt_token_ids=list(data["prompt_token_ids"]),
            completion_token_ids=list(data["completion_token_ids"]),
            completion_logprobs=data.get("completion_logprobs"),
            finish_reason=data.get("finish_reason"),
        )


@dataclass
class ChatResponse:
    """
    Normalized inference output for training/logging.
    Keep this minimal. Put transport/vendor junk in the returned `info` dict.

    Token IDs are required for token-in/token-out API consistency.
    """
    text: str
    prompt_token_ids: List[int]
    completion_token_ids: List[int]
    completion_logprobs: Optional[List[float]] = None
    finish_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            self.completion_logprobs is not None
            and len(self.completion_token_ids) != len(self.completion_logprobs)
        ):
            log.warning(
                "ChatResponse completion_token_ids/completion_logprobs length mismatch "
                "(%d vs %d).",
                len(self.completion_token_ids),
                len(self.completion_logprobs),
            )

    def to_info(self) -> Dict[str, Any]:
        """
        Canonical serialization of "training-relevant" fields into the shared `info`
        dict shape used throughout the project.
        """
        info: Dict[str, Any] = {
            "completion": self.text,
        }
        if self.finish_reason is not None:
            info["finish_reason"] = self.finish_reason
        return info

    def merge_into_info(self, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Merge this response's canonical fields into an existing `info` dict.
        """
        if info is None:
            info = {}
        info.update(self.to_info())
        return info

    def to_trace(self) -> TokenTrace:
        """
        Build a TokenTrace from this response.
        """
        return TokenTrace(
            prompt_token_ids=list(self.prompt_token_ids),
            completion_token_ids=list(self.completion_token_ids),
            completion_logprobs=(
                list(self.completion_logprobs) if self.completion_logprobs is not None else None
            ),
            finish_reason=self.finish_reason,
        )

# ----- Environment level types -----

Observation = str
Info   = Dict[str, JSON]

@dataclass
class StepOutcome:
    obs: str
    reward: float
    truncated: bool
    terminated: bool
    info: Info = field(default_factory=dict)
    trace: Optional[TokenTrace] = None

@dataclass(frozen=True)
class Snapshot:
    """
    A portable checkpoint of an Environment, not just the outward State.
    `env_kind` and `version` let you refuse incompatible restores.
    """
    env_kind: str
    version: str           # bump on breaking schema changes
    episode_id: str        # source episode lineage
    obs: str           # outward state snapshot
    world: Dict[str, JSON] # extra hidden/internal stuff needed to resume
    created_ns: int = field(default_factory=lambda: time.time_ns())

    def to_json(self) -> str:
        return json.dumps(asdict(self))

# ----- Environment-Agent-Interaction level types -----

@dataclass
class AgentStep:
    index: int
    prompt_messages: List[Message]
    action: str
    action_target: str  # "internal" | "env"
    loop_index: int
    reward: float
    truncated: bool
    terminated: bool
    trace: TokenTrace
    info: Info = field(default_factory=dict)
    reward_components: Dict[str, float] = field(default_factory=dict)
    kind: StepKind = "agent"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts_ns: int = field(default_factory=lambda: time.time_ns())
    turn_id: Optional[str] = None
    parent_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None

@dataclass
class EnvironmentStep:
    index: int
    prev_obs: Observation
    action: str
    parsed_action: Any
    next_obs: Optional[Observation]
    source_agent_step_id: str
    agent_step_ids: List[str]
    reward: float
    truncated: bool
    terminated: bool
    trace: TokenTrace
    info: Info = field(default_factory=dict)
    reward_components: Dict[str, float] = field(default_factory=dict)
    kind: StepKind = "env"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts_ns: int = field(default_factory=lambda: time.time_ns())
    turn_id: Optional[str] = None
    parent_id: Optional[str] = None

Step = Union[AgentStep, EnvironmentStep]

@dataclass
class Rollout:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: List[Step] = field(default_factory=list)
    meta: Dict[str, JSON] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        """
        Total reward for training/eval:
        sums env-step rewards and includes parse-error penalties from
        env-targeted agent steps.
        """
        total = sum(s.reward for s in self.steps if s.kind == "env")
        for step in self.steps:
            if not isinstance(step, AgentStep):
                continue
            if step.action_target != "env":
                continue
            if not step.info.get("parse_error", False):
                continue
            total += float(step.reward)
        return total

    def total_reward_all(self) -> float:
        return sum(s.reward for s in self.steps)

    def total_reward_for(self, kinds: Optional[set[StepKind]] = None) -> float:
        if kinds is None:
            return self.total_reward_all()
        return sum(s.reward for s in self.steps if s.kind in kinds)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def start_ns(self) -> Optional[int]:
        return self.steps[0].ts_ns if self.steps else None

    @property
    def end_ns(self) -> Optional[int]:
        return self.steps[-1].ts_ns if self.steps else None

    @property
    def duration_ns(self) -> Optional[int]:
        start = self.start_ns
        end = self.end_ns
        if start is None or end is None:
            return None
        return end - start

    @property
    def duration_s(self) -> Optional[float]:
        ns = self.duration_ns
        return None if ns is None else ns / 1e9
