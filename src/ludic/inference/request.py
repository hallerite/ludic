from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

from ludic.types import Message
from ludic.inference.sampling import SamplingParams
from ludic.inference.extensions.base import BackendExtensions


@dataclass(frozen=True)
class ReturnSpec:
    """
    What additional *return payload* should the backend provide?

    This is intentionally not a grab-bag of vendor features. It only covers
    training-relevant artifacts that get attached to Step.info (token IDs,
    chosen-token logprobs, etc.).
    """

    return_token_ids: bool = False
    return_chosen_logprobs: bool = False
    top_logprobs_k: int = 1

    def __post_init__(self) -> None:
        if self.top_logprobs_k <= 0:
            raise ValueError(f"top_logprobs_k must be positive, got {self.top_logprobs_k}")

    @staticmethod
    def for_eval(*, return_token_ids: bool = True) -> "ReturnSpec":
        return ReturnSpec(
            return_token_ids=return_token_ids,
            return_chosen_logprobs=False,
            top_logprobs_k=1,
        )

    @staticmethod
    def for_rl(*, top_logprobs_k: int = 1) -> "ReturnSpec":
        return ReturnSpec(
            return_token_ids=True,
            return_chosen_logprobs=True,
            top_logprobs_k=top_logprobs_k,
        )


@dataclass(frozen=True)
class ToolRequest:
    tools: List[Dict[str, Any]]
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"


@dataclass(frozen=True)
class ChatCompletionRequest:
    model: str
    messages: List[Message]
    sampling: SamplingParams = field(default_factory=SamplingParams)
    return_: ReturnSpec = field(default_factory=ReturnSpec)
    seed: Optional[int] = None
    tools: Optional[ToolRequest] = None
    extensions: Optional[BackendExtensions] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceSpec:
    """
    Per-step inference configuration (minus the prompt/messages).

    Protocols pass this through to agents; agents construct a
    ChatCompletionRequest using their configured model and current messages.
    """

    sampling: SamplingParams = field(default_factory=SamplingParams)
    return_: ReturnSpec = field(default_factory=ReturnSpec)
    extensions: Optional[BackendExtensions] = None
