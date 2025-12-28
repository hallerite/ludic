from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    return_token_ids: bool = True
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
    """
    Tool schemas for chat template embedding.

    In token-in mode, tools are embedded into the prompt via the chat template
    (e.g., HuggingFace's apply_chat_template(tools=...)), not sent to the
    server. The model outputs tool calls as text, which are then parsed by
    a ToolParser.

    Note: Unlike the old chat completions API, there is no `tool_choice`
    parameter. Tool calling behavior is entirely prompt-driven.
    """

    tools: List[Dict[str, Any]]


@dataclass(frozen=True)
class TokenCompletionRequest:
    """
    Pre-tokenized completion request.

    Used when the caller has already applied the chat template and wants
    to send raw token IDs to the model. This bypasses vLLM's internal
    template application, giving full control over tokenization for
    drift-free RL training.
    """

    model: str
    prompt_token_ids: List[int]
    prompt_text: Optional[str] = None  # For debugging/logging
    sampling: SamplingParams = field(default_factory=SamplingParams)
    return_: ReturnSpec = field(default_factory=ReturnSpec)
    seed: Optional[int] = None
    extensions: Optional[BackendExtensions] = None


@dataclass(frozen=True)
class InferenceSpec:
    """
    Per-step inference configuration (minus the prompt/messages).

    Protocols pass this through to agents; agents construct a
    TokenCompletionRequest by applying their chat template to the current messages.
    """

    sampling: SamplingParams = field(default_factory=SamplingParams)
    return_: ReturnSpec = field(default_factory=ReturnSpec)
    extensions: Optional[BackendExtensions] = None
