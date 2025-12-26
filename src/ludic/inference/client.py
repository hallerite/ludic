from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

import torch  # type: ignore

from ludic.types import ChatResponse
from ludic.inference.request import TokenCompletionRequest

class ChatClient(Protocol):
    """
    Backend contract.
      - accepts a typed request object
      - maps request -> backend kwargs
      - executes the call and returns (ChatResponse, info)
      - can atomically push a set of parameter tensors to the runtime
    """

    async def complete_tokens(
        self,
        request: TokenCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        """
        Complete from pre-tokenized prompt.

        This uses the completions endpoint (not chat completions) with
        caller-supplied token IDs for drift-free RL training.

        Implementations may raise NotImplementedError if they do not support
        token-in prompts.
        """
        ...

    def sync_weights(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        version: Optional[str | int] = None,
    ) -> str:
        """
        Atomically apply a set of parameter updates.
        Returns the committed version string.
        Should raise specific exceptions on timeout/reject/broadcast failure.
        """
        ...

class VersionedClient(ChatClient, Protocol):
    """
    A ChatClient that is also 'PipelineRL Ready'.
    It can report the version of the policy currently serving requests.
    """
    async def get_policy_version(self) -> int:
        """
        Returns the current monotonic version of the policy weights.
        """
        ...
