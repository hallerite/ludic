"""
Intrinsic scoring protocols for agents.

Intrinsic scores are agent-local evaluations of action quality, computed
during rollout generation. They are analogous to the parser (which validates
action syntax) but evaluate action quality instead.

Two scorer types:
- TokenLevelScorer: Per-token scores (e.g., teacher logprobs for OPD)
- ActionLevelScorer: Scalar per-action scores (e.g., LLM-as-judge)

Scores are attached to AgentActStep and flow into SAWItem for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Union, runtime_checkable


@runtime_checkable
class TokenLevelScorer(Protocol):
    """
    Per-token intrinsic scoring.

    Use cases:
    - Teacher logprobs for On-Policy Distillation (OPD)
    - Token-level reward models
    - Per-token confidence scores

    The scorer receives completion token IDs and returns one score per token.
    """

    name: str

    async def score_tokens(self, token_ids: List[int]) -> List[float]:
        """
        Compute per-token scores for the given completion.

        Args:
            token_ids: Completion token IDs (not including prompt).

        Returns:
            List of scores, one per token. Length must equal len(token_ids).
        """
        ...


@runtime_checkable
class ActionLevelScorer(Protocol):
    """
    Per-action intrinsic scoring (scalar).

    Use cases:
    - LLM-as-a-judge
    - Verifier models
    - Self-consistency scores

    The scorer receives the full context and returns a single scalar.
    """

    name: str

    async def score_action(self, prompt: str, completion: str) -> float:
        """
        Compute a scalar score for the action.

        Args:
            prompt: The prompt text (rendered messages).
            completion: The agent's completion text.

        Returns:
            Scalar score for the action.
        """
        ...


IntrinsicScorer = Union[TokenLevelScorer, ActionLevelScorer]


@dataclass
class VLLMTeacherScorer:
    """
    TokenLevelScorer backed by vLLM server.

    Computes teacher logprobs via teacher-forced prefill using
    the /v1/completions endpoint with echo=True and max_tokens=0.
    """

    base_url: str
    model: str
    name: str = "teacher_logps"
    timeout: float = 60.0

    async def score_tokens(self, token_ids: List[int]) -> List[float]:
        """
        Compute per-token logprobs from teacher model.

        Uses vLLM's echo mode to get logprobs for existing tokens.
        """
        import aiohttp

        url = f"{self.base_url}/v1/completions"
        payload = {
            "model": self.model,
            "prompt": token_ids,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

        logprobs_data = data["choices"][0].get("logprobs", {})
        token_logprobs = logprobs_data.get("token_logprobs", [])

        # First token has no prior context, skip it
        # Return logprobs for completion tokens only
        if token_logprobs and token_logprobs[0] is None:
            token_logprobs = token_logprobs[1:]

        return [float(lp) if lp is not None else 0.0 for lp in token_logprobs]


def make_vllm_teacher_scorer(
    base_url: str,
    model: str,
    *,
    name: str = "teacher_logps",
    timeout: float = 60.0,
) -> TokenLevelScorer:
    """
    Create a TokenLevelScorer backed by vLLM for teacher logprobs.

    This is used for On-Policy Distillation (OPD) where a teacher model
    provides per-token supervision via teacher-forced prefill.

    Args:
        base_url: vLLM server URL (e.g., "http://localhost:8001").
        model: Teacher model name.
        name: Attachment key for scores (default: "teacher_logps").
        timeout: Request timeout in seconds.

    Returns:
        TokenLevelScorer that computes teacher logprobs.

    Example:
        >>> teacher = make_vllm_teacher_scorer(
        ...     base_url="http://localhost:8001",
        ...     model="Qwen/Qwen3-32B",
        ... )
        >>> agent = Agent(client=client, ..., scorers=[teacher])
    """
    return VLLMTeacherScorer(
        base_url=base_url,
        model=model,
        name=name,
        timeout=timeout,
    )
