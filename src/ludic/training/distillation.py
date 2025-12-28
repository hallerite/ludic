"""
On-Policy Distillation (OPD) support for Ludic.

This module provides the TeacherClient protocol and implementations for computing
teacher model logprobs on student-sampled tokens.

Reference: https://thinkingmachines.ai/blog/on-policy-distillation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import tinker


class TeacherClient(Protocol):
    """
    Protocol for teacher models that can compute logprobs on given tokens.

    Unlike ChatClient (which generates text), TeacherClient only evaluates
    the probability of existing token sequences. This is used for on-policy
    distillation where the student samples trajectories and the teacher
    provides per-token supervision via reverse KL.

    The key distinction from ChatClient:
    - ChatClient.complete_tokens() generates new tokens
    - TeacherClient.compute_logprobs() evaluates existing tokens
    """

    async def compute_logprobs(
        self,
        token_ids: List[int],
    ) -> List[float]:
        """
        Compute per-token log probabilities for the given sequence.

        Args:
            token_ids: Full sequence [prompt + completion] as token IDs.
                       The teacher evaluates P(token_i | token_0..i-1) for each i.

        Returns:
            List of logprobs, one per token position (excluding the first token
            which has no prior context). Length = len(token_ids) - 1.

            logprobs[i] = log P(token_ids[i+1] | token_ids[0..i])
        """
        ...


@dataclass
class TinkerTeacherClient:
    """
    TeacherClient backed by a Tinker SamplingClient.

    Uses Tinker's compute_logprobs_async API which efficiently computes
    logprobs in a single forward pass without generating new tokens.

    Example:
        >>> import tinker
        >>> service_client = tinker.ServiceClient()
        >>> sampling_client = service_client.create_sampling_client(
        ...     base_model="Qwen/Qwen3-32B"
        ... )
        >>> teacher = TinkerTeacherClient(sampling_client=sampling_client)
        >>> logprobs = await teacher.compute_logprobs([1, 2, 3, 4, 5])
    """

    sampling_client: Any  # tinker.SamplingClient - use Any to avoid hard dep

    async def compute_logprobs(self, token_ids: List[int]) -> List[float]:
        import tinker

        model_input = tinker.ModelInput.from_ints(token_ids)
        # compute_logprobs_async returns logprobs for all positions including first
        # First token has no prior, so we skip it
        logprobs = await self.sampling_client.compute_logprobs_async(model_input)
        return list(logprobs[1:])


@dataclass
class VLLMTeacherClient:
    """
    TeacherClient backed by a vLLM server.

    Uses the OpenAI-compatible /v1/completions endpoint with echo=True
    and logprobs enabled to get per-token probabilities without generation.

    Note: This requires the prompt to be passed as token IDs and the server
    to support the prompt_logprobs parameter (vLLM extension).

    Example:
        >>> teacher = VLLMTeacherClient(
        ...     base_url="http://localhost:8000",
        ...     model="Qwen/Qwen3-32B",
        ... )
        >>> logprobs = await teacher.compute_logprobs([1, 2, 3, 4, 5])
    """

    base_url: str
    model: str
    timeout: float = 60.0

    async def compute_logprobs(self, token_ids: List[int]) -> List[float]:
        import httpx

        # vLLM's /v1/completions endpoint with prompt as token IDs
        # echo=True returns logprobs for prompt tokens
        # max_tokens=0 prevents any generation
        request_body = {
            "model": self.model,
            "prompt": token_ids,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,  # Return top-1 logprobs
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/completions",
                json=request_body,
            )
            response.raise_for_status()
            result = response.json()

        # Extract logprobs from response
        # vLLM returns logprobs in choices[0].logprobs.token_logprobs
        choice = result["choices"][0]
        token_logprobs = choice.get("logprobs", {}).get("token_logprobs", [])

        if token_logprobs is None:
            raise ValueError(
                "vLLM response did not include token_logprobs. "
                "Ensure the server supports logprobs with echo=True."
            )

        # First token has no logprob (or is None), skip it
        # The rest should align with token_ids[1:]
        logprobs = []
        for lp in token_logprobs[1:]:
            if lp is None:
                # Some implementations return None for special tokens
                logprobs.append(float("-inf"))
            else:
                logprobs.append(float(lp))

        return logprobs
