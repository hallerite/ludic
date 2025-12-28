from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import tinker
from tinker_cookbook.tokenizer_utils import Tokenizer

from ludic.inference.client import ChatClient
from ludic.inference.request import TokenCompletionRequest
from ludic.types import ChatResponse


@dataclass
class TinkerChatClient(ChatClient):
    """
    Ludic ChatClient adapter backed by a Tinker SamplingClient.

    Token-in adapter: expects pre-tokenized prompts (TokenCompletionRequest).
    Update the sampling client between training steps to keep rollouts on-policy.
    """

    sampling_client: tinker.SamplingClient
    tokenizer: Tokenizer
    policy_version: Optional[str] = None

    def set_sampling_client(
        self,
        sampling_client: tinker.SamplingClient,
        *,
        policy_version: Optional[str] = None,
    ) -> None:
        self.sampling_client = sampling_client
        if policy_version is not None:
            self.policy_version = policy_version

    async def complete_tokens(
        self,
        request: TokenCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        if request.prompt_token_ids is None:
            raise ValueError("TokenCompletionRequest.prompt_token_ids is required.")

        model_input = tinker.ModelInput.from_ints(list(request.prompt_token_ids))

        sampling_params = tinker.SamplingParams(
            max_tokens=int(request.sampling.max_tokens),
            temperature=float(request.sampling.temperature),
            top_p=float(request.sampling.top_p),
            stop=request.sampling.stop,
            seed=int(request.seed) if request.seed is not None else None,
        )

        sample_result = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )

        sequence = sample_result.sequences[0]
        completion_tokens = list(sequence.tokens)
        completion_logprobs = sequence.logprobs

        if request.return_.return_chosen_logprobs and completion_logprobs is None:
            raise ValueError(
                "Tinker sampling did not return logprobs, but return_chosen_logprobs=True."
            )

        if not request.return_.return_chosen_logprobs:
            completion_logprobs = None

        text = self.tokenizer.decode(
            completion_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        response = ChatResponse(
            text=text,
            completion_token_ids=completion_tokens,
            completion_logprobs=list(completion_logprobs) if completion_logprobs else None,
            finish_reason=sequence.stop_reason,
            prompt_token_ids=list(request.prompt_token_ids),
        )

        info: Dict[str, Any] = {
            "finish_reason": sequence.stop_reason,
            "policy_version": self.policy_version,
            "prompt_length": len(request.prompt_token_ids),
            "completion_length": len(completion_tokens),
        }
        ignored_params: Dict[str, Any] = {}
        if request.sampling.frequency_penalty:
            ignored_params["frequency_penalty"] = float(request.sampling.frequency_penalty)
        if request.sampling.presence_penalty:
            ignored_params["presence_penalty"] = float(request.sampling.presence_penalty)
        if request.return_.top_logprobs_k > 1:
            ignored_params["top_logprobs_k"] = int(request.return_.top_logprobs_k)
        if ignored_params:
            info["ignored_sampling_params"] = ignored_params

        return response, info

    def sync_weights(
        self,
        params: Mapping[str, Any],
        *,
        timeout_s: float = 600.0,
        version: Optional[str | int] = None,
    ) -> str:
        raise RuntimeError(
            "TinkerChatClient does not support direct weight pushes. "
            "Use training_client.save_weights_and_get_sampling_client(...) and "
            "swap the sampling client via set_sampling_client()."
        )
