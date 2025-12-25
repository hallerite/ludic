from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import Tokenizer

from ludic.inference.client import ChatClient
from ludic.inference.request import ChatCompletionRequest
from ludic.types import ChatResponse, Message


def _coerce_messages(messages: Iterable[Message]) -> list[renderers.Message]:
    out: list[renderers.Message] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(
                "TinkerChatClient only supports text-only messages with string role/content."
            )
        out.append({"role": role, "content": content})
    return out


def _resolve_renderer(
    *,
    model_name: str,
    tokenizer: Tokenizer,
    renderer_name: Optional[str],
    renderer: Optional[renderers.Renderer],
) -> renderers.Renderer:
    if renderer is not None and renderer_name is not None:
        raise ValueError("Provide either renderer or renderer_name, not both.")
    if renderer is not None:
        return renderer
    name = renderer_name or model_info.get_recommended_renderer_name(model_name)
    return renderers.get_renderer(name, tokenizer)


@dataclass
class TinkerChatClient(ChatClient):
    """
    Ludic ChatClient adapter backed by a Tinker SamplingClient.

    This is intended for on-policy rollouts where sampling and training are both
    handled by Tinker. Update the sampling client between training steps to keep
    the rollouts strictly on-policy.
    """

    sampling_client: tinker.SamplingClient
    model_name: str
    tokenizer: Tokenizer
    renderer_name: Optional[str] = None
    renderer: Optional[renderers.Renderer] = None
    policy_version: Optional[str] = None

    def __post_init__(self) -> None:
        self._renderer = _resolve_renderer(
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            renderer_name=self.renderer_name,
            renderer=self.renderer,
        )

    def set_sampling_client(
        self,
        sampling_client: tinker.SamplingClient,
        *,
        policy_version: Optional[str] = None,
    ) -> None:
        self.sampling_client = sampling_client
        if policy_version is not None:
            self.policy_version = policy_version

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        if request.tools is not None:
            raise NotImplementedError(
                "TinkerChatClient does not support tool-calling. "
                "Serialize tool use into text prompts instead."
            )

        messages = _coerce_messages(request.messages)
        model_input = self._renderer.build_generation_prompt(messages)

        stop = request.sampling.stop or self._renderer.get_stop_sequences()
        sampling_params = tinker.SamplingParams(
            max_tokens=int(request.sampling.max_tokens),
            temperature=float(request.sampling.temperature),
            top_p=float(request.sampling.top_p),
            stop=stop,
            seed=request.seed,
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

        parsed_message, _parse_ok = self._renderer.parse_response(completion_tokens)
        text = renderers.ensure_text(parsed_message["content"])

        prompt_token_ids: Optional[List[int]]
        try:
            prompt_token_ids = model_input.to_ints()
        except Exception:
            prompt_token_ids = None

        return_tokens = request.return_.return_token_ids or request.return_.return_chosen_logprobs

        response = ChatResponse(
            text=text,
            completion_token_ids=completion_tokens if return_tokens else None,
            completion_logprobs=list(completion_logprobs) if completion_logprobs else None,
            finish_reason=sequence.stop_reason,
            prompt_token_ids=prompt_token_ids if return_tokens else None,
        )

        info: Dict[str, Any] = {
            "finish_reason": sequence.stop_reason,
            "policy_version": self.policy_version,
            "renderer": type(self._renderer).__name__,
            "prompt_length": len(prompt_token_ids) if prompt_token_ids is not None else None,
            "completion_length": len(completion_tokens),
        }
        if request.sampling.frequency_penalty or request.sampling.presence_penalty:
            info["ignored_sampling_params"] = {
                "frequency_penalty": float(request.sampling.frequency_penalty),
                "presence_penalty": float(request.sampling.presence_penalty),
            }

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
