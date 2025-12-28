from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Mapping, TYPE_CHECKING

import torch

from ludic.types import Observation, Info, Message, ChatResponse, TokenTrace
from ludic.inference.client import ChatClient
from ludic.inference.request import TokenCompletionRequest, InferenceSpec, ToolRequest
from ludic.context.base import ContextStrategy
from ludic.parsers import Parser, ParseResult

if TYPE_CHECKING:
    from ludic.inference.chat_template import ChatTemplate

_DEFAULT_INCOMPLETE_FEEDBACK = (
    "Your response was cut off because it exceeded the token limit. "
    "Please provide a shorter, more concise response."
)

_TOKEN_TRACE_KEYS = {
    "prompt_token_ids",
    "completion_token_ids",
    "completion_logprobs",
}


def _strip_token_trace_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove token-trace fields from client info to avoid duplicating large arrays.
    """
    if not info:
        return {}
    stripped = dict(info)
    for key in _TOKEN_TRACE_KEYS:
        stripped.pop(key, None)

    raw = stripped.get("raw_response")
    if isinstance(raw, dict):
        raw = dict(raw)
        raw.pop("prompt_token_ids", None)
        choices = raw.get("choices")
        if isinstance(choices, list):
            new_choices = []
            for choice in choices:
                if isinstance(choice, dict):
                    choice = dict(choice)
                    choice.pop("token_ids", None)
                    choice.pop("logprobs", None)
                new_choices.append(choice)
            raw["choices"] = new_choices
        stripped["raw_response"] = raw
    return stripped


@dataclass
class AgentActStep:
    prompt_messages: List[Message]
    action: str
    parse_result: Optional[ParseResult]  # None for external tool calls (not final actions)
    info: Dict[str, Any]
    trace: TokenTrace
    action_target: str  # "internal" | "external" | "env"
    loop_index: int
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


@dataclass
class AgentActResult:
    steps: List[AgentActStep]

    @property
    def final_step(self) -> AgentActStep:
        return self.steps[-1]


class Agent:
    """
    A stateful, logical actor that bundles inference, context, and parsing.

    It holds a reference to a (potentially shared) ChatClient and manages
    its own internal state via its ContextStrategy.
    """

    name: str = "agent"

    def __init__(
        self,
        *,
        client: ChatClient,
        model: str,
        ctx: ContextStrategy,
        parser: Parser,
        reject_incomplete_completions: bool = True,
        incomplete_completion_penalty: float = -0.1,
        incomplete_completion_feedback: str = _DEFAULT_INCOMPLETE_FEEDBACK,
        chat_template: Optional["ChatTemplate"] = None,
    ) -> None:
        """
        Initializes the Agent.

        Args:
            client: The ChatClient for inference.
            model: The model name this agent should use.
            ctx: An instance of a ContextStrategy for managing memory.
            parser: An instance of a Parser for decoding actions.
            reject_incomplete_completions: If True, completions that hit max_tokens
                (finish_reason="length") are treated as parse failures with feedback.
            incomplete_completion_penalty: Reward penalty for incomplete completions.
            incomplete_completion_feedback: Feedback message shown to agent when
                its completion is cut off.
            chat_template: ChatTemplate for token-in mode. If None, the agent
                will try to build an HFChatTemplate from client.tokenizer.
        """
        self._client = client
        self._model = model
        self._ctx = ctx
        self._parser = parser
        self._reject_incomplete = reject_incomplete_completions
        self._incomplete_penalty = incomplete_completion_penalty
        self._incomplete_feedback = incomplete_completion_feedback
        if chat_template is None:
            tokenizer = getattr(client, "tokenizer", None)
            if tokenizer is None or not callable(getattr(tokenizer, "apply_chat_template", None)):
                raise ValueError(
                    "Agent requires a chat_template for token-in inference or a client "
                    "with a HuggingFace-compatible tokenizer."
                )
            from ludic.inference.chat_template import HFChatTemplate

            chat_template = HFChatTemplate(tokenizer)
        self._chat_template = chat_template
        self.last_info: Dict[str, Any] = {}

    async def _infer_once(
        self,
        *,
        messages: List[Message],
        inference: Optional[InferenceSpec] = None,
        sampling_seed: Optional[int] = None,
        tools: Optional[ToolRequest] = None,
        timeout_s: Optional[float] = None,
    ) -> Tuple[ChatResponse, Dict[str, Any], Dict[str, Any], TokenTrace]:
        """
        Shared single inference helper.

        Uses token-in mode (applies template ourselves, calls completions endpoint).
        """
        inf = inference or InferenceSpec()

        # Token-in mode: apply chat template ourselves
        tool_schemas = tools.tools if tools else None
        template_result = self._chat_template.apply(
            messages,
            tools=tool_schemas,
            add_generation_prompt=True,
        )

        req = TokenCompletionRequest(
            model=self._model,
            prompt_token_ids=template_result.prompt_token_ids,
            prompt_text=template_result.prompt_text,
            sampling=inf.sampling,
            return_=inf.return_,
            seed=sampling_seed,
            extensions=inf.extensions,
        )

        coro = self._client.complete_tokens(req)

        if timeout_s is None:
            resp, client_info = await coro
        else:
            resp, client_info = await asyncio.wait_for(coro, timeout=timeout_s)

        public_info: Dict[str, Any] = _strip_token_trace_info(client_info)
        last_info: Dict[str, Any] = dict(public_info)
        # Store prompt and completion for logging/training
        last_info["chat_prompt_messages"] = messages
        last_info["chat_completion"] = {"role": "assistant", "content": resp.text}
        resp.merge_into_info(last_info)

        self.last_info = last_info
        trace = resp.to_trace()
        if trace is None:
            raise ValueError(
                "Missing token IDs from inference response. "
                "Ensure ReturnSpec.return_token_ids=True for all calls."
            )
        return resp, public_info, last_info, trace

    def reset(self, system_prompt: Optional[str] = None) -> None:
        """Resets the agent's internal context."""
        self._ctx.reset(system_prompt=system_prompt)
        
    def on_env_reset(self, obs: Observation, info: Info):
        """Called by the protocol *after* env.reset()."""
        self._ctx.on_env_reset(obs, info)
        
    def on_after_step(self, obs: Observation, info: Info):
        """Called by the protocol *after* env.step()."""
        self._ctx.on_after_step(obs, info)

    async def act(
        self,
        inference: Optional[InferenceSpec] = None,
        sampling_seed: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> AgentActResult:
        """
        Runs the think -> act -> parse cycle based on current context.

        This method does *not* take obs/info, as those are fed to the
        agent via on_env_reset() and on_after_step().

        Args:
            inference: The inference configuration for this step.
            sampling_seed: Optional per-request seed for backend sampling.
            timeout_s: Optional timeout for the inference call.

        Returns:
            AgentActResult containing one or more AgentActStep entries.
        """
        # 1. Think (prepare prompt messages from context)
        messages: List[Message] = self._ctx.on_before_act()

        # 2. Act (run inference)
        resp, _client_info, last_info, token_trace = await self._infer_once(
            messages=messages,
            inference=inference,
            sampling_seed=sampling_seed,
            timeout_s=timeout_s,
        )

        # 3. Update memory with the agent's own response
        self._ctx.on_after_act(resp)

        raw_action = resp.text

        # 4. Check for incomplete completion (hit max_tokens)
        if self._reject_incomplete and resp.finish_reason == "length":
            parse_result = ParseResult(
                action=None,
                reward=self._incomplete_penalty,
                obs=self._incomplete_feedback,
            )
            # Mark this in info for downstream tracking
            last_info["incomplete_completion"] = True
        else:
            # 5. Parse (format the raw text action)
            parse_result = self._parser(raw_action)

        step = AgentActStep(
            prompt_messages=messages,
            action=raw_action,
            parse_result=parse_result,
            info=last_info,
            trace=token_trace,
            action_target="env",
            loop_index=0,
        )
        return AgentActResult(steps=[step])

    def push_policy_update(
        self,
        params: Mapping[str, torch.Tensor],
        *,
        timeout_s: float = 600.0,
        version: Optional[str] = None,
    ) -> str:
        """Pushes updated policy parameters to the underlying runtime."""
        if not hasattr(self._client, "sync_weights"):
            raise RuntimeError(
                "Underlying ChatClient does not support policy weight updates "
                "(missing sync_weights)."
            )
        return self._client.sync_weights(
            params,
            timeout_s=timeout_s,
            version=version,
        )
