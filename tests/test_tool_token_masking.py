from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest

from ludic.agents.react_agent import ReActAgent
from ludic.context.full_dialog import FullDialog
from ludic.inference.client import ChatClient, ChatResponse
from ludic.inference.request import ChatCompletionRequest, InferenceSpec, ReturnSpec
from ludic.inference.sampling import SamplingParams
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.training.batching import RolloutEngine
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from ludic.types import Rollout

from tests._mocks import _mock_parser, calculator_tool


class ToolTraceClient(ChatClient):
    def __init__(self, steps: List[Dict[str, Any]]) -> None:
        self._steps = steps
        self._i = 0
        self.last_messages: Optional[List[Dict[str, Any]]] = None

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        if self._i >= len(self._steps):
            raise RuntimeError("ToolTraceClient exhausted all steps")

        step = self._steps[self._i]
        self._i += 1
        self.last_messages = list(request.messages)

        prompt_len = len(request.messages)
        prompt_token_ids = list(range(prompt_len))
        completion_token_ids = [100, 101]

        resp = ChatResponse(
            text=step.get("content", "") or "",
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            finish_reason="stop",
        )

        raw_response = {
            "choices": [
                {
                    "message": step,
                    "finish_reason": "tool_calls" if step.get("tool_calls") else "stop",
                }
            ]
        }
        return resp, {"raw_response": raw_response}

    def sync_weights(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class ConstantCreditAssigner:
    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def compute(self, rollouts: List[Rollout]) -> Dict[Tuple[str, int], float]:
        out: Dict[Tuple[str, int], float] = {}
        for r in rollouts:
            for s in r.steps:
                out[(r.id, s.index)] = self.value
        return out


@pytest.mark.asyncio
async def test_tool_role_tokens_are_masked_out(env_registry) -> None:
    steps = [
        {
            "content": "Calling the tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "calculator_tool",
                        "arguments": json.dumps({"a": 2, "b": 2}),
                    },
                }
            ],
        },
        {
            "content": "final",
            "tool_calls": None,
        },
    ]
    client = ToolTraceClient(steps)

    agent = ReActAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=_mock_parser,
        tools=[calculator_tool],
        max_react_steps=2,
    )

    protocol_registry = {
        "react_protocol": lambda: SingleAgentSyncProtocol(agent=agent)
    }
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={"max_steps": 1, "target": "final"}),
        protocol=ProtocolSpec(kind="react_protocol"),
        num_episodes=1,
        inference=InferenceSpec(
            sampling=SamplingParams(temperature=0.0, max_tokens=8),
            return_=ReturnSpec.for_eval(return_token_ids=True),
        ),
    )

    batch = await engine.generate_batch(
        requests=[request],
        max_steps=1,
        credit_assigner=ConstantCreditAssigner(),
        timeout_s=None,
        concurrency=1,
    )

    assert len(batch.items) == 1
    item = batch.items[0]

    assert client.last_messages is not None
    assert any(msg.get("role") == "tool" for msg in client.last_messages)

    prompt_len = item.meta["prompt_length"]
    assert prompt_len == len(client.last_messages)
    assert all(v == 0 for v in item.action_mask[:prompt_len])
    assert all(v == 1 for v in item.action_mask[prompt_len:])
