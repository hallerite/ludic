from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from ludic.agents import ReToolAgent, CodeExecutionResult
from ludic.context.full_dialog import FullDialog
from ludic.inference.client import ChatClient, ChatResponse
from ludic.inference.request import ChatCompletionRequest, InferenceSpec, ReturnSpec
from ludic.inference.sampling import SamplingParams
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.parsers import boxed_parser
from ludic.training.batching import RolloutEngine
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest


class RollingReToolClient(ChatClient):
    def __init__(
        self,
        responses: List[str],
        completion_ids: List[List[int]],
        completion_logprobs: List[List[float]],
    ) -> None:
        self._responses = list(responses)
        self._completion_ids = list(completion_ids)
        self._completion_logprobs = list(completion_logprobs)
        self._i = 0
        self._rolling_prompt_ids: List[int] = []
        self._last_prompt_len = 0

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        if self._i >= len(self._responses):
            raise RuntimeError("RollingReToolClient exhausted all responses")

        prompt_len = len(request.messages)
        if self._i == 0:
            prompt_token_ids = list(range(prompt_len))
        else:
            if prompt_len < self._last_prompt_len:
                raise RuntimeError("Prompt length decreased between steps")
            extra = prompt_len - self._last_prompt_len
            start = (self._rolling_prompt_ids[-1] + 1) if self._rolling_prompt_ids else 0
            extra_tokens = list(range(start, start + extra))
            prompt_token_ids = list(self._rolling_prompt_ids) + extra_tokens

        completion_token_ids = list(self._completion_ids[self._i])
        completion_logprobs = list(self._completion_logprobs[self._i])

        resp = ChatResponse(
            text=self._responses[self._i],
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
            completion_logprobs=completion_logprobs,
            finish_reason="stop",
        )

        self._rolling_prompt_ids = list(prompt_token_ids) + list(completion_token_ids)
        self._last_prompt_len = prompt_len
        self._i += 1
        return resp, {}

    def sync_weights(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError


@pytest.mark.asyncio
async def test_retool_turn_concatenation_end_to_end(env_registry) -> None:
    responses = [
        "Compute: <code>\n```python\nresult = 2 + 2\n```\n</code>",
        "Answer: \\boxed{4}",
    ]
    completion_ids = [[101, 102], [201]]
    completion_logprobs = [[-0.1, -0.2], [-0.3]]

    client = RollingReToolClient(
        responses=responses,
        completion_ids=completion_ids,
        completion_logprobs=completion_logprobs,
    )

    async def code_sandbox(code: str) -> CodeExecutionResult:
        return CodeExecutionResult(output="4", success=True)

    agent = ReToolAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=boxed_parser,
        code_sandbox=code_sandbox,
        max_code_blocks=2,
    )

    protocol_registry = {
        "retool_protocol": lambda: SingleAgentSyncProtocol(agent=agent),
    }
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
    )

    request = RolloutRequest(
        env=EnvSpec(kind="mock", kwargs={"max_steps": 2, "target": "4"}),
        protocol=ProtocolSpec(kind="retool_protocol"),
        num_episodes=1,
        inference=InferenceSpec(
            sampling=SamplingParams(temperature=0.0, max_tokens=32),
            return_=ReturnSpec.for_rl(),
        ),
    )

    class ConstantCreditAssigner:
        def compute(self, rollouts):  # type: ignore[no-untyped-def]
            return {(r.id, s.index): 1.0 for r in rollouts for s in r.steps}

    batch = await engine.generate_batch(
        requests=[request],
        max_steps=3,
        credit_assigner=ConstantCreditAssigner(),
    )

    assert len(batch.items) == 1
    item = batch.items[0]

    assert item.action_mask.count(1) == 3
    first_action_idx = item.action_mask.index(1)
    assert 0 in item.action_mask[first_action_idx + 1 :]
    assert item.meta["turn_step_count"] == 2
    assert item.meta["turn_has_env_step"] is True
    assert item.meta["step_kind"] == "env"
    assert item.meta["reward"] == pytest.approx(1.1)

    assert item.actor_logps is not None
    assert item.actor_logps.token_logps == [-0.1, -0.2, -0.3]
