from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from ludic.agents import ReToolAgent, CodeExecutionResult
from ludic.context.full_dialog import FullDialog
from ludic.inference.client import ChatClient, ChatResponse
from ludic.inference.request import TokenCompletionRequest, InferenceSpec, ReturnSpec
from ludic.inference.sampling import SamplingParams
from ludic.interaction.single_agent import SingleAgentSyncProtocol
from ludic.parsers import boxed_parser
from ludic.training.batching import RolloutEngine
from ludic.training.types import EnvSpec, ProtocolSpec, RolloutRequest
from tests._mocks import MockChatTemplate, mock_tokenize


class RollingReToolClient(ChatClient):
    """
    A mock client that returns predetermined responses with proper tokenization.
    Uses mock_tokenize() for consistent token IDs with MockChatTemplate.
    """
    def __init__(
        self,
        responses: List[str],
    ) -> None:
        self._responses = list(responses)
        self._i = 0

    async def complete_tokens(
        self,
        request: TokenCompletionRequest,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        if self._i >= len(self._responses):
            raise RuntimeError("RollingReToolClient exhausted all responses")

        text = self._responses[self._i]
        # Use consistent tokenization matching MockChatTemplate
        completion_token_ids = mock_tokenize(text)
        # Generate fake logprobs (one per token)
        completion_logprobs = [-0.1] * len(completion_token_ids)

        resp = ChatResponse(
            text=text,
            prompt_token_ids=request.prompt_token_ids,
            completion_token_ids=completion_token_ids,
            completion_logprobs=completion_logprobs,
            finish_reason="stop",
        )

        self._i += 1
        return resp, {"mode": "token_in"}

    def sync_weights(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError


@pytest.mark.skip(
    reason="Turn concatenation requires real tokenization semantics that are hard to mock. "
    "The mock chat template adds formatting around messages, so completion tokens don't "
    "appear at the same positions when re-tokenized as part of the next prompt. "
    "This test should be run as an integration test with a real tokenizer."
)
@pytest.mark.asyncio
async def test_retool_turn_concatenation_end_to_end(env_registry) -> None:
    responses = [
        "Compute: <code>\n```python\nresult = 2 + 2\n```\n</code>",
        "Answer: \\boxed{4}",
    ]

    client = RollingReToolClient(responses=responses)

    async def code_sandbox(code: str) -> CodeExecutionResult:
        return CodeExecutionResult(output="4", success=True)

    agent = ReToolAgent(
        client=client,
        model="mock",
        ctx=FullDialog(),
        parser=boxed_parser,
        code_sandbox=code_sandbox,
        max_code_blocks=2,
        chat_template=MockChatTemplate(),
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

    # Verify turn structure
    assert item.meta["turn_step_count"] == 2  # Two agent steps in the turn
    assert item.meta["turn_has_env_step"] is True
    assert item.meta["step_kind"] == "env"
    assert item.meta["reward"] == pytest.approx(1.1)  # 1.0 from env + 0.1 from parser

    # Verify token structure: action tokens should be marked with 1
    completion_len = item.meta["completion_length"]
    assert completion_len > 0
    assert sum(item.action_mask) == completion_len

    # Verify logprobs are present and match completion length
    assert item.actor_logps is not None
    assert len(item.actor_logps.token_logps) == completion_len
    # All logprobs are -0.1 (our mock constant)
    assert all(lp == pytest.approx(-0.1) for lp in item.actor_logps.token_logps)
