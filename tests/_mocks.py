from __future__ import annotations

import torch

from typing import Any, Optional, Tuple, Mapping, Dict, List
from ludic.envs.single_agent_env import SingleAgentEnv
from ludic.types import StepOutcome, Observation, Info, Message
from ludic.inference.client import ChatResponse, ChatClient
from ludic.inference.request import TokenCompletionRequest
from ludic.inference.chat_template import ChatTemplate, TemplateResult
from ludic.inference.tool_parser import ToolParseResult, ToolParser
from ludic.agents.base_agent import Agent
from ludic.context.base import ContextStrategy
from ludic.context.full_dialog import FullDialog
from ludic.parsers import Parser, ParseResult

# ---- A default pass-through parser for mocks ----
def _mock_parser(raw: str) -> ParseResult:
    """A simple parser that just passes the text through."""
    return ParseResult(action=raw, reward=0.0, obs=None)


# ---- Mock ChatTemplate for token-in API testing ----

class MockChatTemplate(ChatTemplate):
    """
    A mock ChatTemplate that returns predictable token IDs.
    Used for testing the token-in API without requiring a real tokenizer.

    Token generation uses character-based encoding to simulate real tokenization:
    - Each character maps to a consistent token ID
    - The prompt grows as messages are added
    - This enables turn concatenation tests to work correctly
    """

    def __init__(self, tool_parser: Optional[ToolParser] = None) -> None:
        self._tool_parser = tool_parser
        self._call_count = 0
        self.last_messages: Optional[List[Message]] = None
        self.last_tools: Optional[List[Dict[str, Any]]] = None

    def apply(
        self,
        messages: List[Message],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> TemplateResult:
        self._call_count += 1
        self.last_messages = list(messages)
        self.last_tools = list(tools) if tools is not None else None
        # Generate mock token IDs based on message content
        text_parts = []
        for msg in messages:
            text_parts.append(f"[{msg.get('role', 'unknown')}]")
            if msg.get("content"):
                text_parts.append(str(msg["content"]))
            # Include tool_calls content if present
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        text_parts.append(str(tc.get("function", {}).get("name", "")))
        prompt_text = " ".join(text_parts)

        # Simple mock: each character becomes a deterministic token ID
        # Same character = same token ID, enabling proper turn concatenation
        prompt_token_ids = [(ord(c) % 500) + 1 for c in prompt_text]

        return TemplateResult(
            prompt_token_ids=prompt_token_ids,
            prompt_text=prompt_text,
        )

    def parse_tool_calls(
        self,
        completion_text: str,
    ) -> ToolParseResult:
        if self._tool_parser:
            return self._tool_parser.parse(completion_text)
        return ToolParseResult(tool_calls=None, parse_error=False)

    def supports_tools(self) -> bool:
        return self._tool_parser is not None

# ---- Common test tools --------------------------------------------------

def calculator_tool(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b


def mock_tokenize(text: str) -> List[int]:
    """
    Convert text to token IDs matching MockChatTemplate's tokenization.
    Same character = same token ID, enabling proper turn concatenation.
    """
    return [(ord(c) % 500) + 1 for c in text]


# ---- Mock client ---------------------------------------------------------

class MockClient(ChatClient):
    """Mock client supporting complete_tokens() for token-in API."""

    def __init__(
        self,
        text: str = "1",
        finish_reason: str = "stop",
    ) -> None:
        self._text = text
        self._finish_reason = finish_reason

    async def complete_tokens(
        self,
        request: TokenCompletionRequest,
    ) -> tuple[ChatResponse, Dict[str, Any]]:
        """Token-in API: complete from pre-tokenized prompt."""
        # Use consistent tokenization matching MockChatTemplate
        completion_token_ids = mock_tokenize(self._text)
        resp = ChatResponse(
            text=self._text,
            finish_reason=self._finish_reason,
            prompt_token_ids=request.prompt_token_ids,
            completion_token_ids=completion_token_ids,
        )
        return resp, {"mode": "token_in", "prompt_text": request.prompt_text}

    def sync_weights(self, params: Mapping[str, torch.Tensor], **kwargs) -> str:  # type: ignore[name-defined]
        return "mock-version"

class MockAgent(Agent):
    """
    Real Agent wired to the MockClient.
    Uses FullDialog and a pass-through parser by default.
    """
    def __init__(
        self,
        client: ChatClient | None = None,
        model: str = "mock",
        ctx: ContextStrategy | None = None,
        parser: Parser = _mock_parser,
    ) -> None:
        super().__init__(
            client=client or MockClient(),
            model=model,
            ctx=ctx or FullDialog(),
            parser=parser,
            chat_template=MockChatTemplate(),
        )


# ---- Seedable Mock Client for GRPO Test ----

class SeedableMockClient(ChatClient):
    """
    A mock client that returns a deterministic response based on the
    sampling_seed provided. Also returns mock token IDs.
    """
    def __init__(self, seed_map: Dict[int, str]) -> None:
        self._seed_map = seed_map

    async def complete_tokens(
        self,
        request: TokenCompletionRequest,
    ) -> tuple[ChatResponse, Dict[str, Any]]:
        """Token-in API version."""
        if request.seed is None:
            raise ValueError("SeedableMockClient requires request.seed to be set")
        text_out = self._seed_map.get(int(request.seed), "DEFAULT_FALLBACK")

        resp = ChatResponse(
            text=text_out,
            prompt_token_ids=request.prompt_token_ids,
            completion_token_ids=[10, 11],
        )
        return resp, {"mode": "token_in"}

    def sync_weights(self, params: Mapping[str, torch.Tensor], **kwargs) -> str:
        # Not needed for this test
        return "mock-version"

class SeedableMockAgent(Agent):
    """
    An agent wired to the SeedableMockClient.
    """
    def __init__(
        self,
        seed_map: Dict[int, str],
        ctx: ContextStrategy | None = None,
        parser: Parser = _mock_parser,
    ) -> None:
        super().__init__(
            client=SeedableMockClient(seed_map), 
            model="seedable_mock",
            ctx=ctx or FullDialog(),
            parser=parser,
            chat_template=MockChatTemplate(),
        )


# ---- Mock env ------------------------------------------------------------

class MockEnv(SingleAgentEnv):
    """
    Minimal env for exercising the interaction loop only.

    - reset() -> returns a fixed instruction.
    - step("1") -> reward=1.0, terminated=True, obs="✅ done".
    - step(other) -> reward=-0.1, keep going with obs="❌ try again {t}/{max}".
    - Truncates after `max_steps`.
    - No snapshot support on purpose.
    """

    def __init__(self, *, max_steps: int = 8, target: str = "1") -> None:
        super().__init__()  # <-- Added call to super()
        self.max_steps = max_steps
        self.target = target
        self._t = 0
        self._obs: Observation = "Reply with '1' to finish."

    @property
    def suggested_sysprompt(self) -> Optional[str]:
        return "You are terse. Output only the final answer."

    def env_reset(self, *, seed: Optional[int] = None) -> Tuple[Observation, Info]:  # <-- Renamed
        self._t = 0
        self._obs = "Reply with '1' to finish."
        return self._obs, {}

    def env_step(self, action: str) -> StepOutcome:  # <-- Renamed
        if self._t >= self.max_steps:
            return StepOutcome(obs=self._obs, reward=0.0, truncated=True, terminated=False, info={})

        self._t += 1
        if action.strip() == self.target:
            self._obs = "✅ done"
            return StepOutcome(obs=self._obs, reward=1.0, truncated=False, terminated=True, info={})

        self._obs = f"❌ try again {self._t}/{self.max_steps}"
        truncated = self._t >= self.max_steps
        return StepOutcome(
            obs=self._obs, reward=-0.1, truncated=truncated, terminated=False, info={"attempt": self._t}
        )

    def env_current_obs(self) -> Observation:  # <-- Renamed
        return self._obs
