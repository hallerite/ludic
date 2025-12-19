from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import pytest

from ludic.inference.vllm_client import VLLMChatClient
from ludic.inference.request import ChatCompletionRequest, ReturnSpec
from ludic.inference.sampling import SamplingParams
from ludic.inference.extensions import BackendExtensions, VLLMExtensions


@dataclass(frozen=True)
class DummyExtensions(BackendExtensions):
    kind: str = "dummy"


class _DummyChatCompletions:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    async def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self._captured.update(kwargs)

        @dataclass
        class _DummyMessage:
            content: str = "ok"

        @dataclass
        class _DummyChoice:
            message: _DummyMessage = field(default_factory=_DummyMessage)
            finish_reason: str = "stop"
            token_ids: object | None = None
            logprobs: object | None = None

        class _DummyResp:
            choices = [_DummyChoice()]
            prompt_token_ids = None

            def model_dump(self, **_kwargs):  # type: ignore[no-untyped-def]
                return {"dummy": True}

        return _DummyResp()


class _DummyChat:
    def __init__(self, captured: dict) -> None:
        self.completions = _DummyChatCompletions(captured)


class _DummyAsyncClient:
    def __init__(self, captured: dict) -> None:
        self.chat = _DummyChat(captured)


@pytest.mark.asyncio
async def test_vllm_client_rejects_unknown_backend_extensions() -> None:
    # Bypass __init__ (which expects a running server); the error we test is raised
    # before any network call is attempted.
    client = object.__new__(VLLMChatClient)

    req = ChatCompletionRequest(
        model="mock",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingParams(),
        return_=ReturnSpec.for_eval(return_token_ids=False),
        extensions=DummyExtensions(),
    )

    with pytest.raises(TypeError, match="unsupported request\\.extensions"):
        await client.complete(req)


@pytest.mark.asyncio
async def test_vllm_client_rejects_invalid_max_think() -> None:
    client = object.__new__(VLLMChatClient)

    req = ChatCompletionRequest(
        model="mock",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingParams(),
        return_=ReturnSpec.for_eval(return_token_ids=False),
        extensions=VLLMExtensions(max_think=0),
    )

    with pytest.raises(ValueError, match="max_think must be a positive integer"):
        await client.complete(req)


@pytest.mark.asyncio
async def test_vllm_client_rejects_invalid_repetition_penalty() -> None:
    client = object.__new__(VLLMChatClient)

    req = ChatCompletionRequest(
        model="mock",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingParams(),
        return_=ReturnSpec.for_eval(return_token_ids=False),
        extensions=VLLMExtensions(repetition_penalty=0.0),
    )

    with pytest.raises(ValueError, match="repetition_penalty must be > 0"):
        await client.complete(req)


@pytest.mark.asyncio
async def test_vllm_client_passes_repetition_penalty_when_extension_used() -> None:
    captured: dict = {}
    client = object.__new__(VLLMChatClient)
    client._async_client = _DummyAsyncClient(captured)  # type: ignore[attr-defined]

    req = ChatCompletionRequest(
        model="mock",
        messages=[{"role": "user", "content": "hi"}],
        sampling=SamplingParams(),
        return_=ReturnSpec.for_eval(return_token_ids=False),
        extensions=VLLMExtensions(repetition_penalty=1.0),
    )

    await client.complete(req)
    assert captured["extra_body"]["repetition_penalty"] == 1.0
