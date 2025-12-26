from __future__ import annotations

from dataclasses import dataclass

import pytest

from ludic.inference.vllm_client import VLLMChatClient
from ludic.inference.request import TokenCompletionRequest, ReturnSpec
from ludic.inference.sampling import SamplingParams
from ludic.inference.extensions import BackendExtensions, VLLMExtensions


@dataclass(frozen=True)
class DummyExtensions(BackendExtensions):
    kind: str = "dummy"


class _DummyCompletions:
    """Mock completions endpoint for token-in API tests."""

    def __init__(self, captured: dict) -> None:
        self._captured = captured

    async def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self._captured.update(kwargs)

        @dataclass
        class _DummyChoice:
            text: str = "ok"
            finish_reason: str = "stop"
            token_ids: list = None  # type: ignore[assignment]
            logprobs: object | None = None

            def __post_init__(self):
                if self.token_ids is None:
                    self.token_ids = [100, 101]  # Default mock completion tokens

        class _DummyResp:
            choices = [_DummyChoice()]
            prompt_token_ids = None

            def model_dump(self, **_kwargs):  # type: ignore[no-untyped-def]
                return {"dummy": True}

        return _DummyResp()


class _DummyAsyncClient:
    def __init__(self, captured: dict) -> None:
        self.completions = _DummyCompletions(captured)


@pytest.mark.asyncio
async def test_vllm_client_rejects_unknown_backend_extensions() -> None:
    # Bypass __init__ (which expects a running server); the error we test is raised
    # before any network call is attempted.
    client = object.__new__(VLLMChatClient)

    req = TokenCompletionRequest(
        model="mock",
        prompt_token_ids=[1, 2, 3],
        sampling=SamplingParams(),
        return_=ReturnSpec.for_eval(return_token_ids=False),
        extensions=DummyExtensions(),
    )

    with pytest.raises(TypeError, match="unsupported request\\.extensions"):
        await client.complete_tokens(req)


@pytest.mark.asyncio
async def test_vllm_client_rejects_invalid_max_think() -> None:
    client = object.__new__(VLLMChatClient)

    req = TokenCompletionRequest(
        model="mock",
        prompt_token_ids=[1, 2, 3],
        sampling=SamplingParams(),
        return_=ReturnSpec.for_eval(return_token_ids=False),
        extensions=VLLMExtensions(max_think=0),
    )

    with pytest.raises(ValueError, match="max_think must be a positive integer"):
        await client.complete_tokens(req)


@pytest.mark.asyncio
async def test_vllm_client_rejects_invalid_repetition_penalty() -> None:
    client = object.__new__(VLLMChatClient)

    req = TokenCompletionRequest(
        model="mock",
        prompt_token_ids=[1, 2, 3],
        sampling=SamplingParams(),
        return_=ReturnSpec.for_eval(return_token_ids=False),
        extensions=VLLMExtensions(repetition_penalty=0.0),
    )

    with pytest.raises(ValueError, match="repetition_penalty must be > 0"):
        await client.complete_tokens(req)


@pytest.mark.asyncio
async def test_vllm_client_passes_repetition_penalty_when_extension_used() -> None:
    captured: dict = {}
    client = object.__new__(VLLMChatClient)
    client._async_client = _DummyAsyncClient(captured)  # type: ignore[attr-defined]

    req = TokenCompletionRequest(
        model="mock",
        prompt_token_ids=[1, 2, 3],
        sampling=SamplingParams(),
        return_=ReturnSpec.for_eval(return_token_ids=False),
        extensions=VLLMExtensions(repetition_penalty=1.0),
    )

    await client.complete_tokens(req)
    assert captured["extra_body"]["repetition_penalty"] == 1.0


# ---- Tests for complete_tokens (token-in API) ----


@pytest.mark.asyncio
async def test_complete_tokens_requires_prompt_token_ids() -> None:
    """complete_tokens raises if prompt_token_ids is missing."""
    client = object.__new__(VLLMChatClient)
    client._async_client = _DummyAsyncClient({})

    req = TokenCompletionRequest(
        model="mock",
        prompt_token_ids=None,  # type: ignore[arg-type]
        sampling=SamplingParams(),
    )

    with pytest.raises(ValueError, match="prompt_token_ids is required"):
        await client.complete_tokens(req)


@pytest.mark.asyncio
async def test_complete_tokens_calls_completions_endpoint() -> None:
    """complete_tokens calls completions.create, not chat.completions.create."""
    captured: dict = {}
    client = object.__new__(VLLMChatClient)
    client._async_client = _DummyAsyncClient(captured)

    req = TokenCompletionRequest(
        model="test-model",
        prompt_token_ids=[1, 2, 3],
        sampling=SamplingParams(),
    )

    resp, info = await client.complete_tokens(req)

    assert captured["model"] == "test-model"
    assert captured["prompt"] == [1, 2, 3]
    assert info["mode"] == "token_in"


@pytest.mark.asyncio
async def test_complete_tokens_passes_seed() -> None:
    """complete_tokens passes sampling seed to the backend."""
    captured: dict = {}
    client = object.__new__(VLLMChatClient)
    client._async_client = _DummyAsyncClient(captured)

    req = TokenCompletionRequest(
        model="mock",
        prompt_token_ids=[1, 2, 3],
        sampling=SamplingParams(),
        seed=42,
    )

    await client.complete_tokens(req)

    assert captured["seed"] == 42


@pytest.mark.asyncio
async def test_complete_tokens_passes_return_token_ids() -> None:
    """complete_tokens passes return_token_ids via extra_body."""
    captured: dict = {}
    client = object.__new__(VLLMChatClient)
    client._async_client = _DummyAsyncClient(captured)

    req = TokenCompletionRequest(
        model="mock",
        prompt_token_ids=[1, 2, 3],
        sampling=SamplingParams(),
        return_=ReturnSpec(return_token_ids=True),
    )

    await client.complete_tokens(req)

    assert captured["extra_body"]["return_token_ids"] is True


@pytest.mark.asyncio
async def test_complete_tokens_response_uses_our_prompt_tokens() -> None:
    """complete_tokens returns our prompt_token_ids in the response."""
    captured: dict = {}
    client = object.__new__(VLLMChatClient)
    client._async_client = _DummyAsyncClient(captured)

    our_tokens = [100, 200, 300]
    req = TokenCompletionRequest(
        model="mock",
        prompt_token_ids=our_tokens,
        sampling=SamplingParams(),
    )

    resp, _ = await client.complete_tokens(req)

    # The response should use our token IDs, not the server's
    assert resp.prompt_token_ids == our_tokens


@pytest.mark.asyncio
async def test_complete_tokens_rejects_unknown_extensions() -> None:
    """complete_tokens rejects unsupported backend extensions."""
    client = object.__new__(VLLMChatClient)

    req = TokenCompletionRequest(
        model="mock",
        prompt_token_ids=[1, 2, 3],
        sampling=SamplingParams(),
        extensions=DummyExtensions(),
    )

    with pytest.raises(TypeError, match="unsupported request\\.extensions"):
        await client.complete_tokens(req)


@pytest.mark.asyncio
async def test_complete_tokens_passes_vllm_extensions() -> None:
    """complete_tokens passes vLLM extensions correctly."""
    captured: dict = {}
    client = object.__new__(VLLMChatClient)
    client._async_client = _DummyAsyncClient(captured)

    req = TokenCompletionRequest(
        model="mock",
        prompt_token_ids=[1, 2, 3],
        sampling=SamplingParams(),
        extensions=VLLMExtensions(repetition_penalty=1.2, max_think=100),
    )

    await client.complete_tokens(req)

    assert captured["extra_body"]["repetition_penalty"] == 1.2
    assert captured["extra_body"]["vllm_xargs"]["max_think"] == 100


@pytest.mark.asyncio
async def test_complete_tokens_rejects_invalid_max_think() -> None:
    """complete_tokens rejects invalid max_think values."""
    client = object.__new__(VLLMChatClient)

    req = TokenCompletionRequest(
        model="mock",
        prompt_token_ids=[1, 2, 3],
        sampling=SamplingParams(),
        extensions=VLLMExtensions(max_think=-1),
    )

    with pytest.raises(ValueError, match="max_think must be a positive integer"):
        await client.complete_tokens(req)
