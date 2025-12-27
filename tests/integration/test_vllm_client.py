from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from ludic.inference import HFChatTemplate
from ludic.inference.vllm_client import VLLMChatClient
from ludic.inference.request import TokenCompletionRequest, ReturnSpec
from ludic.inference.extensions import VLLMExtensions
from ludic.inference.sampling import SamplingParams

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def build_prompt_token_ids(model_name: str, messages: list[dict]) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    template = HFChatTemplate(tokenizer)
    return template.apply(messages, add_generation_prompt=True).prompt_token_ids


@pytest.mark.asyncio
async def test_vllm_client_completion_roundtrip(
    vllm_client: VLLMChatClient,
    vllm_model_name: str,
) -> None:
    """
    Basic completion sanity check against the live vLLM server.
    """
    sampling = SamplingParams()

    messages = [
        {
            "role": "system",
            "content": "You are a test assistant. Answer very briefly.",
        },
        {
            "role": "user",
            "content": "Reply with the single word 'pong'.",
        },
    ]

    prompt_token_ids = build_prompt_token_ids(vllm_model_name, messages)
    resp, info = await vllm_client.complete_tokens(
        TokenCompletionRequest(
            model=vllm_model_name,
            prompt_token_ids=prompt_token_ids,
            sampling=sampling,
            return_=ReturnSpec.for_eval(return_token_ids=True),
        )
    )

    assert isinstance(resp.text, str)
    assert resp.text.strip() != ""

    used_args = info.get("used_args", {})
    assert used_args.get("temperature") == sampling.temperature
    assert used_args.get("max_tokens") == sampling.max_tokens

    assert "raw_response" in info
    assert isinstance(info["raw_response"], dict)


@pytest.mark.asyncio
async def test_vllm_client_same_seed_is_deterministic(
    vllm_client: VLLMChatClient,
    vllm_model_name: str,
) -> None:
    """
    With a fixed seed and deterministic sampling settings, we expect
    fully deterministic outputs: same text, same token IDs, same
    prompt token IDs.
    """
    seed_value = 1234

    sampling = SamplingParams(temperature=0.0, max_tokens=256)

    messages = [
        {
            "role": "system",
            "content": "You are deterministic for a fixed seed.",
        },
        {
            "role": "user",
            "content": "Say a short random-looking sentence.",
        },
    ]

    prompt_token_ids = build_prompt_token_ids(vllm_model_name, messages)
    results = []
    for _ in range(3):
        resp, _ = await vllm_client.complete_tokens(
            TokenCompletionRequest(
                model=vllm_model_name,
                prompt_token_ids=prompt_token_ids,
                sampling=sampling,
                seed=seed_value,
                return_=ReturnSpec.for_eval(return_token_ids=True),
            )
        )
        assert resp.text.strip() != ""
        results.append(resp)

    r0 = results[0]
    for r in results[1:]:
        assert r.text == r0.text
        assert r.completion_token_ids == r0.completion_token_ids
        assert r.prompt_token_ids == r0.prompt_token_ids


@pytest.mark.asyncio
async def test_vllm_client_returns_token_ids_and_detok(
    vllm_client: VLLMChatClient,
    vllm_model_name: str,
) -> None:
    """
    Ensure return_token_ids=True yields token IDs + prompt token IDs.
    """
    sampling = SamplingParams()

    messages = [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Say 'hello'."},
    ]

    prompt_token_ids = build_prompt_token_ids(vllm_model_name, messages)
    resp, _ = await vllm_client.complete_tokens(
        TokenCompletionRequest(
            model=vllm_model_name,
            prompt_token_ids=prompt_token_ids,
            sampling=sampling,
            return_=ReturnSpec.for_eval(return_token_ids=True),
        )
    )

    assert resp.text.strip() != ""
    assert resp.completion_token_ids is not None
    assert resp.prompt_token_ids is not None


@pytest.mark.asyncio
async def test_vllm_global_think_processor_triggers_at_very_small_max_think(
    vllm_client: VLLMChatClient,
    vllm_model_name: str,
) -> None:
    """
    Stronger sanity check: with interrupt_thinking=1, the logits processor
    should immediately start forcing the '</think>' sequence, so the closing
    tag should appear right at or very near the beginning of the completion.
    """
    interrupt_thinking = 1  # trigger almost immediately

    sampling = SamplingParams()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a chain-of-thought model. Always wrap your reasoning "
                "between <think> and </think>, then give the final answer."
            ),
        },
        {
            "role": "user",
            "content": "Explain briefly why 2 + 2 = 4, then answer.",
        },
    ]

    prompt_token_ids = build_prompt_token_ids(vllm_model_name, messages)
    resp, _ = await vllm_client.complete_tokens(
        TokenCompletionRequest(
            model=vllm_model_name,
            prompt_token_ids=prompt_token_ids,
            sampling=sampling,
            seed=123,
            return_=ReturnSpec.for_eval(return_token_ids=True),
            extensions=VLLMExtensions(max_think=interrupt_thinking),
        )
    )

    assert resp.text.strip() != ""
    assert resp.completion_token_ids is not None
    completion_ids = resp.completion_token_ids

    tokenizer = AutoTokenizer.from_pretrained(
        vllm_model_name,
        trust_remote_code=True,
    )
    closing_ids = tokenizer.encode("</think>", add_special_tokens=False)
    assert closing_ids, "Expected non-empty tokenization for '</think>'."

    def find_subseq(haystack: list[int], needle: list[int]) -> int:
        if not needle:
            return -1
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i : i + len(needle)] == needle:
                return i
        return -1

    closing_idx = find_subseq(completion_ids, closing_ids)
    assert closing_idx != -1, "Expected '</think>' token sequence in completion."

    # With max_think=1 we expect the processor to fire essentially immediately.
    # Allow a tiny tolerance, but it should be at or right after the first token.
    assert closing_idx <= 2, (
        f"Expected '</think>' to appear very early (<=2), got index {closing_idx}."
    )
