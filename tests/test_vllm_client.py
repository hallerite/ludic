import os
import pytest

RUN_REAL_TESTS = os.environ.get("RUN_REAL_VLLM_TESTS") == "1"

requires_vllm_server = pytest.mark.skipif(
    not RUN_REAL_TESTS,
    reason="RUN_REAL_VLLM_TESTS not set"
)

#@requires_vllm_server  # type: ignore[misc]
def test_vllm_client_chat():
    """Check VLLMClient chat with live vLLM server — no CUDA required."""
    from ludic_envs.inference.vllm_client import VLLMClient
    #from trl.extras.vllm_client import VLLMClient
    client = VLLMClient()
    breakpoint()
    response = client.chat(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
    )

    assert "choices" in response
    reply = response["choices"][0]["message"]["content"].lower()
    assert "4" in reply
