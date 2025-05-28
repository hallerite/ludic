import os

import pytest

# ---------------------------------------------------------------------------
# Optional integration test: real vLLM + Qwen2 0.5B on GPU
# ---------------------------------------------------------------------------
# This test is **skipped** unless two conditions hold:
#   1) CUDA is available, *and*
#   2) The environment variable `RUN_REAL_VLLM_TESTS=1` is set.
#
# Why the environment variable?  Loading a model weights (~1‑2 GB) from the
# Hugging Face Hub is slow and not appropriate for lightweight CI by default.
# You can enable locally via:
#
#     RUN_REAL_VLLM_TESTS=1 pytest -q tests/test_sample_gpu.py
# ---------------------------------------------------------------------------

CUDA_AVAILABLE = False
try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    pass

RUN_REAL_TESTS = os.environ.get("RUN_REAL_VLLM_TESTS") == "1"

requires_gpu = pytest.mark.skipif(
    not (CUDA_AVAILABLE and RUN_REAL_TESTS),
    reason="CUDA not available or RUN_REAL_VLLM_TESTS not set",
)

# ---------------------------------------------------------------------------
# Actual test
# ---------------------------------------------------------------------------


@requires_gpu  # type: ignore[misc]
def test_sampling_with_qwen():
    """Sanity‑check our `sample` helper against a **real** tiny model."""

    from vllm import LLM, SamplingParams  # Local import to avoid CI overhead

    from ludic_envs.inference.sample import sample

    # ---- Load model -----------------------------------------------------
    #   * Qwen2‑0.5B is HF‑model‑id: "Qwen/Qwen2-0.5B" (0.5‑billion params)
    #   * The test purposely uses `enforce_eager=True` so we don't need CUDA
    #     graph capture and the model fits on modest GPUs (4‑6 GB VRAM).
    # --------------------------------------------------------------------
    model_id = "Qwen/Qwen2-0.5B"

    model = LLM(
        model=model_id,
        dtype="float16",  # Keeps memory under control
        #enforce_eager=True,
        gpu_memory_utilization=0.80,  # Leave headroom for test harness
    )

    # Compose a minimal chat prompt in RolloutGenerator format
    prompts = [
        [{"system": "You are a concise assistant."}, {"user": "2+2?"}],
        [{"system": "You are a bot."}, {"user": "Capital of France?"}],
    ]

    sampling_params = SamplingParams(max_tokens=4, temperature=0)

    replies_txt, replies_raw = sample(model, prompts, sampling_params)

    assert len(replies_txt) == 2
    assert all(isinstance(t, str) for t in replies_txt)

    # Basic sanity: expect the first answer to contain "4" and second "Paris"
    joined = "\n".join(replies_txt).lower()
    assert "4" in joined and "paris" in joined
