from __future__ import annotations

"""Utility for batched sampling with vLLM.

This helper is intentionally *stateless* so it can be imported by both
`RolloutGenerator` *and* trainers such as `GRPOTrainer`.  It converts a list of
chat‑style prompts into a form vLLM understands, calls `model.generate`, and
returns two flat lists:
    1) `replies_txt` – the assistant replies as plain strings (len == n_envs)
    2) `replies_raw` – the raw vLLM `Generation` objects (or any metadata the
       caller might need downstream).

The function makes **no** assumptions about the model beyond it exposing a
`generate` method compatible with `vllm.LLM`.  It also works for *both* "chat"
and plain‑text models.
"""

from typing import List, Dict, Any, Tuple, Sequence

try:
    from vllm import LLM, SamplingParams  # type: ignore
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "vLLM is required for the `sample` utility. Install with\n"
        "    pip install vllm\n"  # noqa: E501
    ) from e

__all__ = ["sample"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_sampling_params(sampling_params: Any) -> "SamplingParams":
    """Return a `SamplingParams` instance no matter what the caller passes."""
    if isinstance(sampling_params, SamplingParams):
        return sampling_params
    if isinstance(sampling_params, dict):
        return SamplingParams(**sampling_params)
    raise TypeError(
        "`sampling_params` must be a `vllm.SamplingParams` instance or a dict, "
        f"got {type(sampling_params).__name__}."
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample(
    model: "LLM",
    prompts: List[List[Dict[str, str]]],
    sampling_params: Any,
) -> Tuple[List[str], List[Any]]:
    """Generate assistant replies for *each* prompt using vLLM.

    Parameters
    ----------
    model:
        A **running** `vllm.LLM` instance (GPU-backed) from which to sample.
    prompts:
        A list of chat-style prompts: each is a list of messages, where each
        message is a dict with `{"role": ..., "content": ...}`. Example:
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What's 2+2?"}
            ]
    sampling_params:
        Either a `vllm.SamplingParams` instance **or** a plain dict of kwargs
        forwarded to the constructor.

    Returns
    -------
    replies_txt:
        List of assistant replies as strings.
    replies_raw:
        The raw `Generation` objects from vLLM (or whatever metadata future
        trainers might need).
    """

    sp = _ensure_sampling_params(sampling_params)

    # Validate prompt structure
    for i, prompt in enumerate(prompts):
        for msg in prompt:
            if not (isinstance(msg, dict) and set(msg) == {"role", "content"}):
                raise TypeError(
                    f"Prompt {i} contains invalid message format: {msg}. "
                    "Expected: {'role': ..., 'content': ...}"
                )

    generations = model.chat(prompts, sampling_params=sp, use_tqdm=False)

    replies_txt: List[str] = []
    replies_raw: List[Any] = []

    for gen in generations:
        if not gen.outputs:
            raise RuntimeError("vLLM returned zero outputs for a prompt.")
        replies_txt.append(gen.outputs[0].text)
        replies_raw.append(gen)

    return replies_txt, replies_raw
