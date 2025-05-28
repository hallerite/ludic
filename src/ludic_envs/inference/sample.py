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


def _prompt_is_chat(prompt: Sequence[Dict[str, str]]) -> bool:
    """Heuristically decide whether a prompt is in chat format."""
    # The RolloutGenerator produces [{"system": ...}, {"user": ...}, ...]
    # which we treat as chat.  Treat as chat if every element has exactly one
    # key and that key is in the standard chat roles.
    CHAT_ROLES = {"system", "user", "assistant"}
    return all(len(m) == 1 and next(iter(m.keys())) in CHAT_ROLES for m in prompt)


def _flatten_to_text(prompt: Sequence[Dict[str, str]]) -> str:
    """Serialize a chat prompt to a *single* string (for non‑chat models)."""
    parts = []
    for message in prompt:
        role, content = next(iter(message.items()))
        parts.append(f"<{role}>: {content}")
    # Double‑newline between turns keeps things readable but token‑agnostic.
    return "\n\n".join(parts)


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
        A **running** `vllm.LLM` instance (GPU‑backed) from which to sample.
    prompts:
        One prompt *per environment*: each is a list of single‑key dicts, e.g.
        `[{"system": "…"}, {"user": "…"}]`.
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

    # Decide whether we can keep the chat structure or need to flatten.
    chat_mode = _prompt_is_chat(prompts[0])

    if chat_mode:
        # vLLM natively understands chat format → we can pass through directly
        _inputs: List[List[Dict[str, str]]] = prompts  # type: ignore[assignment]
    else:
        # Fallback for plain‑text models: concatenate into a single string
        _inputs = [_flatten_to_text(p) for p in prompts]  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Call the model *once* for the entire batch.  vLLM will dispatch to the
    # GPUs and stream results efficiently.
    # ------------------------------------------------------------------
    generations = model.generate(_inputs, sp, use_tqdm=False)

    # vLLM guarantees `len(generations) == len(prompts)`.
    replies_txt: List[str] = []
    replies_raw: List[Any] = []

    for gen in generations:
        # The `Generation` object may contain multiple candidates; trainers can
        # decide whether they want top‑1 or something else.  Here we default to
        # the first candidate for simplicity.
        if not gen.outputs:
            raise RuntimeError("vLLM returned zero candidates for a prompt.")

        replies_txt.append(gen.outputs[0].text)
        replies_raw.append(gen)

    return replies_txt, replies_raw
