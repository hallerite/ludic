from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base import BackendExtensions


@dataclass(frozen=True)
class VLLMExtensions(BackendExtensions):
    """
    vLLM-only extensions / generation controls.

    Example: `max_think` is a custom extension that only works for a
    particular vLLM fork that includes a logits processor.

    Why not put these in `SamplingParams`?
    - Ludic's `SamplingParams` is intentionally aligned with OpenAI-style knobs
      (`temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, ...).
    - vLLM (and HF ecosystems) add extra sampling controls like
      `repetition_penalty` that are not standard OpenAI parameters.
    - Keeping those in `VLLMExtensions` prevents silently depending on
      backend-specific behavior when swapping ChatClient implementations.
    """

    max_think: Optional[int] = None
    # Neutral value is 1.0 ("no repetition penalty").
    repetition_penalty: float = 1.0
    extra_body_overrides: Dict[str, Any] = field(default_factory=dict)
    kind: str = "vllm"

