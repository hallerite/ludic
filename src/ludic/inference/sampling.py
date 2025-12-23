from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass(frozen=True)
class SamplingParams:
    """
    Portable sampling parameters (no vendor-specific knobs).

    This is intentionally *not* a catch-all. vLLM/OpenAI extensions belong in
    backend-specific request objects (see `ludic.inference.request`).
    """

    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None

    def to_openai_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
            "top_p": float(self.top_p),
            "frequency_penalty": float(self.frequency_penalty),
            "presence_penalty": float(self.presence_penalty),
        }
        if self.stop:
            kwargs["stop"] = self.stop
        return kwargs
