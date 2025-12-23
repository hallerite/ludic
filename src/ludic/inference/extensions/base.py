from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class BackendExtensions:
    """
    Backend-specific request knobs.

    This is the typed, backend-agnostic escape hatch for non-portable controls.
    Each backend can define its own subclass (e.g. VLLMExtensions) without the
    core request schema mentioning any specific backend by name.
    """

    kind: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

