from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

log = logging.getLogger(__name__)

RESERVED_INFO_KEYS: set[str] = set()


def merge_step_info(
    *,
    client_info: Mapping[str, Any],
    env_info: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    reserved_keys: set[str] = RESERVED_INFO_KEYS,
) -> Dict[str, Any]:
    """
    Merge Step.info dictionaries with safety around "reserved" keys.

    Token traces live on Step.trace, so Step.info should stay JSON-only.

    Precedence:
      1) client_info seeds the dict
      2) env_info overwrites client_info for non-reserved keys
      3) extra overwrites for non-reserved keys

    If a reserved key collision occurs (and values differ), we keep the existing
    value and warn, to avoid silently corrupting training metadata.
    """
    merged: Dict[str, Any] = dict(client_info)

    def _merge_in(src: Optional[Mapping[str, Any]], *, src_name: str) -> None:
        if not src:
            return
        for k, v in src.items():
            if k in reserved_keys and k in merged and merged[k] != v:
                log.warning(
                    "Step.info collision on reserved key %r from %s; keeping client value.",
                    k,
                    src_name,
                )
                continue
            merged[k] = v

    _merge_in(env_info, src_name="env_info")
    _merge_in(extra, src_name="extra")
    return merged
