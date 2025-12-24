from __future__ import annotations

import re
from typing import Optional
from dataclasses import dataclass


def extract_code_blocks(text: str) -> list[str]:
    """Extract code blocks from text (tagged or plain fenced)."""
    patterns = [
        re.compile(
            r"<code>\s*```(?:python)?\s*(.*?)\s*```\s*</code>",
            flags=re.DOTALL | re.IGNORECASE,
        ),
        re.compile(
            r"```(?:python)?\s*(.*?)\s*```",
            flags=re.DOTALL | re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        blocks = [match.group(1).strip() for match in pattern.finditer(text)]
        if blocks:
            return blocks
    return []


def extract_last_boxed_content(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{...} occurrence."""
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None

    def _parse_braced(start_brace_idx: int) -> Optional[str]:
        if start_brace_idx >= len(text) or text[start_brace_idx] != "{":
            return None

        depth = 0
        i = start_brace_idx
        while i < len(text):
            ch = text[i]
            prev = text[i - 1] if i > 0 else ""

            if ch == "{" and prev != "\\":
                depth += 1
            elif ch == "}" and prev != "\\":
                depth -= 1
                if depth == 0:
                    return text[start_brace_idx + 1 : i]
            i += 1
        return None

    for m in reversed(matches):
        inner = _parse_braced(m.end() - 1)
        if inner is not None:
            return inner.strip()
    return None


@dataclass(frozen=True)
class ReToolParseResult:
    """Result of parsing ReTool-style output."""

    has_code: bool
    code_block: Optional[str] = None
    final_answer: Optional[str] = None


class ReToolParser:
    """Parser for ReTool-style responses with code fences and \\boxed{} answers."""

    def parse(self, text: str) -> ReToolParseResult:
        """
        Parse ReTool-formatted text.

        Returns:
            ReToolParseResult indicating whether code blocks and/or final answer are present.
        """
        code_blocks = extract_code_blocks(text)
        final_answer = extract_last_boxed_content(text)

        # Prioritize code blocks if found (agent should continue)
        if code_blocks:
            return ReToolParseResult(
                has_code=True,
                code_block=code_blocks[0],  # Use first code block
                final_answer=final_answer,
            )

        # No code, check for final answer
        return ReToolParseResult(
            has_code=False,
            code_block=None,
            final_answer=final_answer,
        )
