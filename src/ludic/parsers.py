from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------
# ParseResult and semantic parser API
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ParseResult:
    """
    Result of a semantic parser.

    - action: parsed/cleaned action, or None if parsing fails
    - reward: parser-level reward (penalty for format errors, etc.)
    - obs: optional synthetic observation the agent receives on failure
    """
    action: Optional[str]
    reward: float
    obs: Optional[str]


Parser = Callable[[str], ParseResult]


def compose_parsers(*parsers: Parser) -> Parser:
    """
    Chain multiple Parser functions left-to-right.

    If any parser fails (action=None), return that failure with
    accumulated reward.
    """
    def _p(raw: str) -> ParseResult:
        current = ParseResult(action=raw, reward=0.0, obs=None)

        for parser in parsers:
            result = parser(current.action)  # type: ignore[arg-type]
            if result.action is None:
                return ParseResult(
                    action=None,
                    reward=current.reward + result.reward,
                    obs=result.obs,
                )
            # success: accumulate reward
            current = ParseResult(
                action=result.action,
                reward=current.reward + result.reward,
                obs=None,
            )

        return current
    return _p


# ---------------------------------------------------------------------
# XML parser factory
# ---------------------------------------------------------------------

def xml_parser(
    tag: str,
    *,
    exact: bool = False,
    kind: str = "inner",
    success_reward: float = 0.1,
    error_reward: float = -1.0,
) -> Parser:
    """
    Create a Parser based on an XML-ish tag contract.

    Args:
        tag: Tag name (e.g. "move" for <move>...</move>).
        exact: If True, require the entire output to be exactly one tag
            (aside from surrounding whitespace). If False, succeed if the tag
            appears anywhere in the text.
        kind:
            - "inner": return the inner text inside <tag>...</tag>
            - "remainder_after_prefix": require the output start with <tag>...</tag>
              and return the remainder after the closing tag (must be non-empty).
    """
    tag_re = re.escape(tag)

    if kind == "inner":
        if exact:
            pattern = re.compile(
                rf"^\s*<{tag_re}>(.*?)</{tag_re}>\s*$",
                flags=re.DOTALL | re.IGNORECASE,
            )
            expectation = f"Expected output to be exactly <{tag}>...</{tag}> (and nothing else)."
        else:
            pattern = re.compile(
                rf"<{tag_re}>(.*?)</{tag_re}>",
                flags=re.DOTALL | re.IGNORECASE,
            )
            expectation = f"Expected <{tag}>...</{tag}>."

        def _p(raw: str) -> ParseResult:
            try:
                m = pattern.search(raw)
                if not m:
                    raise ValueError(expectation)

                inner = m.group(1).strip()
                if not inner:
                    raise ValueError(f"Empty <{tag}> tag.")

                return ParseResult(action=inner, reward=success_reward, obs=None)

            except Exception as e:
                return ParseResult(
                    action=None,
                    reward=error_reward,
                    obs=f"Invalid action format: {e}",
                )

        return _p

    if kind == "remainder_after_prefix":
        if exact:
            raise ValueError("exact=True is not supported for kind='remainder_after_prefix'")

        pattern = re.compile(
            rf"^\s*<{tag_re}>(.*?)</{tag_re}>\s*(.+)$",
            flags=re.DOTALL | re.IGNORECASE,
        )
        expectation = f"Expected '<{tag}>...</{tag}>' prefix followed by content."

        def _p(raw: str) -> ParseResult:
            try:
                m = pattern.match(raw)
                if not m:
                    raise ValueError(expectation)

                remainder = m.group(2).strip()
                if not remainder:
                    raise ValueError(f"Missing content after </{tag}>.")

                return ParseResult(action=remainder, reward=success_reward, obs=None)

            except Exception as e:
                return ParseResult(
                    action=None,
                    reward=error_reward,
                    obs=f"Invalid action format: {e}",
                )

        return _p

    raise ValueError(f"Unknown kind={kind!r}")


def xml_tag_parser(
    tag: str,
    *,
    exact: bool = False,
    success_reward: float = 0.1,
    error_reward: float = -1.0,
) -> Parser:
    """
    Create a Parser that extracts the inner text inside <tag>...</tag>.

    Notes:
        If exact=False, this succeeds if the tag appears anywhere in the text
        (e.g. "The answer is <move>A1</move>"). If exact=True, the output must
        be exactly one tag (aside from surrounding whitespace).
    """
    return xml_parser(
        tag,
        exact=exact,
        kind="inner",
        success_reward=success_reward,
        error_reward=error_reward,
    )


def think_prefix_parser(
    raw: str,
    *,
    success_reward: float = 0.1,
    error_reward: float = -1.0,
) -> ParseResult:
    """
    STRICT <think>...</think> prefix parser.

    Required:
        <think> ... </think> ANSWER

    Output:
        action = ANSWER
    """
    return xml_parser(
        "think",
        kind="remainder_after_prefix",
        success_reward=success_reward,
        error_reward=error_reward,
    )(raw)


# Backwards-friendly alias for readability in older examples/tests.
cot_prefix_parser = think_prefix_parser


# ---------------------------------------------------------------------
# Strict \boxed{...} answer parser
# ---------------------------------------------------------------------

def extract_last_boxed_content(raw: str) -> Optional[str]:
    """
    Extract the content of the last LaTeX \\boxed{...} occurrence.

    Supports nested braces inside the boxed content (e.g. \\boxed{\\frac{1}{2}}).
    Returns None if no well-formed \\boxed{...} is found.
    """
    matches = list(re.finditer(r"\\boxed\s*\{", raw))
    if not matches:
        return None

    def _parse_braced(start_brace_idx: int) -> Optional[str]:
        if start_brace_idx >= len(raw) or raw[start_brace_idx] != "{":
            return None

        depth = 0
        i = start_brace_idx
        while i < len(raw):
            ch = raw[i]
            prev = raw[i - 1] if i > 0 else ""

            if ch == "{" and prev != "\\":
                depth += 1
            elif ch == "}" and prev != "\\":
                depth -= 1
                if depth == 0:
                    return raw[start_brace_idx + 1 : i]
            i += 1
        return None

    # Prefer the last occurrence (the model may include intermediate boxes).
    for m in reversed(matches):
        inner = _parse_braced(m.end() - 1)
        if inner is not None:
            return inner
    return None


def boxed_parser(
    raw: str,
    *,
    success_reward: float = 0.1,
    error_reward: float = -1.0,
) -> ParseResult:
    """
    STRICT parser that requires the final answer to appear inside \\boxed{...}.

    Rewards:
        Defaults to +0.1 on success and -1.0 on failure; override via keyword
        args or functools.partial for custom parser instances.
    """
    try:
        inner = extract_last_boxed_content(raw)
        if inner is None:
            raise ValueError("Expected \\boxed{...} with the final answer.")

        inner = inner.strip()
        if not inner:
            raise ValueError("Empty \\boxed{} content.")

        # Positive intrinsic reward for good formatting
        return ParseResult(action=inner, reward=success_reward, obs=None)

    except Exception as e:
        return ParseResult(
            action=None,
            reward=error_reward,
            obs=f"Invalid boxed answer: {e}",
        )
