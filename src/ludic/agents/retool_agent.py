from __future__ import annotations

from typing import Optional, Callable, Awaitable
from dataclasses import dataclass

from ludic.agents.base_agent import Agent, AgentActResult, AgentActStep
from ludic.inference.request import InferenceSpec
from ludic.parsers import ParseResult

from ludic.agents.retool_parser import ReToolParser


@dataclass(frozen=True)
class CodeExecutionResult:
    """Result of executing code in a sandbox."""

    output: str
    success: bool
    error: Optional[str] = None


CodeSandbox = Callable[[str], Awaitable[CodeExecutionResult]]


class ReToolAgent(Agent):
    """
    Agent for ReTool-style interleaved code execution.

    The model generates free-form text with embedded fenced Python code blocks
    (optionally wrapped in <code> tags) and \\boxed{...} final answers. Code blocks
    are executed in an async sandbox and results are injected back into the
    conversation as <interpreter>...</interpreter> messages.

    This is "in-band" tool calling - the inference backend (vLLM) sees only standard
    chat messages, with no special API for tool invocation.
    """

    def __init__(
        self, code_sandbox: CodeSandbox, max_code_blocks: int = 10, **kwargs
    ) -> None:
        """
        Args:
            code_sandbox: Async callable that executes Python code and returns result.
            max_code_blocks: Maximum number of code blocks to execute per episode.
            **kwargs: Passed to base Agent.
        """
        super().__init__(**kwargs)
        self.code_sandbox = code_sandbox
        self.max_code_blocks = max_code_blocks
        self._retool_parser = ReToolParser()

    async def act(
        self,
        inference: Optional[InferenceSpec] = None,
        sampling_seed: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> AgentActResult:
        """
        Run interleaved generation with code execution.

        Returns:
            AgentActResult with one or more AgentActSteps:
            - Internal steps: action_target="internal" (text + code blocks)
            - Final step: action_target="env" (contains \\boxed{} answer)
        """
        steps: list[AgentActStep] = []
        code_count = 0

        while True:
            # Check if we're on final try
            is_final_try = code_count >= self.max_code_blocks

            # 1. Prepare prompt from current context
            messages = self._ctx.on_before_act()

            # If final try, force the model to give a final answer
            if is_final_try:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "You have used your maximum number of code blocks. "
                            "You must now provide your final answer in \\boxed{...} format."
                        ),
                    }
                ]

            # 2. Inference (generate)
            resp, public_info, last_info, token_trace = await self._infer_once(
                messages=messages,
                inference=inference,
                sampling_seed=sampling_seed,
                timeout_s=timeout_s,
            )

            raw_action = resp.text

            # 3. Check for incomplete completion
            if self._reject_incomplete and resp.finish_reason == "length":
                parse_result = ParseResult(
                    action=None,
                    reward=self._incomplete_penalty,
                    obs=self._incomplete_feedback,
                )
                last_info["incomplete_completion"] = True

                # Treat as final step but failed
                self._ctx.on_after_act(resp)
                steps.append(
                    AgentActStep(
                        prompt_messages=messages,
                        action=raw_action,
                        parse_result=parse_result,
                        info=last_info,
                        trace=token_trace,
                        action_target="env",
                        loop_index=len(steps),
                    )
                )
                return AgentActResult(steps=steps)

            # 4. Parse ReTool format
            retool_result = self._retool_parser.parse(raw_action)

            # 5. Handle code blocks (skip if final try)
            if not is_final_try and retool_result.has_code and retool_result.code_block:
                code_result = await self._execute_code(retool_result.code_block)

                # Inject interpreter feedback into context
                interpreter_msg = self._format_interpreter_message(code_result)
                self._ctx.on_after_act(resp)
                self._ctx._messages.append({"role": "user", "content": interpreter_msg})

                # Record as internal step
                parse_result = ParseResult(action=None, reward=0.0, obs=None)
                last_info["code_block"] = retool_result.code_block
                last_info["interpreter_output"] = code_result.output
                last_info["code_success"] = code_result.success
                if not code_result.success:
                    last_info["code_error"] = code_result.error

                steps.append(
                    AgentActStep(
                        prompt_messages=messages,
                        action=raw_action,
                        parse_result=parse_result,
                        info=last_info,
                        trace=token_trace,
                        action_target="internal",
                        loop_index=len(steps),
                    )
                )
                code_count += 1
                continue

            # 6. Handle final answer (or parse failure)
            parse_result = self._parser(raw_action)
            if parse_result.action is None:
                last_info["parse_error"] = True

            # Update context with final response
            self._ctx.on_after_act(resp)

            steps.append(
                AgentActStep(
                    prompt_messages=messages,
                    action=raw_action,
                    parse_result=parse_result,
                    info=last_info,
                    trace=token_trace,
                    action_target="env",
                    loop_index=len(steps),
                )
            )
            return AgentActResult(steps=steps)

    async def _execute_code(self, code: str) -> CodeExecutionResult:
        """Execute code in sandbox and handle errors."""
        try:
            return await self.code_sandbox(code)
        except Exception as e:
            return CodeExecutionResult(
                output="",
                success=False,
                error=str(e),
            )

    def _format_interpreter_message(self, result: CodeExecutionResult) -> str:
        """Format sandbox result as <interpreter>...</interpreter> message."""
        output = result.output if result.success else f"Error: {result.error}"
        return f"<interpreter>{output}</interpreter>"
