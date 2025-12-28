from __future__ import annotations
from typing import Callable, List, Dict, Any, Optional, Tuple

from ludic.agents.tool_agent import ToolAgent
from ludic.agents.base_agent import AgentActResult, AgentActStep
from ludic.inference.request import InferenceSpec, ToolRequest
from ludic.inference.tool_parser import ToolParseResult
from ludic.parsers import ParseResult
from ludic.types import ChatResponse, TokenTrace

_DEFAULT_TOOL_PARSE_ERROR_OBS = "Invalid tool call format."
_DEFAULT_FINAL_ANSWER_PROMPT = (
    "You have exhausted your reasoning steps. "
    "You must output your final move now."
)


class ReActAgent(ToolAgent):
    """
    An agent that implements the ReAct pattern:
    [Think -> Tool Call] * n -> Act.

    It supports an execution loop where the model can call auxiliary tools
    multiple times before emitting a final answer for the environment.

    Tool Scopes:
      - tools: Executed by the agent (calculator, code interpreter).
        Results are added to context and the agent continues reasoning.
      - external_tools: Not executed by agent - returned to protocol.
        The protocol decides how to handle them (delegation, env, etc.).

    If the max_react_steps limit is reached, it forces a final generation
    attempt without tools to produce a valid environment action.
    """

    def __init__(
        self,
        *,
        tools: Optional[List[Callable]] = None,
        external_tools: Optional[List[Callable]] = None,
        max_react_steps: int = 5,
        tool_parse_error_penalty: float = -1.0,
        tool_parse_error_feedback: str = _DEFAULT_TOOL_PARSE_ERROR_OBS,
        final_answer_prompt: str = _DEFAULT_FINAL_ANSWER_PROMPT,
        **kwargs,
    ):
        """
        Args:
            tools: Tools executed by agent (results come back to agent).
            external_tools: Tools agent can call but doesn't execute.
                Returned to protocol which decides how to handle them.
            max_react_steps: Maximum number of internal think/tool loops.
            tool_parse_error_penalty: Reward penalty when tool call parsing fails.
            tool_parse_error_feedback: Feedback shown when tool call parsing fails.
            final_answer_prompt: Message injected when forcing a final answer
                after exhausting react steps.
            **kwargs: Passed to base Agent.
        """
        super().__init__(
            tools=tools,
            external_tools=external_tools,
            **kwargs,
        )
        self.max_react_steps = max_react_steps
        self._tool_parse_error_penalty = tool_parse_error_penalty
        self._tool_parse_error_feedback = tool_parse_error_feedback
        self._final_answer_prompt = final_answer_prompt

        # Safety check: Context must explicitly flag support for tools
        if not self._ctx.supports_tools:
            raise TypeError("ReActAgent requires a context with supports_tools=True. ")

    def _extract_content_and_tool_calls(
        self,
        resp: ChatResponse,
        info: Dict[str, Any],
    ) -> Tuple[Optional[str], ToolParseResult]:
        """
        Extract content and tool_calls from the response.

        Parses tool calls from raw text using the chat template.
        """
        content = resp.text
        tool_parse = self._chat_template.parse_tool_calls(content)
        return content, tool_parse

    async def act(
        self, 
        inference: InferenceSpec | None = None,
        sampling_seed: int | None = None,
        timeout_s: float | None = None
    ) -> AgentActResult:
        
        # 1. Setup inference config (tools enabled by default)
        inf = self._with_tools(inference)
        tools_req: ToolRequest = self._tool_request()
        steps: List[AgentActStep] = []
        
        # 2. ReAct Loop
        for step_i in range(self.max_react_steps):
            
            # --- THE SHOT CLOCK LOGIC ---
            # If we are on the last step, force a final answer.
            is_final_try = (step_i == self.max_react_steps - 1)
            
            messages = self._ctx.on_before_act()

            if is_final_try:
                # A. Force the agent to wrap up
                # We inject a temporary "system" instruction into the prompt
                # (Note: We don't save this to self._ctx, just for this one call)
                messages = messages + [{
                    "role": "user",
                    "content": self._final_answer_prompt,
                }]
                tools_req_this: ToolRequest | None = None
            else:
                tools_req_this = tools_req

            # 3. Inference (shared helper)
            resp, info, last_info, token_trace = await self._infer_once(
                messages=messages,
                inference=inf,
                sampling_seed=sampling_seed,
                tools=tools_req_this,
                timeout_s=timeout_s,
            )

            # Extract content/tool_calls (handles token-in mode via chat template)
            content, tool_parse = self._extract_content_and_tool_calls(resp, info)
            tool_calls = tool_parse.tool_calls
            raw_action = content or ""

            # Handle tool parse errors
            if tool_parse.parse_error:
                last_info["tool_parse_error"] = True
                self._ctx.add_assistant_step(content, None)
                parse_result = ParseResult(
                    action=None,
                    reward=self._tool_parse_error_penalty,
                    obs=self._tool_parse_error_feedback,
                )
                steps.append(
                    AgentActStep(
                        prompt_messages=messages,
                        action=raw_action,
                        parse_result=parse_result,
                        info=last_info,
                        trace=token_trace,
                        action_target="env",
                        loop_index=step_i,
                    )
                )
                return AgentActResult(steps=steps)

            # Handle incomplete completions
            if self._reject_incomplete and resp.finish_reason == "length":
                parse_result = ParseResult(
                    action=None,
                    reward=self._incomplete_penalty,
                    obs=self._incomplete_feedback,
                )
                last_info["incomplete_completion"] = True
            else:
                parse_result = self._parser(raw_action)

            # 4. Handle Final Panic Move
            if is_final_try:
                # We forced it to output text. Return whatever it gave us.
                final_text = raw_action
                
                # Update context so the agent remembers its final decision
                # We pass None for tool_calls because tools were disabled
                self._ctx.add_assistant_step(final_text, None)

                steps.append(
                    AgentActStep(
                        prompt_messages=messages,
                        action=final_text,
                        parse_result=parse_result,
                        info=last_info,
                        trace=token_trace,
                        action_target="env",
                        loop_index=step_i,
                    )
                )
                return AgentActResult(steps=steps)

            # 5. Normal Logic (Update Context & Check Tools)
            self._ctx.add_assistant_step(content, tool_calls)

            if tool_calls:
                # Check if any tool call targets an external tool
                if self.has_external_tool_call(tool_calls):
                    # --- EXTERNAL TOOL PHASE ---
                    # Don't execute locally - return to protocol
                    # Execute any internal tools first (hybrid calls)
                    tool_results = self._run_tool_calls(tool_calls, internal_only=True)

                    # External tool calls don't have parse_result - the protocol
                    # will handle them and return a result, then agent continues.
                    steps.append(
                        AgentActStep(
                            prompt_messages=messages,
                            action=raw_action,
                            parse_result=None,  # Not a final action
                            info=last_info,
                            trace=token_trace,
                            action_target="external",
                            loop_index=step_i,
                            tool_calls=tool_calls,
                            tool_results=tool_results if tool_results else None,
                        )
                    )
                    return AgentActResult(steps=steps)

                # --- INTERNAL TOOL PHASE ---
                tool_results = self._run_tool_calls(tool_calls)
                steps.append(
                    AgentActStep(
                        prompt_messages=messages,
                        action=raw_action,
                        parse_result=parse_result,
                        info=last_info,
                        trace=token_trace,
                        action_target="internal",
                        loop_index=step_i,
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                    )
                )
                # Continue loop -> Model sees result and thinks again
                continue

            # --- FINAL ANSWER PHASE ---
            # No tool calls means the model is talking to the user/env.
            # We expect the content to contain the action here.
            # The format is defined by self._parser (XML, JSON, Regex, etc).
            final_text = raw_action
            steps.append(
                AgentActStep(
                    prompt_messages=messages,
                    action=final_text,
                    parse_result=parse_result,
                    info=last_info,
                    trace=token_trace,
                    action_target="env",
                    loop_index=step_i,
                )
            )
            return AgentActResult(steps=steps)

        raise RuntimeError("ReActAgent loop exited unexpectedly.")
