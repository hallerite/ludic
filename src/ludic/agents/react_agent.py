from __future__ import annotations
from typing import Callable, List, Tuple, Dict, Any, Optional

from ludic.agents.tool_agent import ToolAgent
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

    If the max_react_steps limit is reached, it forces a final generation
    attempt without tools to produce a valid environment action.
    """

    def __init__(
        self,
        tools: List[Callable],
        max_react_steps: int = 5,
        tool_parse_error_penalty: float = -1.0,
        tool_parse_error_feedback: str = _DEFAULT_TOOL_PARSE_ERROR_OBS,
        final_answer_prompt: str = _DEFAULT_FINAL_ANSWER_PROMPT,
        **kwargs,
    ):
        """
        Args:
            tools: List of python functions the agent can call.
            max_react_steps: Maximum number of internal think/tool loops.
            tool_parse_error_penalty: Reward penalty when tool call parsing fails.
            tool_parse_error_feedback: Feedback shown when tool call parsing fails.
            final_answer_prompt: Message injected when forcing a final answer
                after exhausting react steps.
            **kwargs: Passed to base Agent.
        """
        super().__init__(tools=tools, **kwargs)
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
    ) -> Tuple[ParseResult, str, Dict[str, Any], Optional[TokenTrace]]:
        
        # 1. Setup inference config (tools enabled by default)
        inf = self._with_tools(inference)
        tools_req: ToolRequest = self._tool_request()
        last_info: Dict[str, Any] = {}
        last_trace: Optional[TokenTrace] = None
        
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
            last_trace = token_trace

            # Extract content/tool_calls (handles both token-in and standard modes)
            content, tool_parse = self._extract_content_and_tool_calls(resp, info)
            tool_calls = tool_parse.tool_calls
            if tool_parse.parse_error:
                last_info["tool_parse_error"] = True
                self._ctx.add_assistant_step(content, None)
                return (
                    ParseResult(
                        action=None,
                        reward=self._tool_parse_error_penalty,
                        obs=self._tool_parse_error_feedback,
                    ),
                    content or "",
                    last_info,
                    last_trace,
                )

            # 4. Handle Final Panic Move
            if is_final_try:
                # We forced it to output text. Return whatever it gave us.
                final_text = content or ""
                parse_result = self._parser(final_text)
                
                # Update context so the agent remembers its final decision
                # We pass None for tool_calls because tools were disabled
                self._ctx.add_assistant_step(final_text, None)
                
                return parse_result, final_text, last_info, last_trace

            # 5. Normal Logic (Update Context & Check Tools)
            self._ctx.add_assistant_step(content, tool_calls)

            if tool_calls:
                # --- EXECUTION PHASE ---
                self._run_tool_calls(tool_calls)
                
                # Continue loop -> Model sees result and thinks again
                continue
            
            else:
                # --- FINAL ANSWER PHASE ---
                # No tool calls means the model is talking to the user/env.
                # We expect the content to contain the action here.
                # The format is defined by self._parser (XML, JSON, Regex, etc).
                final_text = content or ""
                parse_result = self._parser(final_text)
                return parse_result, final_text, last_info, last_trace

        # This should technically be unreachable due to the is_final_try block,
        # but good for safety.
        fallback_msg = "Error: Loop logic failure."
        return self._parser(fallback_msg), fallback_msg, last_info, last_trace
