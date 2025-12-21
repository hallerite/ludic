from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import tempfile
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from ludic.agents.base_agent import Agent
from ludic.inference.request import ChatCompletionRequest, InferenceSpec
from ludic.parsers import ParseResult
from ludic.types import Message, TokenTrace

logger = logging.getLogger(__name__)

DEFAULT_RLM_SYSTEM_PROMPT = (
    "You are a recursive language model. The long context is stored in a "
    "python variable named `ctx` inside a REPL. To inspect it, output a "
    "python code block. You will receive the stdout. When ready, respond "
    "with FINAL(<answer>) or FINAL_VAR(<var_name>) to use a string variable "
    "from the REPL. You can also call RLM_CALL(<text>) for a submodel answer."
)

_FINAL_VAR_RE = re.compile(r"FINAL_VAR\((.*?)\)", re.DOTALL)
_FINAL_RE = re.compile(r"FINAL\((.*?)\)", re.DOTALL)
_RLM_CALL_RE = re.compile(r"RLM_CALL\((.*?)\)", re.DOTALL)
_CODE_BLOCK_RE = re.compile(
    r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE
)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _extract_first(pattern: re.Pattern[str], text: str) -> Optional[str]:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _maybe_unquote(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return raw
    if raw[0] in ("'", '"') and raw[-1] == raw[0]:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw.strip(raw[0])
    return raw


class _PythonReplay:
    def __init__(
        self,
        *,
        python_cmd: str,
        timeout_s: float,
        max_output_chars: int,
        prelude: Optional[str] = None,
    ) -> None:
        self._python_cmd = python_cmd
        self._timeout_s = timeout_s
        self._max_output_chars = max_output_chars
        self._prelude = prelude or ""
        self._cells: List[str] = []
        self._ctx_path: Optional[str] = None

    def reset(self, context: str) -> None:
        self._cells = []
        self._set_context(context)

    def clear(self) -> None:
        self._cells = []

    def _set_context(self, context: str) -> None:
        if self._ctx_path:
            try:
                os.unlink(self._ctx_path)
            except OSError:
                pass
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="ludic_rlm_ctx_", delete=False
        )
        tmp.write(context)
        tmp.flush()
        tmp.close()
        self._ctx_path = tmp.name

    def run(self, code: str) -> str:
        prev_cells = list(self._cells)
        self._cells.append(code)
        return self._exec(prev_cells, code)

    def eval_var(self, name: str) -> str:
        return self._exec(self._cells, f"print(str({name}))")

    def _exec(self, prev_cells: List[str], code: str) -> str:
        if not self._ctx_path:
            return "Error: Missing context file."

        prev_json = json.dumps(prev_cells, ensure_ascii=True)
        code_json = json.dumps(code, ensure_ascii=True)
        prelude = self._prelude

        script = textwrap.dedent(
            f"""
            import contextlib
            import io
            import json
            import traceback

            ctx_path = {self._ctx_path!r}
            with open(ctx_path, "r", encoding="utf-8") as f:
                ctx = f.read()

            globals_dict = {{"ctx": ctx}}
            prelude = {prelude!r}
            if prelude:
                exec(prelude, globals_dict)

            prev_cells = json.loads({prev_json!r})
            current_cell = json.loads({code_json!r})

            def _run(cell: str) -> None:
                exec(cell, globals_dict)

            for cell in prev_cells:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        _run(cell)
                    except Exception:
                        pass

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                try:
                    _run(current_cell)
                except Exception:
                    traceback.print_exc()

            out = stdout_buf.getvalue()
            err = stderr_buf.getvalue()
            if err:
                out = out + "\\n[stderr]\\n" + err
            print(out, end="")
            """
        )

        try:
            result = subprocess.run(
                [self._python_cmd, "-"],
                input=script,
                text=True,
                capture_output=True,
                timeout=self._timeout_s,
                check=False,
            )
        except FileNotFoundError:
            return f"Error: python executable not found: {self._python_cmd}"
        except subprocess.TimeoutExpired:
            return f"Error: python exec timed out after {self._timeout_s}s."

        output = result.stdout or ""
        if result.stderr:
            output = output + "\n[stderr]\n" + result.stderr

        output = output.rstrip()
        if not output:
            output = "(no output)"

        if len(output) > self._max_output_chars:
            output = output[: self._max_output_chars] + "...(truncated)"
        return output


class RlmAgent(Agent):
    """
    RLM-style agent that can iteratively execute python and submodel calls
    before emitting a final action.
    """

    def __init__(
        self,
        *,
        client,
        model: str,
        ctx,
        parser,
        max_steps: int = 8,
        context_key: str = "context",
        allow_context_fallback: bool = True,
        python_cmd: str = "python",
        python_timeout_s: float = 5.0,
        max_output_chars: int = 4000,
        repl_prelude: Optional[str] = None,
        sub_model: Optional[str] = None,
        sub_system_prompt: Optional[str] = None,
        sub_inference: Optional[InferenceSpec] = None,
        exhausted_penalty: float = -1.0,
        exhausted_feedback: str = "RLM exceeded max steps without FINAL(...).",
        **kwargs,
    ) -> None:
        super().__init__(
            client=client,
            model=model,
            ctx=ctx,
            parser=parser,
            **kwargs,
        )
        self.max_steps = max_steps
        self.context_key = context_key
        self.allow_context_fallback = allow_context_fallback
        self._sub_model = sub_model or model
        self._sub_system_prompt = sub_system_prompt
        self._sub_inference = sub_inference or InferenceSpec()
        self._exhausted_penalty = exhausted_penalty
        self._exhausted_feedback = exhausted_feedback
        self._tool_counter = 0

        if not self._ctx.supports_tools:
            raise TypeError("RlmAgent requires a context with supports_tools=True.")

        self._repl = _PythonReplay(
            python_cmd=python_cmd,
            timeout_s=python_timeout_s,
            max_output_chars=max_output_chars,
            prelude=repl_prelude,
        )

    def reset(self, system_prompt: Optional[str] = None) -> None:
        super().reset(system_prompt=system_prompt)
        self._repl.clear()

    def on_env_reset(self, obs: str, info: Dict[str, Any]):
        context = None
        if info is not None:
            context = info.get(self.context_key)
        if context is None and self.allow_context_fallback:
            context = obs
        if context is None:
            context = ""
            logger.warning(
                "RlmAgent received no context; set info[%r] or enable fallback.",
                self.context_key,
            )
        self._repl.reset(str(context))
        self._ctx.on_env_reset(obs, info)

    async def _run_subcall(self, text: str) -> str:
        messages: List[Message] = []
        if self._sub_system_prompt:
            messages.append({"role": "system", "content": self._sub_system_prompt})
        messages.append({"role": "user", "content": text})
        req = ChatCompletionRequest(
            model=self._sub_model,
            messages=messages,
            sampling=self._sub_inference.sampling,
            return_=self._sub_inference.return_,
            extensions=self._sub_inference.extensions,
        )
        resp, _info = await self._client.complete(req)
        return resp.text

    def _next_tool_id(self, prefix: str) -> str:
        self._tool_counter += 1
        return f"{prefix}-{self._tool_counter}"

    async def act(
        self,
        inference: Optional[InferenceSpec] = None,
        sampling_seed: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Tuple[ParseResult, str, Dict[str, Any], Optional[TokenTrace]]:
        inf = inference or InferenceSpec()
        rlm_actions: List[Dict[str, Any]] = []
        last_info: Dict[str, Any] = {}
        last_trace: Optional[TokenTrace] = None
        last_raw = ""

        for _step_i in range(self.max_steps):
            messages = self._ctx.on_before_act()
            resp, _client_info, last_info, token_trace = await self._infer_once(
                messages=messages,
                inference=inf,
                sampling_seed=sampling_seed,
                timeout_s=timeout_s,
            )
            last_trace = token_trace
            last_raw = resp.text
            self._ctx.on_after_act(resp)

            if self._reject_incomplete and resp.finish_reason == "length":
                call_id = self._next_tool_id("incomplete")
                self._ctx.add_tool_result(call_id, "incomplete", self._incomplete_feedback)
                rlm_actions.append(
                    {"type": "incomplete", "feedback": self._incomplete_feedback}
                )
                continue

            final_var = _extract_first(_FINAL_VAR_RE, resp.text)
            if final_var:
                var_name = _maybe_unquote(final_var)
                if not _IDENT_RE.match(var_name):
                    call_id = self._next_tool_id("error")
                    msg = f"Invalid FINAL_VAR identifier: {var_name!r}"
                    self._ctx.add_tool_result(call_id, "rlm_error", msg)
                    rlm_actions.append({"type": "error", "message": msg})
                    continue

                output = self._repl.eval_var(var_name)
                rlm_actions.append(
                    {"type": "final_var", "var": var_name, "value": output}
                )
                parse_result = self._parser(output)
                last_info["rlm_actions"] = rlm_actions
                return parse_result, last_raw, last_info, last_trace

            final_text = _extract_first(_FINAL_RE, resp.text)
            if final_text:
                parse_result = self._parser(final_text)
                rlm_actions.append({"type": "final", "text": final_text})
                last_info["rlm_actions"] = rlm_actions
                return parse_result, last_raw, last_info, last_trace

            rlm_call = _extract_first(_RLM_CALL_RE, resp.text)
            if rlm_call:
                call_text = _maybe_unquote(rlm_call)
                reply = await self._run_subcall(call_text)
                call_id = self._next_tool_id("rlm_call")
                self._ctx.add_tool_result(call_id, "rlm_call", reply)
                rlm_actions.append(
                    {"type": "rlm_call", "query": call_text, "response": reply}
                )
                continue

            code_block = _extract_first(_CODE_BLOCK_RE, resp.text)
            if code_block:
                output = self._repl.run(code_block)
                call_id = self._next_tool_id("python")
                self._ctx.add_tool_result(call_id, "python", output)
                rlm_actions.append(
                    {"type": "python", "code": code_block, "output": output}
                )
                continue

            call_id = self._next_tool_id("error")
            msg = (
                "No FINAL(...), FINAL_VAR(...), RLM_CALL(...), or python code "
                "block found."
            )
            self._ctx.add_tool_result(call_id, "rlm_error", msg)
            rlm_actions.append({"type": "error", "message": msg})

        parse_result = ParseResult(
            action=None,
            reward=self._exhausted_penalty,
            obs=self._exhausted_feedback,
        )
        last_info["rlm_actions"] = rlm_actions
        return parse_result, last_raw, last_info, last_trace
