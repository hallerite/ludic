# Math Tool Examples

This directory contains examples for evaluating tool-augmented agents on GSM8K math problems.

## Available Examples

### 1. ReAct Agent (`eval_math_tool_gsm8k_vllm.py`)

Uses a **ReActAgent** with OpenAI-style function calling:

```bash
PYTHONPATH=. uv run python examples/math_tool/eval_math_tool_gsm8k_vllm.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 --port 8000 \
    --limit 10
```

- Tool format: Structured function calls (`python_eval(expression="2+2")`)
- Tool mechanism: OpenAI/vLLM tool calling API
- Internal steps: `action_target="internal"` for tool calls
- Final step: `action_target="env"` for numeric answer

### 2. ReTool Agent (`eval_retool_gsm8k_vllm.py`)

Uses a **ReToolAgent** with in-band code execution:

```bash
PYTHONPATH=. uv run python examples/math_tool/eval_retool_gsm8k_vllm.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 --port 8000 \
    --limit 10
```

- Tool format: Free-form text with `<code>```python\n...\n```</code>` blocks
- Tool mechanism: In-band parsing, no special API needed
- Internal steps: `action_target="internal"` for code blocks
- Final step: `action_target="env"` for `\boxed{...}` answer

## Key Differences

| Aspect | ReAct Agent | ReTool Agent |
|---------|-------------|---------------|
| **Tool format** | Structured function calls | Free-form text with tags |
| **Backend API** | Requires tool calling support | Standard chat API only |
| **Code execution** | Tool result in OpenAI format | Result in `<interpreter>...</interpreter>` |
| **Final answer** | Extracted from text | Must be in `\boxed{...}` |
| **Step recording** | All agent steps recorded | Including internal code blocks (ReTool) vs only env steps (ReAct) |
| **Inspired by** | ReAct paper | ReTool paper (code-interleaved RL) |

## System Prompt Differences

**ReAct Agent:**
```
You are a careful math tutor. You may call the python_eval tool to
compute arithmetic expressions. Call it with a single argument named
'expression' that contains only a plain arithmetic expression (no
variables, no functions). When you answer, put the final numeric
answer inside \boxed{...}.
```

**ReTool Agent:**
```
You are a careful math tutor. Think step-by-step. When you need to
compute something, write Python code inside <code>```python
...
```</code> tags. The code will be executed and result will be
shown to you. Place your final numeric answer inside \boxed{...}.
```

## Output

Both scripts generate a JSONL file with:
- `rollout_id`, `step_index`, `reward`
- `action`, `total_reward`, `completion_length`
- `step_kind`: "agent" or "env"
- `action_target`: For agent steps, "internal" (code blocks) or "env" (final answer)
- Decoded prompt/completion (if `--decode-tokens`)
- Tool/code block statistics

**Note:** ReTool script records **ALL agent steps** (including internal code block steps), while ReAct script records only env steps. This gives complete visibility into the agent's reasoning and tool usage.

Metrics:
- `accuracy`: Fraction of correct answers
- `parse_error_rate`: Fraction of parse failures
- `avg_completion_tokens`: Average response length
- `avg_[code_blocks|tool_calls]_*`: Code/tool usage stats

## Requirements

```bash
uv sync --extra examples
```

Requires:
- `datasets`: For loading GSM8K
- `math-verify`: For answer verification
- `transformers`: Optional, for token decoding
- vLLM server running at `--host:--port`
