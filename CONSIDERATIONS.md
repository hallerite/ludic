# Considerations (Notes for Later)

## DCP (Distributed Checkpointing) and Resuming With Different GPU Counts

We use PyTorch **DCP** (“Distributed Checkpointing”) APIs for FSDP2:

- Saving gathers a **full (unsharded)** state dict with `full_state_dict=True` (and `cpu_offload=True`).
- Loading uses `set_model_state_dict(..., full_state_dict=True, broadcast_from_rank0=True)` so rank0 provides the full weights and each rank **re-shards locally**. But this behavior is untested and I intend to explicitly test it.

Implication:

- **Resuming with a different `world_size` / GPU count should work**, because we are saving/loading a **full** state dict (not a sharded DTensor checkpoint).

When it would *not* work:

- If we saved a **sharded** checkpoint (DTensor placements tied to a device mesh), then the checkpoint would generally be coupled to the original sharding/world-size (unless additional resharding logic is implemented).

## Checkpoint Format Notes (HF Sharded Saving)

Our checkpoint code collects a full state dict via DCP, but then writes it using HuggingFace `save_pretrained(..., state_dict=...)` when available (see `src/ludic/training/checkpoint.py`).

Notes:

- `save_pretrained()` may write **sharded** weights for large models (e.g. `model-00001-of-000xx.safetensors` + `model.safetensors.index.json`).
- Our loader (`CheckpointManager._read_state_dict`) supports:
  - single-file `pytorch_model.bin` / `pytorch_model.pt`
  - sharded torch checkpoints via `pytorch_model.bin.index.json`
  - single-file `model.safetensors`
  - sharded safetensors via `model.safetensors.index.json` (requires `safetensors`)

## Truncation Semantics (Env vs LLM vs Protocol)

There are multiple "things that look like truncation" in the system. For correctness and for clean training stats, these are now tracked as distinct signals.

### Current Implementation

- **Env-level episode endings:** `StepOutcome` and `Step` carry `terminated` and `truncated` booleans (`src/ludic/types.py`), and interaction protocols stop when either is true.
- **Protocol time-limit truncation:** When a protocol hits `max_steps` without env termination/truncation:
  - The last step's `truncated` is set to `True`
  - `Step.info["truncation_reason"] = "max_steps"` is added
  - `Rollout.meta["episode_truncated"] = True` and `Rollout.meta["truncation_reason"] = "max_steps"` are set
  - For parser failure steps, the synthetic `next_obs` is preserved (not cleared to None)
- **LLM "finish reason" is separate:** the vLLM/OpenAI response `finish_reason` is captured into `ChatResponse.finish_reason` and merged into `Step.info` (via `ChatResponse.to_info()`), but this is *not* the same as env truncation.
- **Incomplete LLM completions (default = reject):** if the model returns `finish_reason == "length"` and the `Agent` has `reject_incomplete_completions=True` (default), the step is treated as a parse failure:
  - `ParseResult.action=None` so the protocol does not call `env.step()`
  - `Step.info["incomplete_completion"] = True` is set (and `finish_reason` is preserved in `Step.info`)
  - the raw truncated assistant message is still appended to the agent context, and the agent also receives the synthetic feedback observation for the next turn
- **Training sees truncation:** `RolloutEngine.generate_batch()` now propagates `Step.truncated` and `Step.terminated` into `SAWItem.meta`, along with `episode_truncated` and `truncation_reason` from `Rollout.meta`.
- **Training-time sequence truncation:** micro-batching caps sequences at `max_seq_len`, sets `seq_len_truncated`/`seq_len_*`, and marks `SAWItem.meta["truncated"]=True` with `truncation_reason="max_seq_len"` when no episode-level reason is present.

### Definitions

Three concepts are kept separate:

- `terminated`: the environment reached a true terminal state (success/failure terminal).
- `truncated`: the environment (or protocol) ended an episode due to a time limit / external cutoff (not necessarily terminal in the MDP sense). In training metadata, this can also be set when sequences are capped at `max_seq_len` (see `seq_len_truncated`).
  - `truncation_reason = "max_steps"`: protocol time limit hit
  - `truncation_reason = "env"`: environment-initiated truncation
- `llm_finish_reason`: the model stopped generating for some reason (`stop`, `length`, `tool_calls`, etc.).

### Should we mask out truncated samples for training?

Not universally; it depends on what `truncated` means in the environment and the algorithm:

- If truncation is a **Gym-style time limit** (episode cut off but task is "continuing"), then pure Monte Carlo returns (no value bootstrap) treat truncation like terminal and can bias learning. In that case:
  - a "correct" RL treatment is to **bootstrap** from the value function at the truncation boundary (requires a critic/value model), or
  - if you're doing pure Monte Carlo without bootstrapping, it can be safer to **drop** truncated episodes or downweight them to avoid bias (at the cost of data efficiency).
- If truncation is a **meaningful failure mode** ("timed out" is a real terminal condition for the task), then you should **keep** truncated samples and ensure the reward signal reflects the timeout.

### LLM `finish_reason == "length"` is not env truncation

`finish_reason=="length"` typically means the model hit `max_tokens` and produced an incomplete completion/action.

In Ludic (by default), these incomplete completions are rejected at the `Agent` level and treated like parse failures (see above), so they can be tracked and filtered separately from env/protocol truncation.

## Why AgentStep and EnvironmentStep?

The core insight: **training needs visibility into the agent's reasoning process, not just its final actions.**

Consider a ReAct-style agent that:
1. Thinks: "I should calculate 2+3 first"
2. Calls calculator tool → gets "5"
3. Thinks: "Now I can answer"
4. Outputs: `<move>5</move>`

If we only logged the final action, we'd lose steps 1-3 entirely. But those steps contain:
- Token traces we want to train on (the model's reasoning)
- Tool call/result pairs that shaped the final decision

**AgentStep** captures every model call:
- Prompt tokens (state) + completion tokens (action) + TokenTrace
- Tool calls made and results received
- `action_target` indicating what happens next

**EnvironmentStep** captures state transitions:
- Previous observation, parsed action, next observation
- Environment reward
- References back to the AgentSteps that produced this action

This separation means:
- Rollouts contain the full reasoning trace
- Training can concatenate all AgentSteps in a turn into one sample
- Environments stay simple (they never see tool calls, just parsed actions)

## Tool Scopes: Internal vs External

Tools are divided by **who handles them**:

**Internal tools** (`tools` parameter):
- Agent executes them locally (calculator, code interpreter, etc.)
- Results are added to agent's context
- Agent continues reasoning with the new information
- Recorded as AgentStep with `action_target="internal"`

**External tools** (`external_tools` parameter):
- Agent returns control to the protocol
- Protocol decides how to handle (delegation, sub-agent calls, etc.)
- Protocol feeds results back to agent context
- Agent continues reasoning
- Recorded as AgentStep with `action_target="external"`, `parse_result=None`

**Final actions** (no tool call, or after tool loop completes):
- Agent outputs text that the parser extracts an action from
- Protocol calls `env.step()` with the parsed action
- Recorded as AgentStep with `action_target="env"` + EnvironmentStep

Key distinction: **external tools are NOT environment actions**. They're intermediate
steps where the protocol does something (call a sub-agent, fetch data, etc.) and the
agent continues. The environment only sees the final parsed action.

## action_target Semantics

Each AgentStep has an `action_target` indicating what the protocol should do:

| action_target | parse_result | Protocol behavior |
|---------------|--------------|-------------------|
| `"internal"`  | set          | Nothing - agent handled internally, loop continues |
| `"external"`  | `None`       | Call `external_tool_handler`, feed result to agent, call `act()` again |
| `"env"`       | set          | Parse action, call `env.step()`, record EnvironmentStep |

The `parse_result=None` for external tools is intentional: they're not final actions,
so there's nothing to parse. The protocol handles them and the agent continues.

## Training Implications

- Tool result messages are prompt tokens (`action_mask=0`). Only assistant completions
  are action tokens.
- Turn concatenation stitches all AgentSteps together, so internal/external tool calls
  are included in training automatically.
- Credit/weight comes from the final step in the turn (EnvironmentStep if present,
  otherwise last AgentStep for parse failures).

Current behavior:
- `ReActAgent` runs multiple model calls and returns per-call AgentSteps with token
  traces, plus one final env-targeted step.
- `SingleAgentSyncProtocol` handles external tools via `external_tool_handler` callback.
- `MultiAgentSyncProtocol` logs AgentSteps but does not yet execute external tools.

## Delegation: Sub-Agent Calls via External Tools

External tools enable **hierarchical agent architectures** where a parent agent can
delegate subtasks to child agents. The key insight: the agent doesn't know *how* an
external tool is handled—it just gets a result back. The protocol decides whether to
call an API, spawn a sub-agent, or do something else entirely.

### How It Works

1. Parent agent calls `delegate(task="solve this subproblem")` (an external tool)
2. Protocol receives the tool call with `action_target="external"`
3. Protocol spawns a sub-agent with its own context, runs it to completion
4. Sub-agent produces its own rollout (with its own AgentSteps)
5. Protocol extracts the result and feeds it back to parent agent's context
6. Parent agent continues reasoning with the result
7. Both rollouts are collected for training

```
Parent Agent                          Protocol                         Sub-Agent
     |                                    |                                 |
     |-- delegate(task) ----------------->|                                 |
     |   [action_target="external"]       |                                 |
     |                                    |-- spawn + run ---------------->|
     |                                    |                                 |
     |                                    |<-- sub_rollout + result -------|
     |                                    |                                 |
     |<-- result fed to context ---------|                                 |
     |                                    |                                 |
     |-- continue reasoning              |                                 |
```

### Why This Matters for Training

Both the parent and child rollouts contain valuable signal:
- **Parent rollout**: Learns *when* to delegate and *how* to use results
- **Child rollout**: Learns *how* to solve the delegated subtask

Credit assignment can flow through the hierarchy:
- Child gets reward based on subtask success
- Parent gets reward based on final task success (which depends on good delegation)

This is inspired by "Context-Folding" (branch/return operations for long-horizon agents)
but implemented at the protocol level rather than requiring special agent architectures.

### Implementation Path

The infrastructure is in place:
- `external_tools` parameter on `ToolAgent`/`ReActAgent`
- `external_tool_handler` callback on `SingleAgentSyncProtocol`
- `action_target="external"` with `parse_result=None` for external tool calls

What's needed for delegation:
- A `DelegatingProtocol` that:
  - Maintains a registry of sub-agent factories
  - Spawns sub-agents when `delegate` tool is called
  - Collects sub-rollouts alongside parent rollouts
  - Returns aggregated rollouts for training
- Credit assignment strategies that handle hierarchical rollouts

## Turn-Concatenated Training Samples (Default)

Online RL batching builds **one SAWItem per agent turn**, not per step.

Why:
- We want a **true transcript** of the internal loop (reasoning + code + feedback)
  to be the training signal, not a bag of per-step fragments.
- We do **not** want to rely on chat template retokenization; instead we stitch the
  rollout-time token traces from each AgentStep.

How:
- Start with the first AgentStep's prompt token IDs.
- Append each AgentStep's completion token IDs as action tokens (action_mask = 1).
- For subsequent AgentSteps, append only the **prompt suffix** that extends the
  existing sequence (action_mask = 0), then append that step's completion.
- Interpreter outputs are user messages in the prompt, so they are **masked out**
  automatically.
- Credit/weight is taken from the **final step in the turn** (env step if present,
  otherwise the last agent step for parse failures).

Constraint:
- This requires **append-only** context histories. Context strategies that truncate
  or rewrite prior messages (e.g., thinking truncation in prompts) will break the
  prefix-matching invariant, so they are **not supported** for online training yet.

## Future: First-Class Evaluation + Better Layering

Right now, evaluation utilities and examples often live near the training stack because they reuse `RolloutEngine` and reducers. This is convenient, but it muddies the conceptual layering:

- Rollout execution is not inherently “training”.
- Evaluation is not inherently “training” (especially in disaggregated inference setups where eval is “send requests to inference GPUs”).

Potential future refactor (larger change):

- Promote rollout execution to a non-training namespace (e.g. `ludic.rollouts` / `ludic.execution`) by moving `RolloutEngine` out of `ludic.training.*`.
- Create a first-class `ludic.eval` module that depends on rollout execution and produces:
  - step/episode outputs suitable for analysis
  - aggregated metrics via reducers
  - optional CLI helpers for examples
- Make `Trainer.eval()` a thin consumer of `ludic.eval` (rather than owning eval logic/config), keeping Trainer focused on optimization.

This would make evaluation a first-class citizen, reduce duplicated example code, and keep the training module’s surface area smaller/cleaner.

## Future: Remove `RolloutRequest.num_episodes`

`RolloutRequest` currently supports `num_episodes > 1`, which makes one “request” behave like a small workload (“run this template N times”).

This is convenient in a few places, but it is conceptually awkward and has practical downsides:

- It conflates **one execution spec** with **many executions**, making progress reporting and per-episode metadata harder to reason about.
- It complicates algorithms that already treat requests as “one execution trace” (e.g., GRPO-style request expansion).

Planned refactor (breaking change):

- Remove `RolloutRequest.num_episodes` and any engine code that expands it internally.
- Define the semantics as: **1 `RolloutRequest` ⇒ 1 episode** (which may still yield multiple `Rollout`s for multi-agent protocols).
- Handle multiplicity at a higher layer by explicitly generating more requests, e.g. via:
  - `RequestStrategy.expand(...)` (one-to-many request transforms), and/or
  - a small helper to “repeat” a request with deterministic seeding.
