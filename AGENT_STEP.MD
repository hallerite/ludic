# AgentStep / EnvironmentStep Plan

Intent: represent both internal agent loops and environment transitions as first-class steps, while keeping a single flattened timeline and full traceability from env actions back to the internal chain that produced them.

## Core semantics
- Every model call (including tool loops) creates an AgentStep.
- Environment transitions only happen when an AgentStep is marked as a final env-facing action.
- Rollouts contain a single ordered list of steps with explicit kind tags.
- Each EnvironmentStep references the AgentStep that caused it and the internal chain that led there.

## Step model (proposed)

### BaseStep (shared fields)
- id: str (uuid)
- index: int (monotonic within rollout, across all step kinds)
- kind: "agent" | "env"
- reward: float (sum of reward components, but see reward_components)
- reward_components: {"env": float, "parser": float, "intrinsic": float}
- terminated: bool
- truncated: bool
- info: dict (JSON)
- trace: TokenTrace (required; prompt + completion token ids)
- ts_ns: int
- turn_id: str (groups an internal loop + its env step)
- parent_id: str | None (for nested subagent traces or derived steps)

### AgentStep
Represents a single model call.
- prompt_messages: list[Message] (everything the agent saw)
- action: str (assistant completion text)
- action_target: "internal" | "env" (gate for env.step)
- loop_index: int (0..N within turn)
- tool_calls: list | None (optional, raw tool call schema)
- tool_results: list | None (optional, but can be large; consider truncation)
- source: "model" | "subagent" (optional)

### EnvironmentStep
Represents the env transition triggered by a final AgentStep and carries the full internal chain.
- prev_obs: str (the env observation that started the turn)
- action: str (raw assistant text from source AgentStep)
- parsed_action: any (action actually passed to env.step)
- next_obs: str | None (None on terminal)
- source_agent_step_id: str
- agent_step_ids: list[str] (the internal chain, in-order, for traceability)

## Protocol flow
- On env.reset: start a new turn_id.
- On each agent act call:
  - record an AgentStep with prompt_messages and completion.
  - parse every AgentStep; mark action_target based on parsing or explicit final-step signaling.
- If action_target == "env":
  - run env.step with parsed_action from the final AgentStep (no extra parse pass).
  - create EnvironmentStep with source_agent_step_id and the in-order chain of agent_step_ids for this turn.
  - advance turn_id for the next loop.
- Rewards:
  - AgentStep.reward_components contain intrinsic/parser rewards.
  - EnvironmentStep.reward_components contain env rewards.
  - reward = sum(components) for convenience; credit assignment can pick components or step kinds.

## Tool semantics
- Internal tools (read-only w.r.t. env) stay inside AgentStep chains.
- Env tools should be treated as env-facing actions (action_target="env") and emit EnvironmentSteps.
- tool_scope ("internal" | "env") can be stored in AgentStep.info for downstream filtering.
- Tool results are stored on the triggering AgentStep (tool_calls + tool_results in order), not as separate steps.

## Training / collation
- Batch sources now concatenate the entire agent turn into **one** SAWItem.
- All AgentStep completions in the turn are action tokens (action_mask = 1).
- Tool/interpreter feedback appears as user messages in the prompt and is masked out.
- Credit/weight is taken from the final step in the turn (env step if present, else the last agent step).
- This requires append-only context histories; context strategies that truncate or rewrite prior messages are not supported for online training.
- Credit assignment can be scoped by step kind and/or reward_components.

## Breaking changes
- Step becomes a tagged union (AgentStep / EnvironmentStep). Rollouts and batch sources must update.
- JSONL format changes; no compatibility guarantees.

## Open questions
- Should EnvironmentStep.reward include parser reward, or keep parser reward only on AgentStep?
- Do we store tool_results inline in AgentStep, or store a summarized form to avoid bloating rollouts?
- Do we need a separate token-level loss_mask now or later for tool-call exclusion?
