# PLAN: Context-Folding + FoldGRPO (in the spirit of the paper)

This document proposes an implementation plan for **Context-Folding** and a **FoldGRPO-like** training signal inside Ludic.

Target paper: *Scaling Long-Horizon LLM Agent via Context-Folding* (Sun et al., 2025).

The goal is to match the paper’s **behavioral shape** and **system-level constraints** (active context stays small; long interaction history is handled via branch/fold) while staying consistent with Ludic’s design intent: loosely coupled pieces, explicit interaction protocol, and training consuming `SAWItem`s rather than “episodes”.

---

## 0) Scope and success criteria

### In scope
- **Context-Folding agentic mechanism**
  - Agent can `branch(description, prompt)` into a sub-trajectory and later `return(message)` to fold the branch, keeping only a concise summary in the main thread.
  - **Plan/Execution** flavor: high-level reasoning in main thread; token-heavy tool usage is routed into branches; **no nested branching** (paper’s simplification).
- **RL training in the spirit of FoldGRPO**
  - Keep GRPO-style grouping and advantage normalization as the primary signal for final success.
  - Add **process-shaped penalties** that nudge folding behaviors (unfolded-token penalty, out-of-scope penalty, tool-failure penalty).
  - Make the fold operation apply to **what the model conditions on** (folded prompt view), not necessarily physically deleting all stored transcript.

### Not in scope (initially)
- Perfect replication of the paper’s **token-level clipped objective** across the whole tree (requires deeper surgery of `SAWItem`/loss).
- True **KV-cache rollback** on the inference server (vLLM doesn’t expose full rollback; we can get most of the benefit from prefix caching and prompt folding).
- Parallel branching (paper treats as open question); implement single active branch first.

### Acceptance tests / “done” definitions
1. **Functional folding**: in a toy environment, agent can create a branch, do several tool calls inside it, `return()` a summary, and the main-thread prompt no longer contains the branch’s intermediate tool traces.
2. **Training works**: a training script can collect rollouts and train with GRPO + process rewards without crashing; metrics show the main-thread prompt length decreases over training.
3. **Measurable compression**: log `main_active_tokens`, `branch_tokens_total`, `fold_ratio = total_tokens / main_active_tokens` per episode.
4. **Safety invariants**: no nested branches (or explicit bounded nesting), branch must terminate via `return` or max steps; failures are logged and penalized.

---

## 1) Key observations about current Ludic architecture (where to hook)

Ludic already has the right seams:
- **Context / memory policy**: `src/ludic/context/base.py`, `src/ludic/context/full_dialog.py`, `src/ludic/context/truncated_thinking.py`
- **Explicit env-agent loop**: `src/ludic/interaction/base.py`, `src/ludic/interaction/single_agent.py`, `src/ludic/interaction/multi_agent.py`
- **Training is “just SAW batches”**: `src/ludic/training/types.py`, `src/ludic/training/batching/rollout_engine.py`, `src/ludic/training/trainer.py`
- **GRPO exists**: request expansion `src/ludic/training/batching/intra_batch_control.py`, group baseline `src/ludic/training/credit_assignment.py`, clipped surrogate loss in `src/ludic/training/loss.py`

Main friction points relative to the paper:
- The current `ReActAgent` performs multiple tool calls **inside `Agent.act()`** (`src/ludic/agents/react_agent.py`), which hides intermediate tokens/steps from the protocol and training dataflow.
  - For folding to be learnable and for process penalties to apply cleanly, **tool invocations should become explicit protocol steps** (or at least explicit “substeps” in rollout metadata).

---

## 2) Design: what “Context Folding” means in Ludic

### Terms
- **Main thread**: the primary conversation context that should stay compact.
- **Branch**: a temporary sub-context created for a localized subtask.
- **Fold**: when a branch ends, its internal turns are removed from the *active prompt view* and replaced by a short summary message that is appended to main.

### What we must preserve
- For agent behavior: the model must see the **folded prompt view** `F(history)` at every generation step.
- For training: we still need to keep enough metadata to compute rewards/penalties and to produce `SAWItem`s from model token traces.

### Proposed “in spirit” compromise
- Store **full transcript** internally for debugging/analysis.
- Expose only **folded transcript** to the model (`ContextStrategy.on_before_act()` returns folded view).
- In rollouts, represent each model generation that we train on as a `Step`, and attach metadata that indicates:
  - `thread`: `"main"` or `"branch"`
  - `branch_id`: stable ID for branch steps
  - `fold_event`: emitted on return (folded summary + folded token counts)
  - `active_prompt_tokens`: tokens in the folded prompt at that generation (we can approximate from prompt token ids if available)

---

## 3) New primitives to add (minimal, composable)

### 3.1 New context strategy: `FoldingContext`
Create `src/ludic/context/folding.py` implementing a `ContextStrategy` that supports:
- `branch(description, prompt)` and `return(message)` as **context-management actions**.
- Two “stores”:
  - `self._main_messages`: canonical main thread transcript
  - `self._branch_messages`: transcript for the active branch (or `None`)
- A “mode”:
  - `self._mode in {"main","branch"}`
- A “fold log”:
  - list of fold events with `branch_id`, counts, and the returned summary

Behavioral API (internal; protocol calls these methods):
- `start_branch(description: str, prompt: str) -> BranchHandle`
  - Appends a *branch-created marker* into main (as in paper, this can be templated).
  - Switches mode to branch; seeds branch context with:
    - shared prefix (system prompt + main so far) OR
    - a fresh context containing an explicit “branch prompt” + necessary pinned context
  - The choice matters for learnability and KV-cache friendliness; in spirit of the paper, prefer **shared prefix**.
- `finish_branch(message: str) -> None`
  - Generates a templated message summarizing the result.
  - Switches back to main, and appends only the summary (folding away internal branch transcript from the prompt view).

Prompt view:
- `on_before_act()` returns:
  - if in main: main messages + folded summaries (no internal branch details)
  - if in branch: main prefix + branch transcript (no ability to branch again)

Note: Ludic already has “prompt-only transformation” precedent in `TruncatedThinkingContext` (`src/ludic/context/truncated_thinking.py`).

### 3.2 Action contract: represent `branch` and `return` as actions

We need the model to emit branch/return “calls” in a way the protocol can parse.

Two viable approaches:
1. **Tool-calling style**: represent `branch` and `return` as tools using `ToolAgent`.
   - Pros: structured JSON args; consistent with OpenAI/vLLM tool calling; easy to intercept.
   - Cons: current `ToolAgent` executes python callables immediately; `return` needs to be handled specially (it changes protocol/context mode, not the env).
2. **Text contract + parser**: model emits e.g. `<branch>{"description":...,"prompt":...}</branch>` and `<return>{"message":...}</return>`.
   - Pros: no dependency on tool-calling behavior; works with any inference backend.
   - Cons: more brittle; must ensure robust parsing.

Recommendation: implement **(1) tool-calling** first, but ensure we can also run in pure-text mode via a parser later.

Concrete plan for tool-calling:
- Add two callables:
  - `branch(description: str, prompt: str) -> str`
  - `return(message: str) -> str`
- Do **not** execute them as normal tools in `ToolAgent._run_tool_calls`; instead, the protocol should intercept these tool calls and treat them as control-plane actions.
  - That implies either:
    - a specialized agent class that reports tool calls but doesn’t execute them, or
    - a protocol wrapper that can override tool execution for specific tool names.

---

## 4) Interaction protocol: make folding explicit and learnable

### 4.1 Create `FoldingSingleAgentProtocol`
Add a new protocol in `src/ludic/interaction/folding_single_agent.py` (or similar) that:
- Runs in two states:
  - **Planning state** (main thread):
    - Allowed actions: `branch(...)` OR “final env action” (depends on env) OR “non-tool reasoning” (discouraged tool use).
  - **Execution state** (branch):
    - Allowed actions: tool calls needed for the subtask + `return(message)`
    - Disallow nested `branch`.
- Handles parser failures and time limits similarly to `SingleAgentSyncProtocol` (`src/ludic/interaction/single_agent.py`).

Crucially: represent every model generation we optimize as a `Step`:
- When model calls a tool:
  - log a `Step` for the tool-call “assistant output”
  - execute the tool and log tool result into context
  - continue
- When model emits a branch call:
  - log a `Step` (thread=main, action=branch call)
  - transition state to branch and call `FoldingContext.start_branch(...)`
  - the tool result can be a template message (“branch created …”)
- When model emits a return call:
  - log a `Step` (thread=branch, action=return call)
  - call `FoldingContext.finish_branch(message)`
  - transition state back to main

This is the major architectural choice that makes folding trainable without hidden internal loops.

### 4.2 Where environment stepping happens
For many long-horizon “agentic coding” setups, the “environment” is effectively tool-based, not a classic MDP stepper.

Two patterns:
- **Tool-only env** (recommended for SWE-style):
  - Env is minimal; most work is tools; termination is `finish()` style.
- **Env step** (tic-tac-toe style):
  - The final action in main thread is parsed and applied via `env.step`.

Plan: keep env step semantics identical to `SingleAgentSyncProtocol` for compatibility:
- In main thread, once an env action is emitted (parse success), call `env.step`.
- In branch mode, **do not call `env.step`**; branch is for tool work only (paper plan/execution).

---

## 5) Rollout + metadata: what to log for training and analysis

### 5.1 Step.info additions
Add consistent keys in `Step.info` (merged safely using `merge_step_info` in `src/ludic/interaction/info.py`):
- `fold_thread`: `"main"` | `"branch"`
- `fold_branch_id`: `str | None`
- `fold_event`: one of:
  - `"branch_start"` (on branch tool call)
  - `"branch_return"` (on return tool call)
  - `None`
- `fold_main_active_prompt_len`: int (approx tokens in folded prompt view at generation time)
- `fold_branch_prompt_len`: int (optional)
- `tool_name`: name when applicable
- `tool_failed`: bool
- `out_of_scope`: bool (if judged)

### 5.2 Episode-level Rollout.meta additions
- `fold_num_branches`
- `fold_total_tokens` (sum completion tokens)
- `fold_main_tokens`
- `fold_branch_tokens`
- `fold_ratio` (total / main)

### 5.3 Tokenization drift
For GRPO and any ratio objective, Ludic prefers using backend-provided `completion_token_ids` and `completion_logprobs` (`src/ludic/training/batching/rollout_engine.py`, `src/ludic/training/algorithm.py`).

Requirement: ensure inference requests used by folding protocol set:
- `return_token_ids=True`
- `return_chosen_logprobs=True` (or accept backfill teacher-forcing for fully on-policy runs)

---

## 6) FoldGRPO “in spirit”: algorithm + rewards

We want to stay close to Ludic’s current “algorithm = credit + loss” framing:
- final success reward (RLVR style; typically {0,1} or test pass/fail)
- GRPO grouping (same env seed, different sampling seeds)
- plus shaping penalties to guide folding behaviors

### 6.1 Where to put process rewards
In Ludic, weights come from the `CreditAssigner`, which currently consumes `Rollout`s and returns a scalar per `(rollout_id, step_index)` (`src/ludic/training/types.py`, `src/ludic/training/credit_assignment.py`).

We can implement FoldGRPO-style shaping by:
- keeping the **environment reward** in `Step.reward`
- and adding **process reward** by augmenting the computed weight per step:
  - `weight(step) = group_adv(rollout) + process_bonus(step)`

This deviates from the paper’s token-level `Q_{i,t}` inside the clipped objective, but it’s “in spirit” and keeps changes localized.

### 6.2 Implement `FoldGRPOCreditAssigner`
Add `src/ludic/training/foldgrpo.py` (or add to `credit_assignment.py`) with a new assigner:
- Inputs: `group_size`, thresholds, penalty magnitudes, optional judge.
- Steps:
  1. Compute base GRPO advantage per rollout: `A_group = R_ep - mean(R_group)` (optionally normalize).
  2. For each step, compute `Q_step`:
     - **Unfolded token penalty** (main thread only):
       - If `fold_main_active_prompt_len` exceeds `prompt_budget * threshold` (e.g. 0.5 of active context limit), apply `Q=-1` to main-thread steps *except* the branch-start step(s).
       - This encourages branching before main gets “too big”.
     - **Out-of-scope penalty** (branch only):
       - Judge whether branch stayed within scope.
       - If out-of-scope, apply `Q=-0.2` to all steps in that branch.
     - **Failure penalty**:
       - If `tool_failed` or parse error, apply `Q=-1` to that step.
  3. Return `weight(step) = A_group + Q_step` (optionally clip to a range).

Notes:
- This assigner needs access to:
  - group IDs (`rollout.meta["request_meta"]["group_id"]`) already required by `GroupNormalizedReturn`.
  - fold/thread metadata in steps.
- Keep all config values in a dataclass (`FoldGRPOConfig`) for clarity.

### 6.3 Judge integration (“out-of-scope”)
Implement an optional judge module:
- `src/ludic/scoring/folding_scope_judge.py` (or similar) with a simple interface:
  - `judge(branch_prompt: str, return_message: str) -> bool` (True means in-scope)
- Use an LLM judge client (any `ChatClient`) configured separately.

Because network is often restricted and judge calls add cost, ensure:
- the judge is optional
- you can run without it (e.g., skip penalty or approximate with heuristic regex checks)
- the judge results are cached per `(branch_prompt, return_message)` hash during a rollout batch.

---

## 7) Training loop wiring (batching, grouping, async)

### 7.1 Request expansion (GRPO)
Reuse `GRPORequestStrategy` (`src/ludic/training/batching/intra_batch_control.py`) as-is.

### 7.2 RolloutEngine + SAWItem creation
`RolloutEngine` already converts `Step.info` token traces into `SAWItem`s (`src/ludic/training/batching/rollout_engine.py`).

Ensure folding protocol:
- puts `prompt_token_ids`, `completion_token_ids`, `completion_logprobs`, `finish_reason` into `Step.info` via `ChatResponse.merge_into_info`.
- sets `Step.reward` to include env reward + any immediate penalties you want baked into returns (optional; but prefer penalties in credit assigner to keep env clean).

### 7.3 Loss selection
To stay “in spirit” and minimize changes:
- Keep `make_grpo()`’s `ClippedSurrogateLoss` (`src/ludic/training/loss.py`) initially.
- Swap credit assigner from `GroupNormalizedReturn` to `FoldGRPOCreditAssigner`.

Later “more faithful” option:
- Implement token-level per-step weights and a token-level clipped loss (requires extending `SAWItem` to carry per-token advantages or “adv_masked” tensors).

### 7.4 Metrics to add
Add reducers or batch-level stats in `aggregate_stats` to report:
- `fold/main_active_prompt_len_mean`
- `fold/num_branches_mean`
- `fold/fold_ratio_mean`
- `fold/tool_calls_mean`
- `fold/out_of_scope_rate`
- `fold/return_rate` (# branches that returned / # branches started)

---

## 8) Example environments: how to get a working demo quickly

### 8.1 Toy environment for correctness
Create a small single-agent env:
- “Research-style”: tools are `search` and `open_page` but backed by a local corpus (no network) to keep tests deterministic.
- Or a “filesystem exploration” tool that reads files from a known directory and returns snippets.

Goal: demonstrate folding behavior without needing SWE-Bench infrastructure.

### 8.2 SWE-like environment (later)
If you want to approach the paper’s SWE setting:
- follow the repo’s existing patterns (examples + sandboxed tooling)
- implement tools:
  - `execute_bash(cmd)`
  - `read_file(path)`
  - `edit_file(path, ...)`
  - `run_tests(...)`
and a verifier that yields terminal reward.

---

## 9) Prompting / behavioral scaffolding

Even with RL, folding benefits from a decent scaffold.

System prompt guidelines:
- Explicitly teach the two modes:
  - Main thread: plan, decide what to delegate; avoid heavy tool logs.
  - Branch: execute subtask; when done, summarize and return.
- Hard constraints:
  - “Do not create branches inside a branch.”
  - “Always return from a branch once the subtask is complete.”
  - “Return message must include: key facts, file paths, commands run, and any open uncertainties.”

Add a small prompt template helper to keep prompts consistent across tasks.

---

## 10) Testing strategy

Add unit tests for:
- FoldingContext prompt view folding:
  - after a branch does N tool calls, `on_before_act()` in main thread does not include those tool messages.
- Protocol state machine:
  - branch start transitions to branch mode
  - return transitions back to main and logs folded summary
  - nested branch attempt yields parse/tool error and penalty
- Credit assigner penalties:
  - main prompt length threshold triggers unfolded-token penalty
  - tool failure penalty attaches to the right steps

Add a small integration test that runs a 1–2 episode rollout using a fake ChatClient that emits scripted tool calls.

---

## 11) Implementation checklist (ordered)

### Phase A — Folding mechanism (no RL)
1. Implement `FoldingContext` (`src/ludic/context/folding.py`).
2. Implement `FoldingSingleAgentProtocol` that:
   - uses tool calling
   - makes tool calls and folding transitions explicit steps
3. Add a scripted demo with a fake client + fake tools; verify folded prompt size.

### Phase B — Trainable dataflow (GRPO + shaping)
4. Add fold metadata logging in `Step.info` and `Rollout.meta`.
5. Implement `FoldGRPOCreditAssigner` (step-level shaping on top of group advantages).
6. Add an algorithm preset `make_foldgrpo(...)` alongside `make_grpo()` in `src/ludic/training/algorithm.py`.
7. Add metrics reducers for fold stats.

### Phase C — Optional “paper-closer” improvements
8. Token-level process rewards:
   - extend `SAWItem` to carry per-token weights or store them in `meta`
   - implement tokenwise clipped objective variant
9. Judge model integration + caching.
10. vLLM efficiency knobs:
   - ensure `return_token_ids` and `logprobs` are on
   - experiment with prefix cache resets between episodes; measure speed.

---

## 12) Risks and mitigations

- **Hidden tool loops**: avoid `ReActAgent` for folding; keep the loop in the protocol.
- **Training instability**: shaping penalties can dominate; gate them (apply only after threshold) and log magnitudes.
- **Judge variance/cost**: make judge optional; cache aggressively; consider heuristics.
- **Token length estimation**: prompt token ids may be unavailable on some backends; provide fallback to approximate with string length and document the limitation.

---

## 13) What “good” looks like after Phase B

You should be able to run a GRPO-style training loop where:
- episodes that succeed but keep too much in main thread are penalized,
- policies learn to:
  - branch earlier for “heavy” work,
  - return promptly with concise, useful summaries,
  - keep the main active context small,
- and you can empirically plot:
  - `pass@1` (or success rate) vs `main_active_tokens`,
  - average tool calls vs branch count,
  - fold ratio vs difficulty bucket.

