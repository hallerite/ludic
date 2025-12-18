# PLAN: Replace `sampling_args` With Typed Inference Requests

This document proposes a breaking, typed replacement for ad-hoc `sampling_args` dicts by introducing a first-class *inference request* object. The goal is to eliminate `extras`/`extra_body` dict conventions from the library surface, make trace requirements explicit (token IDs + chosen-token logprobs), and set up a cleaner path toward “token-out by default” online RL (no retokenization, no re-derived logprobs).

Assumption: we do **not** care about backwards compatibility. We can make breaking changes across the repo (examples/tests/public surface) to delete dict-based sampling args entirely.

## 0) Context: What exists today

### Current types & resolution
- `SamplingArgs` is a `TypedDict` in `src/ludic/types.py` (partial dict: `temperature`, `max_tokens`, `seed`, and `extras` for vendor-specific fields).
- `SamplingConfig` is a *fully-resolved* frozen dataclass in `src/ludic/inference/sampling.py`.
- Resolution today:
  - `Agent.act(sampling_args: SamplingArgs)` → `resolve_sampling_args(sampling_args)` → `SamplingConfig` → `ChatClient.complete(..., sampling=SamplingConfig)`.

### Where dicts are constructed (in practice)
- Examples build nested dicts like:
  - `{"temperature": ..., "max_tokens": ..., "extras": {"extra_body": {"return_token_ids": True}}}`
  - Training variants also request logprobs via nested `extra_body`:
    - `{"extras": {"extra_body": {"return_token_ids": True, "logprobs": True, "top_logprobs": 1}}}`
- See:
  - `examples/gsm8k/train_gsm8k.py`
  - `examples/tic_tac_toe/train_tic_tac_toe.py`
  - `examples/fsdp2_training/train_math_fsdp2.py`
  - `examples/pipeline_rl/run_actor.py`
  - `src/ludic/eval/cli.py` (`sampling_args_from_cli`)

### Why the dict approach is brittle *in this repo*
- `extras` is overloaded:
  - “vendor-specific” sampling fields
  - OpenAI request fields for tools (`ToolAgent` injects `tools`, `tool_choice`)
  - vLLM-specific `extra_body` and `vllm_xargs` conventions
- Validation is weak: nesting/typing (`extra_body`, `logprobs`, `top_logprobs`) is easy to misspell or misplace.
- Some logic mutates dicts in non-trivial ways (e.g. `GRPORequestStrategy` editing sampling seeds).

## 1) Goals (what “sensible object” means)

### Functional goals
- Replace dict literals in scripts/examples with typed objects.
- Make inference requests JSON-serializable (needed for pipeline/Redis and logging).
- Remove “hidden mutation” patterns (GRPO seed edits, tool injection via `extras`).

### Research/architecture goals
- Make trace requirements explicit and enforceable:
  - token IDs (prompt + completion) for drift-free RL
  - chosen-token logprobs for ratio objectives (GRPO/PPO-style)
- Move online RL to “token-out by default”:
  - remove retokenization from the online rollout/batching path
  - remove implicit logprob injection in clients
  - keep “teacher-forced backfill” as an explicit opt-in debug path (or delete it)

## 2) Proposal: introduce `ChatCompletionRequest` and split Sampling vs Return vs Vendor

### New “unit of inference”
Introduce a typed request object that represents *one* backend call.

Suggested new file: `src/ludic/inference/request.py` (and re-export from `src/ludic/inference/__init__.py`).

Core type:
- `ChatCompletionRequest` (dataclass):
  - `model: str`
  - `messages: list[Message]` (token-in can come later)
  - `sampling: SamplingParams`
  - `return_: ReturnSpec`
  - `tools: ToolRequest | None`
  - `extensions: BackendExtensions | None` (backend-specific knobs)

This replaces “sampling args dicts” as the thing that gets passed around.

### Split the concerns explicitly
These should be separate dataclasses, not one “extras” dict:

- `SamplingParams` (pure sampling knobs; no vendor fields):
  - `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `stop`
  - optional: `seed` (see §3; recommended to handle seeds in `RolloutRequest` instead)

- `ReturnSpec` (training-critical return payload requirements):
  - `return_token_ids: bool`
  - `return_chosen_logprobs: bool`
  - optional: `top_logprobs_k: int = 1` (if you want top-k; otherwise only chosen-token logprobs)

- `ToolRequest` (OpenAI tool calling surface):
  - `tools: list[dict[str, Any]]` (or a typed schema object later)
  - `tool_choice: str | dict[str, Any] | None`

- `BackendExtensions` (backend-specific extension base type):
  - subclasses live under `ludic.inference.extensions.*`

- `VLLMExtensions` (vLLM-only extensions / generation controls):
  - `max_think: int | None` (your custom vLLM feature; maps to `extra_body["vllm_xargs"]["max_think"]`)
  - `repetition_penalty: float = 1.0` (HF/vLLM ecosystem knob; not OpenAI-standard)
  - optional escape hatch: `extra_body_overrides: dict[str, Any]` (strongly discouraged, but isolated here rather than in `SamplingParams`)

This explicitly avoids forcing “vLLM” into the core request schema: each backend
gets its own typed extension object, and only that backend’s `ChatClient`
implementation interprets it.

### Naming note: why “ReturnSpec” and not “TraceRequest” / “ExtraSpec”
Ludic already uses “trace” terminology at the RL/rollout level (`TraceCollector`, “trajectories”), so calling this `TraceRequest` risks confusion: this field is *not* requesting a rollout trace, it is requesting additional *inference return payload* (token IDs/logprobs) that gets attached to Step.info.

Similarly, avoid `ExtraSpec`: it becomes a junk drawer and recreates today’s `extras` problem. The key separation is:
- `ReturnSpec`: “what should the backend return?” (token IDs / logprobs)
- `BackendExtensions`/`VLLMExtensions`: “how should this specific backend behave?” (e.g. your `max_think` / logits-processor controls)

### Mapping responsibility (important)
Only the backend adapter (`VLLMChatClient`) should know how to map:
- `SamplingParams` → OpenAI kwargs (`temperature`, `max_tokens`, ...)
- `ReturnSpec` / `VLLMExtensions` → vLLM OpenAI-compat (`extra_body`, `logprobs`, `top_logprobs`, `return_token_ids`, `vllm_xargs`)
- `ToolRequest` → OpenAI tool calling fields (`tools`, `tool_choice`)

The rest of the codebase must never manually construct or merge `extra_body` dicts.

### vLLM footgun: HF `generation_config` overriding sampling
vLLM can load model-provided Hugging Face `generation_config` defaults that
silently override sampling parameters (as seen in logs like “Default sampling
parameters have been overridden…”). This is a footgun for research, because the
server’s hidden defaults can drift away from Ludic’s configured `SamplingParams`.

Policy:
- When launching `ludic.inference.vllm_server`, default to `--generation-config vllm`
  unless explicitly overridden by the caller.
- This makes `SamplingParams` (and `VLLMExtensions`) the source of truth.

### Presets (optional, but recommended)
Provide ergonomic constructors that match how examples are written today:
- `ReturnSpec.for_rl()` (token ids on, chosen-logprobs on)
- `ReturnSpec.for_eval()` (token ids on, chosen-logprobs off)
- `SamplingParams.train(temperature=..., max_tokens=...)`

## 3) Breaking changes (recommended “global optimum” cuts)

### 3.1 Delete dict-based sampling args and `extras`
Hard deletion:
- Remove `SamplingArgs` from `src/ludic/types.py`.
- Remove `extras` from `SamplingConfig` in `src/ludic/inference/sampling.py` (or delete `SamplingConfig` entirely and use `SamplingParams`).
- Delete `resolve_sampling_args()` and any code that merges dicts.

Concrete changes:
- Change all `sampling_args` parameters/fields to `request: ChatCompletionRequest` (or `inference: InferenceSpec` that the agent turns into a request).
- Update `ChatClient.complete(...)` signature to accept `ChatCompletionRequest`, not `SamplingConfig`.
  - `src/ludic/inference/client.py`
  - `src/ludic/inference/vllm_client.py`

### 3.2 Split env seed vs sampling seed at the request level
Today:
- env seed is `RolloutRequest.seed`
- sampling seed is a nested `sampling_args["seed"]`
and GRPO mutates that nested seed.

Replace with explicit fields on `RolloutRequest`:
- `env_seed: int | None`
- `sampling_seed: int | None`

Then:
- protocols pass `sampling_seed` to the agent call
- the agent sets OpenAI `seed` from `sampling_seed`
- `GRPORequestStrategy` edits `sampling_seed` directly (no introspection, no nested edits)

This is a breaking change, but it removes a large class of brittleness.

### 3.3 Make online RL token-out by default
Remove retokenization from the online rollout/batching path:
- In `RolloutEngine.generate_batch` and `RolloutBatchSource`, delete `retokenize` and `tokenize`.
- Require `prompt_token_ids` and `completion_token_ids` in `Step.info` for online batches.
- If the algorithm needs ratio objectives, also require chosen-token logprobs in `Step.info` (via `ReturnSpec.return_chosen_logprobs=True`).

Keep retokenization only in offline/SFT codepaths (`OfflineBatchSource`), where text-only datasets are expected.

## 4) Refactors to support the new request surface

### 4.1 GRPO request expansion
`src/ludic/training/batching/intra_batch_control.py`:
- Expand groups by editing `RolloutRequest.sampling_seed` and `RolloutRequest.env_seed` directly.
- (Optionally) attach a `group_id` to request meta, as today.

### 4.2 ToolAgent / ReActAgent
`src/ludic/agents/tool_agent.py`, `src/ludic/agents/react_agent.py`:
- Stop mutating “sampling extras”.
- Instead, build/modify a `ChatCompletionRequest` (or a smaller `InferenceSpec`) by setting:
  - `tools=ToolRequest(...)` to enable tool calling
  - `tools=None` to disable tool calling (shot clock)
  - `return_.return_token_ids=True` forced on whenever tools are enabled (so training stays drift-free)

## 5) Migrate examples + CLI helpers (stop writing dicts)

### Examples
Replace dict literals with typed `SamplingParams` + `ReturnSpec` (or a single `InferenceSpec`) in:
- `examples/gsm8k/train_gsm8k.py`:
  - training: needs token ids + chosen-token logprobs
  - eval: needs token ids
- `examples/fsdp2_training/train_math_fsdp2.py` (same pattern)
- `examples/tic_tac_toe/train_tic_tac_toe.py`:
  - token ids (logprobs optional depending on algorithm)
- `examples/pipeline_rl/run_actor.py`:
  - token ids

### Eval CLI
`src/ludic/eval/cli.py`:
- `sampling_args_from_cli()` should return typed objects (or an `InferenceSpec`), not a dict.

## 6) Testing plan

### Tests to add/adjust
- Request mapping tests for `VLLMChatClient`:
  - `ChatCompletionRequest(trace=..., vendor=...)` produces correct OpenAI kwargs
  - no implicit injection (e.g. logprobs) when not requested
- GRPO seed expansion edits `RolloutRequest.sampling_seed` (see `tests/test_requests_helpers.py`)
- Tool injection/removal works via typed request objects (update `tests/test_tool_agent.py`, `tests/test_react_agent.py`)
- Online batching requires token IDs (update `tests/test_rollout_engine.py` accordingly)

## 7) Roadmap: “token IDs + rollout-time logprobs” first-class (no retokenization/backfill)

### Near term: make the correct thing easy
- Encode trace requirements in typed presets:
  - `ReturnSpec.for_rl()` turns on token IDs + chosen-token logprobs
  - `ReturnSpec.for_eval()` turns on token IDs only
- Avoid implicit behavior in clients:
  - vLLM logprob/token-id requests are driven only by `ReturnSpec`

### Medium term: tighten invariants
- For ratio-based objectives:
  - missing behavior logprobs should be a hard error by default.
  - teacher-forced backfill (`src/ludic/training/algorithm.py`) becomes an explicit debug/research option, not the default.
- Online RL: fail early if token ids are missing (and if needed, logprobs missing).

### Long term: token-in / token-out API
- Introduce an explicit tokenized client path where the backend returns:
  - prompt token IDs
  - completion token IDs
  - chosen-token logprobs
- Extend offline JSONL formats to optionally store tokenized artifacts so offline training can also avoid retokenization when data was generated from a compatible runtime.

## 8) Acceptance criteria (definition of done)
- No dict-based `sampling_args` remain in the repo.
- `ChatClient.complete()` accepts a typed request object, and only backend adapters map it to vendor kwargs.
- `RolloutRequest` carries explicit `env_seed` and `sampling_seed`; GRPO expansion edits those directly.
- Online RL batching requires token IDs (and, when required by the algo, rollout-time chosen-token logprobs).
- Tool usage is expressed via typed request fields, not `extras` mutations.
- Tests cover:
  - request→vendor kwargs mapping
  - GRPO seed expansion
  - tool injection behavior
  - online batching token-id requirement
