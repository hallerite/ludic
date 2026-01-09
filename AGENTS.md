# Ludic (Agent Notes)

Ludic is an LLM-RL *library* built to be hackable. The core idea is to keep components loosely coupled so you can swap pieces (envs, prompts, context/memory, parsing, rollout generation, batching, credit assignment, losses, distributed weight pushing) without rewriting the whole stack.

This repo is *research scaffolding*, not production infrastructure.

## What This Is *Not* (Avoid RLHF Mental Models)

Ludic is **not** primarily an “RLHF framework” in the common 2023–2024 sense (preference optimization / alignment pipelines).

- **Not DPO-style training**: the default mental model is *not* “optimize preferences directly against a reference policy” (DPO/IPO/KTO/etc.).
- **Not the classical RLHF stack**: it is *not* organized around “SFT → train reward model → PPO with KL-to-reference”.

Instead, Ludic is closer to **classical RL** – specifically policy-gradient methods – where:
- The LLM is the **policy**.
- The environment (plus verifiers/judges/parsers) provides **rewards**.
- “Algorithms” are expressed as **credit assignment** (how you turn rewards into per-step weights) + a **loss** (how you turn weights + tokens into gradients).

## Design Intent (Condensed)

- **Separate Agent vs Environment**
  - **Environment** = state transition function (+ optional scalar reward) with minimal assumptions; can be multi-agent by default.
  - **Agent** = LLM *with state* (prompt harness + memory + parsing + optional tools).
    - Internal tools = executed by agent (calculator, code interpreter); don't change env state
    - External tools = returned to protocol for handling (delegation, sub-agents)
  - Rationale: reuse environments across different "agent harnesses" (memory schemes, parsers, prompts, tools) and reuse harness pieces across environments.

- **Make the interaction loop explicit**
  - Neither env nor agent “owns” rollout generation. An **InteractionProtocol** owns the agent<-->env loop and produces rollouts.
  - This makes rollouts modifiable (single-agent, turn-based, multi-agent, tool-augmented, etc.) without entangling env/agent internals.

- **Decouple inference from training**
  - Training should not block on rollout latency.
  - The Trainer consumes *batches of training-ready samples* from a `BatchSource`, not raw env episodes.

- **One Trainer, many algorithms**
  - The Trainer is just an optimization loop.
  - “Algorithm” = (credit assignment → weights) + (loss on tensors). No `PPOTrainer`/`GRPOTrainer` class explosion.

## Mental Model: Data Flow

1. **Intent / Curriculum** → list of `RolloutRequest` (pure data: which env+protocol to run, seeds, sampling args, metadata).
2. **Rollout execution** (`RolloutEngine`) → `Rollout` objects (conversation-like trajectories of `Step`s).
3. **Credit assignment** (`CreditAssigner`) → per-step scalar `weight`.
4. **Training samples** → `SAWItem` = **S**tate/**A**ction tokens + masks + **W**eight (+ metadata).
5. **BatchSource** supplies `SAWBatch` to the Trainer:
   - **sync**: trainer blocks while rollouts are generated
   - **pipeline**: actor process pushes `SAWItem`s into Redis; learner blocks on Redis pop (optionally tags each item with `policy_version` so the learner can drop stale experience via `TrainerConfig.max_lag`)
6. **Trainer** collates `SAWItem`s into dense tensors, runs `RLAlgorithm.loss`, steps optimizer, then publishes weights to inference runtime.

## Core Abstractions (Where + What)

- **Shared types (rollouts, steps, truncation flags)**: `src/ludic/types.py`
- **Steps (agent vs env)**: See `CONSIDERATIONS.md` for the full rationale. The short version:
  - **AgentStep**: Every model call, including internal tool loops. Contains `TokenTrace` for training.
  - **EnvironmentStep**: State transitions (`env.step()` outcomes). References the triggering AgentSteps.
  - Why separate? Training needs the full reasoning trace, not just final actions. A ReAct agent might call 3 tools before outputting an action—all those calls have token traces we want to train on.
  - Rollouts keep a single timeline of both kinds; online batching concatenates all AgentSteps in a turn into one `SAWItem`.

- **Environment kernel (multi-agent by default)**: `src/ludic/envs/env.py`
  - `LudicEnv.reset() -> {agent_id: (obs, info)}`
  - `LudicEnv.step(actions) -> {agent_id: StepOutcome}`
  - `active_agents` determines which agents should act on a step.
  - Convenience for common "Gym-like" cases: `src/ludic/envs/single_agent_env.py` (`SingleAgentEnv`).
  - Dataset-backed QA env: `src/ludic/envs/dataset_qa_env.py` (`DatasetQAEnv`) – one-shot QA tasks on dataset samples with custom verifiers.

- **Agent = LLM + state**: `src/ludic/agents/base_agent.py`
  - Wraps a `ChatClient` (inference backend), a `ContextStrategy` (memory/prompt building), and a `Parser` (action decoding + intrinsic format rewards/penalties).
  - Handles incomplete completions (`finish_reason == "length"`) as parse failures (optional) to avoid training on truncated actions.
  - Extended agent types:
    - `ToolAgent` (`src/ludic/agents/tool_agent.py`): Base for tool-calling agents. Supports two tool scopes:
      - `tools`: Internal tools executed by agent (calculator, code interpreter). Results go to context, agent continues.
      - `external_tools`: Tools returned to protocol for handling (delegation, sub-agents). Protocol feeds results back.
    - `ReActAgent` (`src/ludic/agents/react_agent.py`): Multi-step ReAct pattern [Think → Tool]* → Act. Returns `action_target` indicating what happens next: `"internal"` (handled), `"external"` (protocol handles), or `"env"` (final action).

- **Context strategy (memory/prompt policy)**: `src/ludic/context/base.py`
  - Hooks: `on_env_reset`, `on_before_act`, `on_after_act`, `on_after_step`.
  - Implementations:
    - `src/ludic/context/full_dialog.py` (`FullDialog`): Simple strategy that keeps entire dialog history.
    - `src/ludic/context/truncated_thinking.py` (`TruncatedThinkingContext`): Truncates `<think>...</think>` blocks in the *prompt* while keeping full history internally.

- **Parser (semantic action extraction + format reward)**: `src/ludic/parsers.py`
  - Returns `ParseResult(action | None, reward, obs)`; failures can yield synthetic feedback observations.
  - Parsers compose (`compose_parsers`) so you can enforce multiple formatting contracts.

- **Interaction protocols (own the loop)**: `src/ludic/interaction/base.py`
  - Single-agent synchronous loop: `src/ludic/interaction/single_agent.py`
    - Supports `external_tool_handler` callback for handling external tool calls
  - Multi-agent loop (per-agent rollouts via `TraceCollector`): `src/ludic/interaction/multi_agent.py`, `src/ludic/interaction/step_collector.py`
  - Key behaviors:
    - Parser failures are handled *inside the protocol* (synthetic step, no `env.step()` call), so env stays parser-agnostic.
    - External tool calls (`action_target="external"`) are routed through `external_tool_handler`; results are fed back to agent context and the agent continues reasoning.
  - **Delegation pattern**: External tools enable hierarchical agents where a parent can spawn sub-agents. The protocol handles the sub-agent's rollout and returns results to the parent. Both rollouts are collected for training. See `CONSIDERATIONS.md` for details.
  - Utility: `src/ludic/interaction/info.py` provides `merge_step_info()` for safely merging step metadata with collision detection on reserved keys.

- **Rollout execution + collation**: `src/ludic/training/batching/rollout_engine.py`
  - Stateless “factory floor”: instantiates env + protocol per request, runs episodes concurrently, returns rollouts.
  - Converts rollouts → `SAWItem`s using either:
    - exact token IDs returned by the inference backend (preferred), or
    - `retokenize=True` with a caller-provided tokenizer.
  - Practical note: Token-in mode (see README) ensures drift-free RL by using rollout-time token IDs directly. Use `ReturnSpec.for_rl()` or set `return_token_ids=True` in `InferenceSpec` to get token IDs from the backend.

- **Batch sources (trainer talks to these, not the engine)**: `src/ludic/training/types.py`
  - Sync: `src/ludic/training/batching/synced_batching.py` (`RolloutBatchSource`)
  - Offline/SFT: `src/ludic/training/batching/offline.py` (`OfflineBatchSource`) – reads pre-collected rollouts from JSONL for SFT/offline RL.
  - Pipeline/Redis: `src/ludic/training/batching/pipeline.py` (`PipelineBatchSource`, `run_pipeline_actor`)
  - Curriculum builder: `src/ludic/training/batching/requests_from_dataset.py` (`make_requests_fn_from_queue()`) builds rollout requests from queued dataset items.

- **Algorithm injection (credit + loss)**: `src/ludic/training/algorithm.py`
  - `RLAlgorithm = (CreditAssigner, Loss)`
  - Presets: `make_reinforce()`, `make_reinforce_baseline()`, `make_grpo()`, `make_dr_grpo()`, `make_gspo()`, `make_cispo()`, `make_gmpo()`, `make_sft()`
  - Credit assigners: `src/ludic/training/credit_assignment.py` – `MonteCarloReturn`, `GroupNormalizedReturn`, `EpisodicReturn`, `PerStepReward`, `ConstantCredit`
  - Losses: `src/ludic/training/loss.py` – `ReinforceLoss`, `TokenClippedSurrogateLoss`, `ClippedSurrogateLoss`, `CISPOLoss`, `GMPOLoss`, `MaskedCausalLMCrossEntropyLoss`

- **Trainer (optimization loop only)**: `src/ludic/training/trainer.py`
  - Collates `SAWItem` → tensors and runs `RLAlgorithm.loss`.
  - Publishes weights via `PolicyPublisher` (`src/ludic/distributed/interfaces.py`).

- **Training utilities**:
  - Configuration: `src/ludic/training/config.py` (`TrainerConfig`) – optimization, device placement, gradient accumulation settings.
  - Checkpointing: `src/ludic/training/checkpoint.py` (`CheckpointConfig`, `CheckpointManager`) – periodic checkpoint saving with configurable retention.
  - Sample filters: `src/ludic/training/filters.py` – `drop_truncated()`, `drop_incomplete_completions()`, `drop_parse_errors()`, `combine()` for filtering training samples.
  - Metrics: `src/ludic/training/stats.py` (`Reducer`, `aggregate_stats()`) – declarative metrics aggregation with dotted path source lookup.
  - Logging: `src/ludic/training/loggers.py` (`TrainingLogger` protocol, `PrintLogger`) – console logging with rich formatting.

- **Inference backends / weight pushing**
  - `ChatClient` contract: `src/ludic/inference/client.py`
  - vLLM client (OpenAI-compatible API + optional NCCL weight updates): `src/ludic/inference/vllm_client.py`
  - vLLM server with weight-update endpoints + "max_think" logits processor: `src/ludic/inference/vllm_server.py`
  - vLLM utilities: `src/ludic/inference/vllm_utils.py` – `start_vllm_server()`, `wait_for_vllm_health()` for process management.
  - Sampling config: `src/ludic/inference/sampling.py` (`SamplingConfig`, `resolve_sampling_args()`) – fully-resolved sampling parameters with OpenAI mapping.
  - vLLM publisher adapter: `src/ludic/distributed/adapters/vllm.py`

## GRPO in Ludic (Sampling Strategy + Credit Assignment)

Ludic treats GRPO as an intra-batch strategy rather than a separate trainer:

- **Request expansion**: `src/ludic/training/batching/intra_batch_control.py`
  - `GRPORequestStrategy(group_size=G)` expands 1 logical request into `G` executions:
    - same environment seed (same problem)
    - different sampling seeds (diverse generations)
- **Group-normalized advantage**: `src/ludic/training/credit_assignment.py`
  - `GroupNormalizedReturn` groups rollouts by the *initial observation* and assigns `A = R - mean(R_group)` to every step.

GRPO mental model in this codebase:
- It's still **policy-gradient** training on sampled tokens.
- It avoids a learned **value function** by using a **Monte Carlo / group-relative baseline** (group mean reward for the same prompt) to form advantages.
- If you come from PPO-RLHF: think "PPO-shaped dataflow" without a critic/value model, where the "advantage" is estimated by group comparison rather than by GAE/value bootstrapping.

## GMPO (Geometric-Mean Policy Optimization)

**GMPO** (arXiv:2507.20673) is a variant of GRPO that uses the **geometric mean** of token-level importance ratios instead of the arithmetic mean.

**Core idea**:
- GRPO optimizes: (1/|o|) Σ_t ρ_t * A (arithmetic mean)
- GMPO optimizes: (∏_t ρ_t)^(1/|o|) * A (geometric mean)

The geometric mean is less sensitive to outlier importance ratios, which can help prevent extreme policy updates when individual tokens have unusually high or low ratios.

**Implementation** (`src/ludic/training/loss.py`, `src/ludic/training/algorithm.py`):
- **Loss**: `GMPOLoss` computes the geometric mean in log-space for numerical stability
- **Objective**: J_GMPO = E[ (∏_t min(ρ_t * A, clip(ρ_t, e^-ε_low, e^ε_high) * A))^(1/|o|) * sgn(A) ]
- **Clipping**: Token-level clipping in log-space, wider default range (e^-0.4, e^0.4) vs GRPO's (0.8, 1.2)
- **Normalization**: 1/|o| sequence length normalization
- **Preset**: `make_gmpo(group_size=4)` uses same credit assignment as GRPO (`GroupNormalizedReturn`)

## SFT / Offline RL

Ludic supports supervised fine-tuning (SFT) and offline RL through the same abstractions:

- **SFT as offline RL**: SFT is just offline RL with `weight=1.0` for all samples.
  - Use `make_sft()` which pairs `ConstantCredit(value=1.0)` with `ReinforceLoss`.
  - With uniform weights, REINFORCE loss reduces to standard NLL on action tokens.

- **OfflineBatchSource**: `src/ludic/training/batching/offline.py`
  - Reads rollouts from JSONL files (e.g., output of rejection sampling).
  - Tokenizes state/action pairs and applies credit assignment.
  - Supports shuffling, batching, and sample filtering.

- **Workflow** (cold-start training):
  1. Generate rollouts with `RolloutEngine` or `rejection_sampling.py`.
  2. Filter to keep successful trajectories (e.g., wins only).
  3. Train with `OfflineBatchSource` + `make_sft()` to bootstrap the model.
  4. Continue with online RL (`RolloutBatchSource` + `make_grpo()`) for refinement.

- **AWR-style offline RL**: For advantage-weighted regression, wrap a credit assigner:
  ```python
  # AWR: weight = exp(advantage / temperature)
  base_credit = GroupNormalizedReturn(group_size=G)
  # Then transform weights via exp(w / temp) in a custom assigner
  ```

## Evaluation System

Ludic includes an evaluation subsystem for assessing trained policies:

- **Core evaluation**: `src/ludic/eval/core.py`
  - `run_eval()`: Main evaluation function using RolloutEngine.
  - `_eval_items_from_rollout()`: Converts rollouts to evaluation items with metrics.
- **Evaluator protocol**: `src/ludic/eval/evaluator.py`
  - `Evaluator`: Protocol for evaluation orchestration.
  - `EngineEvaluator`: Implementation backed by RolloutEngine + `Reducer`s for metric aggregation.
- **CLI utilities**: `src/ludic/eval/cli.py`
  - `add_common_eval_args()`, `add_common_start_server_arg()`: Shared argument parsing for eval scripts.

## Distributed Policy Publishing

- **BroadcastPolicyPublisher**: `src/ludic/distributed/publisher.py`
  - Two-plane weight distribution: control plane (HTTP) coordinates update timing, data plane (NCCL) broadcasts tensors.
  - Used for efficient weight updates to vLLM inference workers during training.

## Truncation Semantics (Important)

Ludic distinguishes:
- `terminated`: true terminal from the env
- `truncated`: time limit / cutoff (env-initiated or protocol max-steps)
- `finish_reason`: LLM generation stop reason (separate from env semantics)

See `CONSIDERATIONS.md` for the canonical definitions and what gets propagated into training metadata.

## Examples (Most Useful Entry Points)

- GSM8K training/eval: `examples/gsm8k/README.md`, `examples/gsm8k/train_gsm8k.py`, `examples/gsm8k/eval_gsm8k_vllm.py`
- FSDP2 + vLLM serving: `examples/fsdp2_training/README.md`, `examples/fsdp2_training/train_math_fsdp2.py`
- Pipeline RL (actor/learner split via Redis): `examples/pipeline_rl/run_actor.py`, `examples/pipeline_rl/run_trainer.py`
- TicTacToe single-agent scaffold: `examples/tic_tac_toe/train_tic_tac_toe.py`, `examples/tic_tac_toe/eval_tic_tac_toe_vllm.py`
- Rejection sampling (data generation): `examples/rejection_sampling.py`

Some older scripts in `examples/` may be stale (imports/layout changed); prefer the README-backed examples above as the "current" patterns.

## Extending Ludic (Practical Checklist)

- New task:
  - Implement an env (`SingleAgentEnv` is the easiest path).
  - Decide on output contract and implement a `Parser` (or compose existing ones).
  - Choose a `ContextStrategy` (or write one).
  - Pick/write an `InteractionProtocol` if the default loops aren’t enough.
  - Register env + protocol factories with a `RolloutEngine`.
  - Choose `BatchSource` (sync for debugging; pipeline for throughput).
  - Choose `CreditAssigner` + `Loss` (assemble an `RLAlgorithm`).
