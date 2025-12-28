# Tinker Integration Notes

This doc sketches how to integrate Ludic envs with Tinker as the training and
inference backend. The goal is on-policy RL where Ludic generates rollouts and
Tinker performs sampling + optimization.

## Scope

- Use Ludic for: envs, agents, parsers, interaction protocols, rollout generation.
- Use Tinker for: sampling, training (forward/backward + optim step), checkpointing.
- Keep integration minimal and data-centric: translate Ludic rollouts into
  Tinker `Datum` objects and let Tinker own optimization.

## Proposed Data Flow (On-Policy)

1) Build a `TinkerChatClient` for Ludic agents.
2) Run Ludic `RolloutEngine` with `ReturnSpec.for_rl()` so each `Step` contains
   `TokenTrace` with prompt/completion token IDs + chosen-token logprobs.
3) Use Ludic `CreditAssigner` to compute per-step weights/advantages.
4) Convert rollouts to Tinker `Datum` objects (token-level targets/logprobs/advantages).
5) Call `training_client.forward_backward(...)` and `training_client.optim_step(...)`.
6) Refresh sampling client via `save_weights_and_get_sampling_client(...)` and
   update `TinkerChatClient` so next rollouts are strictly on-policy.

## Quickstart (Tic-Tac-Toe)

Set your Tinker API key (and optionally W&B), then run the example:

```bash
TINKER_API_KEY=... WANDB_API_KEY=... PYTHONPATH=. uv run python \
  examples/tic_tac_toe/train_tic_tac_toe_tinker.py \
  --train-steps 20 \
  --wandb-name ttt-tinker
```

Use `--base-url` to target a non-default Tinker service endpoint.

## Integration Surface

```
integrations/
  tinker/
    client.py                # TinkerChatClient adapter (Ludic ChatClient)
    rollouts_to_datums.py    # Ludic Rollout -> Tinker Datum conversion
```

## TinkerChatClient Adapter

Implement `ludic.inference.client.ChatClient` (and optionally `VersionedClient`)
so Ludic protocols can sample from Tinker.

Responsibilities:
- Accept a `TokenCompletionRequest` containing pre-tokenized prompt IDs.
  - The Agent applies the chat template; the Tinker client stays token-in.
- Call `sampling_client.sample_async(...)` with `tinker.SamplingParams`.
  - Must request chosen-token logprobs for RL.
- Produce `ChatResponse` with:
  - `text` (decoded completion for parsing)
  - `prompt_token_ids`, `completion_token_ids`, `completion_logprobs`
  - `finish_reason` if available
- Attach a `TokenTrace` to each `Step` via the normal Ludic flow.

Notes:
- Ludic applies chat templates client-side and calls completions with token IDs,
  so Tinker stays a pure token-in/token-out backend.
- Tinker does not expose OpenAI tool-calling endpoints. Tool use should be
  encoded into the prompt if needed.

## Rollout -> Datum Conversion

Tinker training expects `Datum` with `target_tokens`, `logprobs`, and
`advantages`. Ludic provides prompt/completion tokens and per-step weights.

Rules:
- `input_ids = prompt_ids + completion_ids`
- `target_tokens = input_ids[1:]`
- `logprobs = [0.0]*prompt_len + completion_logprobs`
- `advantages = [0.0]*prompt_len + [weight]*completion_len`

For multi-step trajectories, prefer the Tinker cookbook logic in
`tinker_cookbook/rl/data_processing.py` (prefix-merge with `trajectory_to_data`)
to avoid fragmenting sequences when observations grow by appending.

Edge cases:
- If a step does not contain token IDs or logprobs, drop the sample or fail fast.
- If `completion_logprobs` length mismatches completion tokens, fail fast.
- If a step is truncated or a parse error occurred, consider dropping it using
  Ludic filters (see `src/ludic/training/filters.py`).

## Credit Assignment / Grouping

Two equivalent options:

1) Keep Ludic credit assigners (e.g., `GroupNormalizedReturn`) and treat the
   per-step weight as the per-token advantage.
2) Group rollouts by initial observation in the converter and compute group
   advantages in the Tinker-style (subtract group mean).

Ludic already supports GRPO-style intra-batch control. That can be reused to
generate groups with shared env seeds and distinct sampling seeds.

## On-Policy Guarantees

To stay on-policy:
- After each optimizer step, call
  `training_client.save_weights_and_get_sampling_client(...)`.
- Swap the `sampling_client` inside `TinkerChatClient` before the next rollout.
- Optionally tag rollouts with a `policy_version` to detect staleness if using
  any async/pipelined collection.

## Logging and Metrics

You can forward Ludic rollout metadata into Tinker metrics:
- `rollout.total_reward`, `step.reward`, `truncated`, `terminated`
- Token counts from `TokenTrace` (prompt/completion lengths)

Tinker cookbook utilities (`tinker_cookbook/utils/ml_log.py`) can be used for
metric logging; just ensure timestamps/step counters align with Ludic batches.

## Known Constraints

- Tool calling: Tinker sampling is token-based; OpenAI-style tool-calling APIs
  are not available. Use ReAct or serialized tool calls in text.
- Multi-agent: Tinker RL envs are single-agent; map each Ludic agent rollout to
  a separate trajectory and group rewards if needed.
- Truncation semantics: decide whether to drop or keep truncated steps; Ludic
  tracks `terminated`, `truncated`, and `finish_reason` separately.
- Exact token alignment: renderer/tokenizer must be identical between sampling
  and training, or logprob alignment breaks.

## Next Steps

1) Implement `integrations/tinker/client.py` with a `TinkerChatClient`.
2) Port `trajectory_to_data(...)` logic into `rollouts_to_datums.py` for safe
   prefix-merge behavior.
3) Create a minimal on-policy loop (`on_policy_loop.py`) that:
   - calls `RolloutEngine.generate_rollouts(...)`
   - converts rollouts to `Datum`
   - trains via `training_client.forward_backward(...)` + `optim_step(...)`
   - refreshes sampling client
4) Add a small example that wires a Ludic env to a Tinker training run.
