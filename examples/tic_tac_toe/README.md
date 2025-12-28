# Tic-Tac-Toe GRPO + LoRA

Small end-to-end RL run that fine-tunes a model with LoRA on a toy Tic-Tac-Toe env using grouped advantages (GRPO-style). Assumes 2 GPUs: GPU0 for vLLM inference, GPU1 for training.

## Pre-trained Model

We provide [`hallerite/Qwen2.5-7B-TTT`](https://huggingface.co/hallerite/Qwen2.5-7B-TTT), a cold-start checkpoint created via:

1. **Rejection sampling**: Used `generate_synth_data.py` to generate episodes with `Qwen/Qwen2.5-7B-Instruct`, filtered to 1124 winning trajectories (see `data/tictactoe_sft_train_data.jsonl`)
2. **SFT**: Trained for 25 steps with effective batch size 32 (batch_size per rank × world_size)

This model understands the `<think>...</think><move>...</move>` format and the `[TRUNCATED]` convention used by `TruncatedThinkingContext`.

## Quick Start (RL from Pre-trained)

### 1) Start vLLM (GPU0)
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python -m ludic.inference.vllm_server \
  --model hallerite/Qwen2.5-7B-TTT
```

### 2) Train (GPU1)
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/tic_tac_toe/train_tic_tac_toe.py \
  --ctx truncated \
  --micro-token-budget 8192 \
  --max-seq-len 1024
```
- Uses `GRPORequestStrategy` + `GroupNormalizedReturn`, LoRA on the HF model, and a strict `<think>...</think><move>...</move>` parser.
- Samples 50/50 agent_starts=True/False so the policy practices both as first and second player.
- Use `--final-save` to force a final checkpoint at the end of training (in addition to periodic saves).
- Tweak `--group-size`, `--concurrency`, `--train-steps`, `--train-temperature`, `--rollouts-per-update`, `--max-steps-per-episode`, `--max-seq-len`, and `--micro-token-budget` as needed.
- Logger options: `--logger rich`, `--logger wandb`, or `--logger rich,wandb` (W&B uses `WANDB_*` env vars).

### 3) Evaluate (optional)
```bash
PYTHONPATH=. uv run python examples/tic_tac_toe/eval_tic_tac_toe_vllm.py \
  --model hallerite/Qwen2.5-7B-TTT \
  --start-server \
  --episodes 200 \
  --ctx truncated
```
- Reports win/loss/draw/illegal/parse-error rates; writes `tictactoe_eval.jsonl` by default.

## Tinker Backend (Serverless RL)

Run the Tinker-backed training loop (no local vLLM required):

```bash
TINKER_API_KEY=... WANDB_API_KEY=... PYTHONPATH=. uv run python \
  examples/tic_tac_toe/train_tic_tac_toe_tinker.py \
  --train-steps 20 \
  --wandb-name ttt-tinker
```

Use `--base-url` if you need to point at a custom Tinker endpoint.

### Using a LoRA adapter with vLLM
If your policy is a LoRA adapter (for example `hallerite/Qwen2.5-7B-TTT-RL`), start vLLM with the base model and load the adapter explicitly. The `--start-server` helper does not pass LoRA flags, so launch the server yourself:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python -m ludic.inference.vllm_server \
  --model hallerite/Qwen2.5-7B-TTT \
  --enable-lora \
  --lora-modules ttt_rl=hallerite/Qwen2.5-7B-TTT-RL
```
Then point eval at the adapter name you chose (`ttt_rl` here) and do **not** use `--start-server`:
```bash
PYTHONPATH=. uv run python examples/tic_tac_toe/eval_tic_tac_toe_vllm.py \
  --model ttt_rl \
  --episodes 200 \
  --ctx truncated
```
Notes:
- You can name the adapter whatever you like (`ttt_rl=`); just pass the same name via `--model`.
- Local adapters also work: `--lora-modules ttt_rl=/path/to/adapter`.

---

## Training from Scratch (Cold-Start with SFT)

If you want to create your own cold-start checkpoint from `Qwen/Qwen2.5-7B-Instruct`:

### 1) Generate synthetic data
```bash
PYTHONPATH=. uv run python examples/tic_tac_toe/generate_synth_data.py \
  --episodes 5000 \
  --output data/tictactoe_sft_train_data.jsonl
```
- Generates rollouts using `FullDialog` (model sees full thinking history)
- Filters to keep only wins
- Transforms `prev_obs` to have `[TRUNCATED]` placeholders (simulates what RL will show)
- Action (model's response) kept intact

Use `--no-transform` to keep raw format (for SFT without truncation).

### 2) SFT on the data
```bash
PYTHONPATH=. uv run torchrun --nproc_per_node=2 examples/tic_tac_toe/sft_tic_tac_toe.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data data/tictactoe_sft_train_data.jsonl \
  --epochs 1 \
  --batch-size 4 \
  --micro-token-budget 8192 \
  --max-seq-len 1024
```
- Uses `OfflineBatchSource` + `make_sft()` (uniform weight NLL)
- Checkpoints saved to `checkpoints_tictactoe_fsdp2/`

### 3) Push to Hub (optional)
```bash
python scripts/push_to_hub.py \
  checkpoints_tictactoe_fsdp2/step_000025 \
  your-username/your-model-name \
  --base-model Qwen/Qwen2.5-7B-Instruct
```

### 4) RL with TruncatedThinkingContext
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/tic_tac_toe/train_tic_tac_toe.py \
  --model your-username/your-model-name \
  --ctx truncated
```
- Model now understands the `[TRUNCATED]` convention
- RL refines strategy while keeping context compact

---

## Files

| File | Purpose |
|------|---------|
| `train_tic_tac_toe.py` | Online RL training (GRPO + LoRA) |
| `eval_tic_tac_toe_vllm.py` | Evaluate win/loss/draw rates |
| `generate_synth_data.py` | Generate & filter winning trajectories for SFT |
| `sft_tic_tac_toe.py` | Offline SFT on pre-generated data (FSDP2) |

## Context Strategies

- `--ctx full` (default): `FullDialog` - model sees complete history including full `<think>` blocks
- `--ctx truncated`: `TruncatedThinkingContext` - previous thinking replaced with `[TRUNCATED]`, saves context length

Use `--ctx truncated` after SFT cold-start for best results.
