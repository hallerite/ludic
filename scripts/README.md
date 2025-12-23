# Scripts

## calibrate_micro_batch.py

Calibrate micro-batch token budgets against a real model on GPU. The script
builds a synthetic SAW batch and runs a forward+backward+step loop to measure
peak memory and speed for each micro-token budget.

### Quick start (auto sweep)

By default, the script sweeps budgets starting at `max_seq_len`, doubles until
it finds an OOM (or reaches `micro_batches == 1`), then binary-searches the
last OK / first OOM range unless the OOM budget would not reduce micro-batches.

Details:
- Exponential phase: start at `max_seq_len`, then multiply budget by 2 each step.
- Binary search phase: if an OOM is found and micro-batch count can still drop,
  search the integer interval `[last_ok, first_oom)` to find the highest OK budget.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python scripts/calibrate_micro_batch.py \
  --model /path/to/model \
  --dtype bf16 \
  --max-seq-len 1024 \
  --target-rollouts 64 \
  --gradient-checkpointing
```

Auto mode prints summary lines:

- `min_micro_token_budget_ok_pow2`: lowest successful budget from the doubling phase.
- `max_micro_token_budget_ok`: highest successful budget found via binary search.
- `oom_micro_token_budget` / `oom_error`: the first failing budget and error.

### Manual budgets

Use explicit budgets to compare specific settings.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python scripts/calibrate_micro_batch.py \
  --model /path/to/model \
  --dtype bf16 \
  --max-seq-len 1024 \
  --target-rollouts 64 \
  --token-budgets 8192,12288,16384,24576
```

You can also pass a single budget:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python scripts/calibrate_micro_batch.py \
  --model /path/to/model \
  --dtype bf16 \
  --max-seq-len 1024 \
  --target-rollouts 64 \
  --micro-token-budget 12288
```

### LoRA support

Two options: load an existing adapter or create one on the fly.

Load a saved adapter:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python scripts/calibrate_micro_batch.py \
  --model /path/to/base \
  --lora-path /path/to/adapter_dir \
  --dtype bf16 \
  --max-seq-len 1024 \
  --target-rollouts 64
```

Create a LoRA adapter on the fly:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python scripts/calibrate_micro_batch.py \
  --model /path/to/base \
  --lora-rank 8 \
  --lora-alpha-mult 2.0 \
  --lora-dropout 0.0 \
  --lora-target-modules all-linear \
  --dtype bf16 \
  --max-seq-len 1024 \
  --target-rollouts 64
```

Notes:
- LoRA requires `peft` (`uv sync --extra examples`).
- `--lora-path` and `--lora-rank` are mutually exclusive.

### FSDP

For FSDP, launch with torchrun:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 scripts/calibrate_micro_batch.py \
  --model /path/to/model \
  --dtype bf16 \
  --max-seq-len 1024 \
  --target-rollouts 64 \
  --fsdp
```
