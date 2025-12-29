# On-Policy Distillation (OPD) Training on GSM8K

Train a smaller student model using dense per-token supervision from a larger teacher model.

OPD combines the benefits of:
- **On-policy learning**: Student samples from itself (not teacher demonstrations)
- **Dense supervision**: Per-token feedback via reverse KL divergence (not sparse rewards)

Reference: https://thinkingmachines.ai/blog/on-policy-distillation

## Prerequisites

- At least 2 GPUs (e.g., 2x A100).
  - GPU 0: Both vLLM servers (student 0.5B + teacher 7B fit together)
  - GPU 1: Training (gradient updates)
- Required extra packages: `datasets`, `math-verify`.

Install deps (once):
```bash
uv sync --extra examples
```

## 1) Start vLLM servers

You need **two** vLLM servers: one for the student (sampling) and one for the teacher (scoring). For these small models, both can share GPU 0.

**Important**: Student and teacher must use the **same tokenizer**. The Qwen2.5 family shares tokenizers across sizes, so this works.

### Terminal 1: Student server (port 8000)
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.4
```

### Terminal 2: Teacher server (port 8001)
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8001 \
  --gpu-memory-utilization 0.5
```

Wait for both servers to report ready before proceeding.

## 2) Train with OPD

In a third terminal, run the OPD training script on GPU 1:
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/opd/train_opd_gsm8k.py \
  --student-model Qwen/Qwen2.5-0.5B-Instruct \
  --teacher-model Qwen/Qwen2.5-7B-Instruct \
  --student-port 8000 \
  --teacher-port 8001 \
  --rollouts-per-update 64 \
  --train-steps 100 \
  --micro-token-budget 16384 \
  --max-seq-len 1024
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--student-model` | `Qwen/Qwen2.5-0.5B-Instruct` | Student model (must match vLLM server) |
| `--teacher-model` | `Qwen/Qwen2.5-7B-Instruct` | Teacher model (must share tokenizer with student) |
| `--student-port` | 8000 | Student vLLM server port |
| `--teacher-port` | 8001 | Teacher vLLM server port |
| `--kl-coeff` | 1.0 | Coefficient for reverse KL loss |
| `--length-normalize` | False | Normalize loss by sequence length |
| `--rollouts-per-update` | 64 | Rollouts per training step |
| `--concurrency` | 32 | Parallel rollout generation |
| `--limit` | None | Limit training samples (None = use all) |
| `--logger` | `rich` | Loggers: rich, print, wandb, none (comma-separated) |
| `--eval-every` | 10 | Eval every N train steps |
| `--eval-limit` | 1000 | Number of test samples for eval |
| `--eval-temperature` | 0.0 | Sampling temperature for eval (greedy) |

### Training logs

Output includes:
- `train/loss`: Reverse KL loss
- `train/reverse_kl_mean`: Mean per-token KL divergence
- `train/correct_rate`: GSM8K accuracy on training samples
- `train/avg_completion_length`: Average tokens per completion
- `eval/accuracy`: GSM8K accuracy on test set
- `eval/parse_error_rate`: Parse error rate on test set

Rollouts are written to `opd_rollouts.jsonl`.

## How OPD works

1. **Student samples**: The student model generates completions for GSM8K problems
2. **Teacher scores**: The teacher model computes per-token logprobs on the student's samples
3. **Reverse KL loss**: Training minimizes `KL(student || teacher) = log π_student - log π_teacher`

This pushes the student to assign high probability to tokens the teacher prefers, while staying on-policy (sampling from itself).

## Tips

- **Same tokenizer is required**: OPD passes token IDs directly from student to teacher. If tokenizers differ, results will be meaningless.
- **Context window**: Ensure prompt + completion fits in teacher's context window. Truncation causes length mismatches.
- **GPU memory**: With larger models, you may need separate GPUs for student and teacher. Adjust `--gpu-memory-utilization` accordingly.
- **KL coefficient**: Start with `--kl-coeff 1.0`. Increase if student diverges too much from teacher.
