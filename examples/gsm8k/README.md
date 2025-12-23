# GSM8K Training/Eval with vLLM

Minimal commands to train and evaluate GSM8K using a vLLM-hosted model and the provided scripts.

## Prerequisites

- At least 2 GPU(s). I used 2 A100s.
- Required extra packages: `datasets`, `math-verify`.

Install deps (once):
```bash
uv sync --extra examples
```

## 1) Start vLLM server

In one terminal, launch vLLM on one GPU:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server --model Qwen/Qwen2.5-0.5B-Instruct
```

Adjust `CUDA_VISIBLE_DEVICES`, `--gpu_memory_utilization`, or other vLLM flags for your hardware.

## 2) Train on GSM8K

In another terminal, run the training script.
Assuming you run it from the top-level of the repository, we need to add `environments/` into our Python path.
Example:
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python examples/gsm8k/train_gsm8k.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --micro-token-budget 16384 \
  --max-seq-len 1024
```

Notes:
- `PYTHONPATH=.` ensures local imports resolve.
- Defaults connect to vLLM at `127.0.0.1:8000`; override `--host/--port` if you run it elsewhere.
- Tweak rollouts, concurrency, and limits via flags like `--rollouts-per-update`, `--concurrency`, `--limit`, `--train-steps`, `--micro-token-budget`, `--max-seq-len`, `--group-size`, `--eval-every`, `--eval-limit`, `--eval-temperature`, and `--logger`.
- Tune `--limit`, `--concurrency`, `--train-steps`, and `--micro-token-budget` to fit your budget/VRAM.
- Training logs include loss, reward, avg_completion_length, and reducer stats (correct/parse-error rates, token totals).
- Checkpoints are written to `checkpoints_gsm8k/` by default.
- Training uses a boxed-answer parser (`boxed_parser`) and a small system prompt (`"First, think step by step. Then put your final answer inside \\boxed{...}."` by default).

## 3) Evaluate

Use the vLLM eval script to measure accuracy and reducer metrics:
```bash
PYTHONPATH=. uv run python examples/gsm8k/eval_gsm8k_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --split test --limit 250 \
  --temperature 0.0 --max-tokens 512 \
  --system-prompt "" \
  --out gsm8k_eval.jsonl
```

Output includes accuracy plus reducer stats (correct_rate, parse_err_rate, avg_completion_length, total_completion_tokens). Results are also written to the JSONL path you set with `--out`.

## Tips

- Eval defaults mirror Qwen’s `evaluate_chat_gsm8k.py`: greedy decoding (`--temperature 0.0`) and no system prompt (`--system-prompt ""`). Pass your training prompt if you want to score the finetuned policy as-trained with the train script, e.g.:
  ```bash
  PYTHONPATH=. uv run python examples/gsm8k/eval_gsm8k_vllm.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --temperature 0.0 --system-prompt "First, think step by step. Then put your final answer inside \\boxed{...}." \
    --out gsm8k_eval_trained_prompt.jsonl
  ```
- Use `--start-server` in `eval_gsm8k_vllm.py` if you want the script to launch vLLM for you.
- Greedy eval (`temperature 0.0`) with an empty system prompt matches the Qwen GSM8K evaluation script (see https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_gsm8k.py).
- Logs/rollouts are written to `gsm8k_train_rollouts.jsonl` (training) and your chosen `--out` path (eval).***
