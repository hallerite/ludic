# FSDP2 MATH Training (3 GPUs + 1 vLLM GPU)

This example shows how to run Ludic with PyTorch FSDP2 for training while serving inference from a separate vLLM instance.

## Layout
- `train_math_fsdp2.py`: MATH training with FSDP2 + strict `<think>...</think> \\boxed{...}` parsing.

## Datasets
- Training: `qwedsacf/competition_math` (train split only; uses `solution` and extracts the final `\\boxed{...}`).
- Eval: `HuggingFaceH4/MATH-500` (uses `solution` and extracts the final `\\boxed{...}`; `answer` is available but not required).

## Assumptions
- 4 GPUs total: GPU0 runs vLLM; GPUs 1–3 run training.
- You start vLLM separately on GPU0 serving `Qwen/Qwen2.5-7B-Instruct`.
- Training runs under `torchrun` with NCCL; vLLM weight pushes use a separate NCCL communicator (`pynccl`) owned by the client. They do not interfere.

## Quickstart (template)
1. Start vLLM on GPU0 (using the bundled server):
   ```bash
   CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server \
     --model Qwen/Qwen2.5-7B-Instruct \
     --gpu_memory_utilization 0.8 \
     --port 8000 \
     --max-num-seqs 32
   ```

2. Launch training on GPUs 1–3 (FSDP2 + grouped-advantage REINFORCE):
   ```bash
   # Pin training to GPUs 1,2,3 so GPU0 stays free for vLLM
   CUDA_VISIBLE_DEVICES=1,2,3 PYTHONPATH=. PYTHONUNBUFFERED=1 uv run torchrun --nproc_per_node=3 \
     examples/fsdp2_training/train_math_fsdp2.py \
     --model Qwen/Qwen2.5-7B-Instruct \
     --vllm-host 127.0.0.1 \
     --vllm-port 8000 \
     --limit 2048 \
     --train-steps 50 --group-size 8 \
     --concurrency 11 --rollouts-per-update 8 --train-temperature 1.0 \
     --eval-before-start --eval-every 10 --eval-limit 100 \
     --eval-concurrency 32 --eval-temperature 0.0 \
     --max-seq-len 1024 --max-completion-tokens 512 \
     --micro-token-budget 16384 \
     --log-level INFO \
     --logger print \
     --rank0-only-output
  ```

   Optional: if you have fast scratch (e.g. `/ephemeral`), set `UV_CACHE_DIR`, `--rollout-log`, and `--checkpoint-dir` there to reduce I/O contention.

3. Checkpoints and logs:
   - Checkpoints: `checkpoints_math_fsdp2/` (rank0 saves).
- Rollout logs: `fsdp2_math_rollouts.rank{RANK}.jsonl`.
- Rank0 prints basic stats; attach `RichLiveLogger` only on rank0.
- Logger options: `--logger rich`, `--logger wandb`, or `--logger rich,wandb` (W&B uses `WANDB_*` env vars).

## Notes
- Mixed precision: uses `fsdp.MixedPrecisionPolicy` with bf16 params / fp32 reductions.
- Activation checkpointing: enabled by default in the script.
- Gradient sync: uses `set_requires_gradient_sync(False/True)` for micro-batched accumulation; no `no_sync`.
- Action parsing: the environment does not “re-parse” model outputs; the protocol’s parser must extract the final answer string used for grading.
- Weight publishing: only rank0 gathers a full state dict (DCP full_state_dict) and pushes weights to vLLM over the separate pynccl communicator.
- Sample sharding: MATH samples are sharded per rank to avoid duplicates; adjust to your data loader if needed.

Tune `--rollouts-per-update`, `--group-size`, `--train-steps`, `--micro-token-budget`, and `--max-seq-len` based on hardware. The script is a scaffold; extend it for eval, better logging, and real hyperparameters.
