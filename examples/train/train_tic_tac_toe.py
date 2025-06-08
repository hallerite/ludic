# First, spin up a vllm server: CUDA_VISIBLE_DEVICES=0 python src/ludic_envs/inference/vllm_server.py --model Qwen/Qwen2.5-7B-Instruct

# Then, start the training: CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 --nnodes=1 --master_addr=localhost --master_port=29500  examples/train/train_tic_tac_toe.py


from datasets import Dataset
from trl import GRPOConfig
from ludic_envs.trainers.trl.grpo import EnvGRPOTrainer
from ludic_envs.envs.mdp.tic_tac_toe import TicTacToe
from ludic_envs.envs.mdp.key_env import KeyAndChestEnv 

GROUP_SIZE = 7

# 1) one-row dummy dataset ── GRPO ignores its contents for env interactions
train_ds = Dataset.from_dict({
    "prompt": ["1", "2", "3", "4", "5", "6", "7", "8"]
})

# 3) GRPO hyper-params
cfg = GRPOConfig(
    # ------ Training Hyperparameters ------
    output_dir="ttt",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    logging_steps=1,
    num_iterations=1,
    log_completions=True,
    save_steps=500,
    report_to="wandb",
    bf16=True,
    beta=0.0,
    use_vllm=True,
    vllm_mode="server",
    num_train_epochs=100,

    # ------ Generation & Model Length Parameters ------
    num_generations=GROUP_SIZE,         # Number of completions per prompt/state
    max_prompt_length=100,            # <--- Adjust: Max length of a Tic-Tac-Toe state representation for Qwen
                                      #      (e.g., "Board: X O . | . . . | . . X\nPlayer: O\nAction:")
    max_completion_length=10,         # <--- Adjust: Max length for a Tic-Tac-Toe action (e.g., "move 4")
                                      #      40 was generous; TTT actions are short.
                                      #      vLLM's max_model_len will be max_prompt_length + max_completion_length
)


trainer = EnvGRPOTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    train_dataset=train_ds,
    args=cfg,

    # --------- env rollout ----------
    env_cls=TicTacToe,
    rollout_max_steps=9,
    rollout_kwargs=dict(
        group_size=GROUP_SIZE,
        seed=123,
    ),
)

trainer.train()