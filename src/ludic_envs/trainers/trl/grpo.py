from __future__ import annotations

from typing import Any, Optional, Type

import torch

from trl import GRPOTrainer
from ludic_envs.inference.rollout_generator import RolloutGenerator
from ludic_envs.envs.env import Env

class EnvGRPOTrainer(GRPOTrainer):

    def __init__(
        self,
        *trainer_args: Any,
        env_cls: Type[Env],
        rollout_max_steps: int = 1,
        rollout_kwargs: Optional[dict[str, Any]] = None,
        **trainer_kwargs: Any,
    ) -> None:
        # ---- we do *not* need an external reward model --------------------
        self.reward_funcs = []  # type: ignore[attr-defined]

        # ---- vanilla GRPO initialisation ---------------------------------
        super().__init__(reward_funcs=[], *trainer_args, **trainer_kwargs)

        # ---- rollout generator -------------------------------------------
        self.rollout_gen = RolloutGenerator(
            env_cls,
            rollout_max_steps,
            **(rollout_kwargs or {}),
        )

    # ------------------------------------------------------------------
    # main override
    # ------------------------------------------------------------------
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        breakpoint()
        # 1) collect rollouts ------------------------------------------------------
        trajectories = self.rollout_gen.collect(
            batch_size=len(inputs),
            model=self.llm,
            sampling_params={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k or -1,
                "min_p": self.min_p or 0.0,
                "max_tokens": self.max_completion_length,
            },
        )
        breakpoint()
        # 2) env-level returns & group baseline ---------------------------
        env_returns, env_gids = [], []
        for traj in trajectories:
            env_returns.append(sum(step["reward"] for step in traj["steps"]))
            env_gids.append(traj["group"])

        device = self.accelerator.device
        env_returns_t = torch.tensor(env_returns, dtype=torch.float32, device=device)
        env_gids_t = torch.tensor(env_gids, dtype=torch.long, device=device)

        grp_sum = torch.bincount(env_gids_t, weights=env_returns_t)
        grp_cnt = torch.bincount(env_gids_t)
        grp_mean = grp_sum / grp_cnt.clamp(min=1)
        adv_env = env_returns_t - grp_mean[env_gids_t]

        # 3) flatten steps ------------------------------------------------
        prompts, completions, adv_list = [], [], []
        for env_idx, traj in enumerate(trajectories):
            for step in traj["steps"]:
                prompts.append(step["prompt"])
                completions.append(step["assistant"])
                adv_list.append(adv_env[env_idx])

        advantages = torch.stack(adv_list).float()


        # 4) tokenise -----------------------------------------------------
        prompt_texts = [
            self.processing_class.apply_chat_template(msgs, tokenize=False)  # type: ignore[attr-defined]
            for msgs in prompts
        ]

        tok_prompt = self.processing_class(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ).to(device)

        tok_comp = self.processing_class(
            completions,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=False,
        ).to(device)
        breakpoint()
        # 5) tensors expected by GRPOTrainer -----------------------------
        return {
            "prompt_ids": tok_prompt.input_ids,
            "prompt_mask": tok_prompt.attention_mask,
            "completion_ids": tok_comp.input_ids,
            "completion_mask": tok_comp.attention_mask,
            "advantages": advantages,
            "old_per_token_logps": None,
        }
