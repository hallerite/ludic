# In your EnvGRPOTrainer.py

from __future__ import annotations

from typing import Any, Optional, Type, Union
from urllib.parse import urlparse # For parsing base_url

import torch

from trl import GRPOTrainer, GRPOConfig # Import GRPOConfig for type hinting
# from trl.trainer.utils import pad # Not needed here if GRPOTrainer handles its own padding
from ludic_envs.inference.rollout_generator import RolloutGenerator
from ludic_envs.envs.env import Env
from ludic_envs.inference.vllm_client import VLLMClient # Your custom VLLMClient

# For type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
    from datasets import Dataset, IterableDataset
    import torch.optim

class EnvGRPOTrainer(GRPOTrainer):

    def __init__(
        self,
        env_cls: Type[Env],
        # --- Core GRPOTrainer arguments ---
        model: Union[str, "PreTrainedModel"], # The policy model being trained
        args: Optional[GRPOConfig] = None,    # GRPOConfig from TRL
        train_dataset: Optional[Union["Dataset", "IterableDataset"]] = None,
        # --- EnvGRPOTrainer specific arguments ---
        rollout_max_steps: int = 1,
        rollout_kwargs: Optional[dict[str, Any]] = None,
        # --- Other GRPOTrainer arguments ---
        **trainer_kwargs: Any,
    ) -> None:
        if "reward_funcs" not in trainer_kwargs: # EnvGRPOTrainer uses env rewards
            trainer_kwargs["reward_funcs"] = []

        # Initialize GRPOTrainer first, which sets up self.args, self.llm, etc.
        super().__init__(model=model, args=args, train_dataset=train_dataset, **trainer_kwargs)

        # ---- Determine the model/client for rollouts based SOLELY on GRPOConfig (self.args) ----
        self.generation_model: Union["LLM", VLLMClient] # Define the attribute

        if self.args.use_vllm: # self.args is the GRPOConfig instance
            if self.args.vllm_mode == "colocate":
                # Rely on the base GRPOTrainer to have initialized self.llm.
                # self.llm is a vllm.LLM instance created by GRPOTrainer.
                if hasattr(self, "llm") and self.llm is not None:
                    print("EnvGRPOTrainer: Using TRL's colocated vLLM engine (self.llm) for rollouts.")
                    self.generation_model = self.llm
                else:
                    raise RuntimeError(
                        "EnvGRPOTrainer: GRPOConfig 'use_vllm=True' and 'vllm_mode=colocate' is set, "
                        "but 'self.llm' was not found or initialized by the base GRPOTrainer. "
                        "This typically occurs if the main model specified for the trainer is "
                        "also intended for vLLM colocation and TRL's vLLM setup succeeded."
                    )
            elif self.args.vllm_mode == "server":
                print("EnvGRPOTrainer: GRPOConfig 'use_vllm=True' and 'vllm_mode=server' is set. "
                      "Initializing custom Ludic VLLMClient for rollouts.")
                
                client_host = self.args.vllm_server_host
                client_port = self.args.vllm_server_port

                # Prefer vllm_server_base_url if provided
                if self.args.vllm_server_base_url:
                    try:
                        parsed_url = urlparse(self.args.vllm_server_base_url)
                        # Override host/port only if parsing is successful and they exist in the URL
                        if parsed_url.hostname: client_host = parsed_url.hostname
                        if parsed_url.port: client_port = parsed_url.port
                    except Exception as e:
                        print(f"Warning: Could not parse vllm_server_base_url '{self.args.vllm_server_base_url}'. "
                              f"Falling back to vllm_server_host/port. Error: {e}")
                
                self.generation_model = VLLMClient(
                    host=client_host,
                    port=client_port,
                    connection_timeout=self.args.vllm_server_timeout if hasattr(self.args, 'vllm_server_timeout') else 60.0
                )
            else:
                raise ValueError(
                    f"EnvGRPOTrainer: GRPOConfig 'use_vllm=True' but has an unsupported "
                    f"'vllm_mode': '{self.args.vllm_mode}'. Expected 'colocate' or 'server'."
                )
        else: # self.args.use_vllm is False
            raise ValueError(
                "EnvGRPOTrainer requires 'use_vllm=True' in GRPOConfig to automatically set up "
                "a generation model for rollouts (either 'colocate' for vllm.LLM or 'server' "
                "for your custom VLLMClient). If 'use_vllm=False', this trainer cannot "
                "currently function as its RolloutGenerator needs a vLLM-compatible interface."
            )

        # ---- Rollout generator initialization ----
        self.rollout_gen = RolloutGenerator(
            env_cls,
            rollout_max_steps,
            **(rollout_kwargs or {}),
        )

    # The methods _get_rollout_sampling_params and _generate_and_score_completions
    # remain the same as in the previous correct version, as they correctly use `self.generation_model`.
    # Make sure they are included in your EnvGRPOTrainer class.

    def _get_rollout_sampling_params(self) -> "SamplingParams":
        """Constructs vLLM SamplingParams for rollouts from trainer attributes."""
        from vllm import SamplingParams 

        return SamplingParams(
            n=1,
            temperature=self.temperature, # from GRPOConfig
            top_p=self.top_p, # from GRPOConfig
            top_k=self.top_k if self.top_k is not None and self.top_k > 0 else -1, # from GRPOConfig
            min_p=self.min_p if self.min_p is not None else 0.0, # from GRPOConfig
            max_tokens=self.max_completion_length, # from GRPOConfig
            repetition_penalty=self.repetition_penalty, # from GRPOConfig
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        device = self.accelerator.device
        #breakpoint()
        rollout_sampling_params = self._get_rollout_sampling_params()
        
        trajectories = self.rollout_gen.collect(
            batch_size=len(inputs),
            model=self.generation_model,
            sampling_params=rollout_sampling_params,
        )

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

        # ----------------------------------------------------------------
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["env/reward_mean"].append(env_returns_t.mean().item())
        self._metrics[mode]["env/reward_std"].append(env_returns_t.std().item())
        self._metrics[mode]["env/advantage_mean"].append(adv_env.mean().item())
        self._metrics[mode]["env/advantage_std"].append(adv_env.std().item())
        # ----------------------------------------------------------------

        prompts_for_policy_model = []
        completions_from_rollout = []
        
        advantages_for_policy_model = []

        for env_idx, traj in enumerate(trajectories):
            for step in traj["steps"]:
                prompts_for_policy_model.append(step["prompt"])
                completions_from_rollout.append(step["assistant"])
                advantages_for_policy_model.append(adv_env[env_idx])

        advantages_tensor = torch.tensor(advantages_for_policy_model, dtype=torch.float32, device=device)

        grp_mean = grp_sum / grp_cnt.clamp(min=1)
        
        prompt_texts_for_policy = [
            self.processing_class.apply_chat_template(
                conversation=msgs, 
                tokenize=False,
                add_generation_prompt=True
            )
            for msgs in prompts_for_policy_model
        ]

        tok_prompt = self.processing_class(
            prompt_texts_for_policy,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.max_prompt_length, # From GRPOConfig
            add_special_tokens=False, 
        ).to(device)

        tok_comp = self.processing_class(
            completions_from_rollout,
            return_tensors="pt",
            padding=True,
            padding_side="right", 
            truncation=True,
            max_length=self.max_completion_length, # From GRPOConfig
            add_special_tokens=False, 
        ).to(device)

        return {
            "prompt_ids": tok_prompt.input_ids,
            "prompt_mask": tok_prompt.attention_mask,
            "completion_ids": tok_comp.input_ids,
            "completion_mask": tok_comp.attention_mask.int(), # turn into int32 to match TRL,
            "advantages": advantages_tensor.to(device),
            "old_per_token_logps": None,
        }