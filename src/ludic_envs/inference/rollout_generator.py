from __future__ import annotations
from typing import Type, List, Dict, Any # Ensure Any is imported

import random
import logging
import json
import os

from ludic_envs.envs.env import Env
from ludic_envs.inference.sample import sample
from ludic_envs.parsers import extract_tag_value   # noqa: F401

# -----------------------------------------------------------------------------
# CONSOLE LOGGING
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__) # Logger for general messages from this module
# BasicConfig should ideally be called once, in your main script or test setup.
# If it's here, it might conflict if other modules also call it.
# For now, we'll keep it as you had it.
logging.basicConfig(
    level=logging.DEBUG, # This will affect the root logger and thus `logger` if not configured separately
    format="%(asctime)s [%(levelname)s] %(module)s: %(message)s", # Added module name
)

# -----------------------------------------------------------------------------
# DEDICATED FILE LOGGER FOR GENERATION DETAILS
# -----------------------------------------------------------------------------
# Create a specific logger instance for generation details
gen_detail_logger = logging.getLogger('GenerationDetailsLogger')
gen_detail_logger.setLevel(logging.INFO)  # Log all INFO messages and above for details
gen_detail_logger.propagate = False  # Prevent these logs from going to the console via root logger

# Setup File Handler - this will create the log file in the current working directory
# when the RolloutGenerator class is first imported/defined.
# It's generally better to set up handlers in the main part of an application,
# but for simplicity in this module, we can do it here.
# Ensure it's only added once.
if not gen_detail_logger.handlers:
    log_file_name = 'rollout_generations.log'
    # Use an absolute path or ensure your script runs from a consistent CWD
    log_file_path = os.path.join(os.getcwd(), log_file_name) 
    
    file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' to overwrite each run
    file_handler.setLevel(logging.INFO)
    
    # Simple formatter for the file, as the message will be pre-formatted
    file_formatter = logging.Formatter('%(asctime)s:\n%(message)s\n--------------------\n')
    file_handler.setFormatter(file_formatter)
    
    gen_detail_logger.addHandler(file_handler)
    # Also log to console where the detail file is being written, using the main module logger
    logger.info(f"Rollout generation details will be logged to: {log_file_path}")


class RolloutGenerator:
    """Collect trajectories as *prompt/assistant/reward* triples (dialog-centric)."""

    def __init__(
        self,
        env_cls: Type[Env],
        max_steps: int,
        *,
        remember_history: bool = False,
        group_size: int = 1, # For GRPO: number of completions per prompt
        seed: int | None = None,
    ) -> None:
        self.env_cls = env_cls
        self.max_steps = max_steps
        self.remember_history = remember_history
        self.group_size = group_size          # FIXME: assumes GRPO!
        self.rng = random.Random(seed)

    def collect(self, batch_size: int, model: Any, sampling_params: Any) -> List[Dict[str, Any]]:
        """
        Roll out *batch_size* distinct seeded scenarios, each with *group_size* clones,
        for *max_steps* and return trajectories.
        Total environments = batch_size * group_size.
        """
        envs: List[Env] = []
        obs_text: List[str] = []
        group_ids: List[int] = []

        # batch_size here refers to the number of unique initial seeds/scenarios for GRPO.
        # Each scenario (group g) will have self.group_size environment instances.
        for g_idx in range(batch_size): # g_idx is the GRPO group index
            seed = self.rng.randint(0, 2**32 - 1)
            logger.debug("Seed for GRPO group %d: %s", g_idx, seed)
            for _ in range(self.group_size): # Create `group_size` clones for this seed
                env = self.env_cls()
                envs.append(env)
                obs_text.append(env.reset(seed=seed))
                group_ids.append(g_idx) # All these clones belong to the same GRPO group g_idx

        n_total_envs = len(envs) # This is batch_size * group_size
        done = [False] * n_total_envs
        histories: List[List[Dict[str, str]]] = [[] for _ in range(n_total_envs)]
        # Initialize trajectories: one per environment instance
        trajectories = [{"group": group_ids[i], "steps": []} for i in range(n_total_envs)]

        for step_idx in range(self.max_steps):
            active_env_indices = [i for i, d_flag in enumerate(done) if not d_flag]

            if not active_env_indices:
                logger.info("All %d environments completed. Ending rollout early at step %d.", n_total_envs, step_idx)
                break

            # Prepare prompts for active environments
            active_prompts: List[List[Dict[str, str]]] = []
            env_indices_for_active_prompts: List[int] = [] # Keep track of original env index

            for env_idx in active_env_indices:
                # Inject system prompt once per env if history is remembered and it's the first step
                if self.remember_history and step_idx == 0 and not histories[env_idx]:
                    histories[env_idx].append(
                        {"role": "system", "content": envs[env_idx].system_prompt}
                    )
                
                active_prompts.append(
                    self._build_prompt(envs[env_idx], obs_text[env_idx], histories[env_idx])
                )
                env_indices_for_active_prompts.append(env_idx)
            
            if not active_prompts: # Should only happen if all envs were done
                continue

            # Sample model only for active environments
            active_replies_txt, active_replies_raw = sample(model, active_prompts, sampling_params)

            # --- Log generations to dedicated file ---
            for i, original_env_idx in enumerate(env_indices_for_active_prompts):
                # i is the index within the 'active_prompts' and 'active_replies_txt' lists
                prompt_for_log = active_prompts[i]
                reply_for_log = active_replies_txt[i]
                
                try:
                    prompt_str_for_log = json.dumps(prompt_for_log, indent=2)
                except TypeError:
                    prompt_str_for_log = str(prompt_for_log) # Fallback if not serializable

                log_message = (
                    f"STEP {step_idx + 1} - Overall EnvID {original_env_idx} (GRPO Group {group_ids[original_env_idx]})\n"
                    f"PROMPT:\n{prompt_str_for_log}\n"
                    f"REPLY:\n{reply_for_log}"
                )
                gen_detail_logger.info(log_message)
            # --- End File Logging ---

            # Process replies and step environments
            reply_idx_counter = 0 # Counter for iterating through active_replies_txt
            for original_env_idx in active_env_indices: # Iterate based on the original env indices that were active
                assistant_reply = active_replies_txt[reply_idx_counter]
                raw_reply_data = active_replies_raw[reply_idx_counter]
                reply_idx_counter += 1

                # The prompt that led to this assistant_reply
                # It's active_prompts[index_within_active_batch]
                # We can find index_within_active_batch using env_indices_for_active_prompts.index(original_env_idx)
                prompt_this_turn = active_prompts[env_indices_for_active_prompts.index(original_env_idx)]

                current_obs_text = obs_text[original_env_idx] # Observation before taking this step

                next_obs, reward, current_env_done_flag, _ = envs[original_env_idx].step(assistant_reply)
                done[original_env_idx] = current_env_done_flag # Update overall done status

                if self.remember_history:
                    histories[original_env_idx].extend([
                        {"role": "user", "content": current_obs_text}, # Log observation leading to action
                        {"role": "assistant", "content": assistant_reply},
                    ])
                
                trajectories[original_env_idx]["steps"].append({
                    "prompt": prompt_this_turn, # The exact prompt given to the model for this step
                    "assistant": assistant_reply,
                    "reward": reward,
                    "raw": raw_reply_data, # Raw output from sample/VLLMClient
                })
                obs_text[original_env_idx] = next_obs # Update observation for the *next* step

            if all(done):
                logger.info("All %d environments completed within max_steps (after step %d processing).", n_total_envs, step_idx)
                break
        
        if step_idx == self.max_steps -1 and not all(done):
             logger.info("Rollout finished due to max_steps (%d), not all environments were done.", self.max_steps)


        return trajectories

    def _build_prompt(
        self,
        env: Env,
        obs: str,
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Return a fresh prompt list. If history is used, it's prepended."""
        current_turn_prompt: List[Dict[str,str]] = []
        if self.remember_history:
            # History already contains system prompt if it's the first turn and remember_history is on
            current_turn_prompt.extend(history) 
        else:
            # If not remembering history, always start with a system prompt
            current_turn_prompt.append({"role": "system", "content": env.system_prompt})
        
        current_turn_prompt.append({"role": "user", "content": obs})
        return current_turn_prompt