from __future__ import annotations
from typing import Type, List, Dict, Any, Tuple
from enum import Enum, auto

import random
import logging
import json
import os

from ludic_envs.envs.env import Env
from ludic_envs.inference.sample import sample
from ludic_envs.parsers import extract_tag_value

# -----------------------------------------------------------------------------
# CONSOLE LOGGING
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(module)s: %(message)s",
)

# -----------------------------------------------------------------------------
# DEDICATED FILE LOGGER FOR GENERATION DETAILS
# -----------------------------------------------------------------------------
gen_detail_logger = logging.getLogger('GenerationDetailsLogger')
gen_detail_logger.setLevel(logging.INFO)
gen_detail_logger.propagate = False

if not gen_detail_logger.handlers:
    log_file_name = 'rollout_generations.log'
    log_file_path = os.path.join(os.getcwd(), log_file_name) 
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s:\n%(message)s\n--------------------\n')
    file_handler.setFormatter(file_formatter)
    gen_detail_logger.addHandler(file_handler)
    logger.info(f"Rollout generation details will be logged to: {log_file_path}")

# --- Enum for managing history strategies ---
class HistoryManagement(Enum):
    """Defines the strategy for managing an agent's memory."""
    NO_HISTORY = auto()      # Agent is memoryless.
    FULL_HISTORY = auto()    # Agent sees the entire conversation history.
    SCRATCHPAD = auto()      # Agent manages its own memory via a scratchpad.


class RolloutGenerator:
    """Collect trajectories as *prompt/assistant/reward* triples."""

    def __init__(
        self,
        env_cls: Type[Env],
        max_steps: int,
        *,
        history_strategy: HistoryManagement = HistoryManagement.NO_HISTORY,
        group_size: int = 1,
        seed: int | None = None,
    ) -> None:
        self.env_cls = env_cls
        self.max_steps = max_steps
        self.history_strategy = history_strategy
        self.group_size = group_size # FIXME: assumes GRPO!
        self.rng = random.Random(seed)

    def collect(self, batch_size: int, model: Any, sampling_params: Any) -> List[Dict[str, Any]]:
        """
        Roll out scenarios and return trajectories. Enum to show how context is managed
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
                group_ids.append(g_idx)

        n_total_envs = len(envs)
        done = [False] * n_total_envs
        histories: List[List[Dict[str, str]]] = [[] for _ in range(n_total_envs)]
        scratchpads: List[str] = ["" for _ in range(n_total_envs)]
        trajectories = [{"group": group_ids[i], "steps": []} for i in range(n_total_envs)]

        for step_idx in range(self.max_steps):
            active_env_indices = [i for i, d_flag in enumerate(done) if not d_flag]

            if not active_env_indices:
                logger.info("All %d environments completed. Ending rollout early at step %d.", n_total_envs, step_idx)
                break

            active_prompts: List[List[Dict[str, str]]] = []
            env_indices_for_active_prompts: List[int] = []

            for env_idx in active_env_indices:
                prompt = self._build_prompt(
                    envs[env_idx],
                    obs_text[env_idx],
                    histories[env_idx],
                    scratchpads[env_idx]
                )
                active_prompts.append(prompt)
                env_indices_for_active_prompts.append(env_idx)
            
            if not active_prompts: # Should only happen if all envs were done
                continue

            active_replies_txt, active_replies_raw = sample(model, active_prompts, sampling_params)

            # --- Log generations to dedicated file ---
            for i, original_env_idx in enumerate(env_indices_for_active_prompts):
                # i is the index within the 'active_prompts' and 'active_replies_txt' lists
                prompt_for_log = active_prompts[i]
                reply_for_log = active_replies_txt[i]
                
                try:
                    prompt_str_for_log = json.dumps(prompt_for_log, indent=2)
                except TypeError:
                    prompt_str_for_log = str(prompt_for_log)

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

                prompt_this_turn = active_prompts[env_indices_for_active_prompts.index(original_env_idx)]
                current_obs_text = obs_text[original_env_idx]

                try:
                    # The environment's parse_action will receive the full assistant_reply.
                    action_content_for_env = assistant_reply 

                    if self.history_strategy == HistoryManagement.SCRATCHPAD:
                        # For SCRATCHPAD, extract and update the scratchpad memory.
                        # The full assistant_reply is still passed to env.parse_action.
                        try:
                            scratchpads[original_env_idx] = extract_tag_value(assistant_reply, "scratchpad")
                        except ValueError:
                            # If the tag is missing, preserve the last known scratchpad.
                            logger.warning(f"Scratchpad tag missing for env {original_env_idx}. Preserving last memory.")
                            pass
                        # The environment is now solely responsible for extracting the <action> tag
                        # from the full assistant_reply (action_content_for_env).
                    
                    # env.parse_action receives the full assistant_reply and extracts the action.
                    action_to_step = envs[original_env_idx].parse_action(action_content_for_env)
                    
                    next_obs, reward, current_env_done_flag, info = envs[original_env_idx].step(action_to_step)

                except ValueError as e: # Catches errors from env.parse_action or env.step
                    error_message = f"Your last response was invalid or badly formatted: {e}. Try again."
                    next_obs, reward, current_env_done_flag, info = (current_obs_text + "\n" + error_message), 0, False, {"illegal_move": True, "error": str(e)}

                done[original_env_idx] = current_env_done_flag
                
                if self.history_strategy == HistoryManagement.FULL_HISTORY:
                    if not histories[original_env_idx]:
                         histories[original_env_idx].append({"role": "system", "content": envs[original_env_idx].system_prompt})
                    histories[original_env_idx].extend([
                        {"role": "user", "content": current_obs_text},
                        {"role": "assistant", "content": assistant_reply},
                    ])
                
                trajectories[original_env_idx]["steps"].append({
                    "prompt": prompt_this_turn,
                    "assistant": assistant_reply,
                    "reward": reward,
                    "raw": raw_reply_data.raw_response,
                })
                obs_text[original_env_idx] = next_obs

            if all(done):
                logger.info("All %d environments completed within max_steps (after step %d processing).", n_total_envs, step_idx)
                break
        
        if step_idx == self.max_steps - 1 and not all(done):
             logger.info("Rollout finished due to max_steps (%d), not all environments were done.", self.max_steps)

        return trajectories
        
    def _build_prompt(self, env: Env, obs: str, history: List[Dict[str, str]], scratchpad: str) -> List[Dict[str, str]]:
        """
        Builds the prompt by composing pieces from the environment based on the strategy.
        """
        prompt: List[Dict[str, str]] = []
        
        # Start with the environment's base system prompt.
        system_prompt_content = env.system_prompt
        
        # If the strategy is SCRATCHPAD, compose the final system prompt.
        if self.history_strategy == HistoryManagement.SCRATCHPAD:
            # Append the specific scratchpad instructions, if the env provides them.
            if hasattr(env, 'scratchpad_instr') and env.scratchpad_instr:
                system_prompt_content += env.scratchpad_instr
            else:
                raise ValueError("Your env does not have a scratchpad instruction..")
            
            # Append the current memory content for this turn.
            if scratchpad:
                system_prompt_content += f"\n\n## Your Current Memory (from previous turn):\n{scratchpad}"

        # --- Assemble the final prompt list ---
        if self.history_strategy == HistoryManagement.FULL_HISTORY:
            prompt.extend(history)
            if not prompt:
                prompt.append({"role": "system", "content": system_prompt_content})
        else:
            # For SCRATCHPAD and NO_HISTORY, the prompt starts with the system message.
            # For SCRATCHPAD, system_prompt_content has been composed above.
            # For NO_HISTORY, it's just the base env.system_prompt.
            prompt.append({"role": "system", "content": system_prompt_content})
        
        prompt.append({"role": "user", "content": obs})
        return prompt