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
    OPTIMAL_SCRATCHPAD = auto()


class RolloutGenerator:
    """Collect trajectories as *prompt/assistant/reward* triples."""

    def __init__(
        self,
        env_cls: Type[Env],
        max_steps: int,
        *,
        history_strategy: HistoryManagement = HistoryManagement.NO_HISTORY,
        env_kwargs: Dict[str, Any] | None = None,
        group_size: int = 1,
        seed: int | None = None,
    ) -> None:
        self.env_cls = env_cls
        self.max_steps = max_steps
        self.history_strategy = history_strategy
        self.group_size = group_size # FIXME: assumes GRPO!
        self.rng = random.Random(seed)
        self.env_kwargs = env_kwargs or {}

    def collect(self, batch_size: int, model: Any, sampling_params: Any) -> List[Dict[str, Any]]:
        """
        Roll out scenarios and return trajectories.
        Total number of environments = batch_size * group_size.
        """
        envs: List[Env] = []
        obs_text: List[str] = []
        group_ids: List[int] = []

        # batch_size here refers to the number of unique initial seeds/scenarios for GRPO.
        # Each scenario (group g) will have self.group_size environment instances.
        for g_idx in range(batch_size): # g_idx is the GRPO group index
            seed = self.rng.randint(0, 2**32 - 1)
            logger.debug(f"Seed for group {g_idx}: {seed}")
            for i_clone in range(self.group_size): 
                env = self.env_cls(**self.env_kwargs)
                envs.append(env)
                obs_text.append(env.reset(seed=seed)) # All clones in a group use the same seed
                group_ids.append(g_idx)
                logger.debug(f"Initialized env instance {len(envs)-1} for group {g_idx}, clone {i_clone} with seed {seed}")

        n_total_envs = len(envs)
        done = [False] * n_total_envs
        histories: List[List[Dict[str, str]]] = [[] for _ in range(n_total_envs)]
        scratchpads: List[str] = ["" for _ in range(n_total_envs)] # Holds the latest scratchpad content for each env
        trajectories = [{"group": group_ids[i], "steps": []} for i in range(n_total_envs)]

        logger.info(f"Starting rollout collection for {batch_size} groups with strategy: {self.history_strategy.name}")

        for step_idx in range(self.max_steps):
            active_env_indices = [i for i, d_flag in enumerate(done) if not d_flag]

            if not active_env_indices:
                logger.info("All %d environments completed. Ending rollout early at step %d.", n_total_envs, step_idx)
                break
            
            logger.debug(f"Step {step_idx + 1}/{self.max_steps}. Active environments: {len(active_env_indices)} / {n_total_envs}")

            active_prompts: List[List[Dict[str, str]]] = []
            env_indices_for_active_prompts: List[int] = [] 

            for original_env_idx in active_env_indices:
                prompt = self._build_prompt(
                    envs[original_env_idx],
                    obs_text[original_env_idx],
                    histories[original_env_idx],
                    scratchpads[original_env_idx] # Pass current scratchpad for this env
                )
                active_prompts.append(prompt)
                env_indices_for_active_prompts.append(original_env_idx)
            
            if not active_prompts:
                continue

            active_replies_txt, active_replies_raw = sample(model, active_prompts, sampling_params)

            # Log generations to dedicated file
            for i, original_idx_in_active_list in enumerate(env_indices_for_active_prompts):
                prompt_for_log = active_prompts[i] # `i` is index in active_prompts
                reply_for_log = active_replies_txt[i] # `i` is index in active_replies_txt
                
                try:
                    prompt_str_for_log = json.dumps(prompt_for_log, indent=2)
                except TypeError:
                    prompt_str_for_log = str(prompt_for_log)

                log_message = (
                    f"STEP {step_idx + 1} - EnvID (Overall) {original_idx_in_active_list} (GRPO Group {group_ids[original_idx_in_active_list]})\n"
                    f"PROMPT:\n{prompt_str_for_log}\n"
                    f"REPLY:\n{reply_for_log}"
                )
                gen_detail_logger.info(log_message)

            # Process replies and step environments
            # Iterate based on env_indices_for_active_prompts to ensure correct mapping
            for i, original_env_idx in enumerate(env_indices_for_active_prompts):
                assistant_reply = active_replies_txt[i]
                raw_reply_data = active_replies_raw[i]
                prompt_this_turn = active_prompts[i]

                try:
                    # The environment handles parsing internally
                    next_obs, reward, current_env_done_flag, info = envs[original_env_idx].step(assistant_reply)

                except Exception as e: 
                    logger.error(
                        f"UNEXPECTED CRITICAL ERROR in env.step() for env {original_env_idx}",
                        exc_info=True
                    )
                    raise 
                
                done[original_env_idx] = current_env_done_flag
                
                # --- UPDATE SCRATCHPAD BASED ON STRATEGY ---
                if self.history_strategy == HistoryManagement.SCRATCHPAD:
                    try:
                        # LLM-managed scratchpad
                        scratchpads[original_env_idx] = extract_tag_value(assistant_reply, "scratchpad")
                        logger.debug(f"Env {original_env_idx} updated scratchpad via LLM.")
                    except ValueError:
                        logger.warning(f"Scratchpad tag missing for env {original_env_idx}. Last scratchpad preserved.")
                
                elif self.history_strategy == HistoryManagement.OPTIMAL_SCRATCHPAD:
                    # Environment-provided "perfect" scratchpad
                    current_env = envs[original_env_idx]
                    if hasattr(current_env, 'get_optimal_scratchpad'):
                        scratchpads[original_env_idx] = current_env.get_optimal_scratchpad()
                        logger.debug(f"Env {original_env_idx} got optimal scratchpad from environment.")
                    else:
                        raise TypeError(f"History strategy is OPTIMAL_SCRATCHPAD but env {type(current_env).__name__} has no 'get_optimal_scratchpad' method.")
                
                # Update history for FULL_HISTORY strategy
                if self.history_strategy == HistoryManagement.FULL_HISTORY:
                    if not histories[original_env_idx]:
                        histories[original_env_idx].append({"role": "system", "content": envs[original_env_idx].system_prompt})
                    histories[original_env_idx].extend([
                        {"role": "user", "content": obs_text[original_env_idx]},
                        {"role": "assistant", "content": assistant_reply},
                    ])
                
                trajectories[original_env_idx]["steps"].append({
                    "prompt": prompt_this_turn,
                    "assistant": assistant_reply,
                    "reward": reward,
                    "raw": raw_reply_data.raw_response if hasattr(raw_reply_data, 'raw_response') else raw_reply_data,
                    "info_from_env": info if info else {}
                })
                obs_text[original_env_idx] = next_obs

            if all(done):
                logger.info("All %d environments completed within max_steps (after step %d processing).", n_total_envs, step_idx + 1)
                break
        
        if step_idx == self.max_steps - 1 and not all(done):
            active_after_max = [i for i, d_flag in enumerate(done) if not d_flag]
            logger.info(
                f"Rollout finished due to max_steps ({self.max_steps}). "
                f"{len(active_after_max)} environments were still active: {active_after_max}."
            )

        return trajectories

    def _build_prompt(self, env: Env, obs: str, history: List[Dict[str, str]], scratchpad: str) -> List[Dict[str, str]]:
        """
        Builds the prompt by composing pieces from the environment based on the strategy.
        """
        prompt: List[Dict[str, str]] = []
        
        # Start with the environment's base system prompt.
        system_prompt_content = env.system_prompt
        
        # Initialize user_content with the current observation.
        # This will be modified for OPTIMAL_SCRATCHPAD.
        user_content_for_turn = obs

        if self.history_strategy == HistoryManagement.SCRATCHPAD:
            # For agent-managed scratchpad, add instructions AND the agent's last memory to system prompt.
            if hasattr(env, 'scratchpad_instr') and env.scratchpad_instr:
                system_prompt_content += env.scratchpad_instr
            else:
                raise ValueError("SCRATCHPAD strategy requires 'scratchpad_instr' in the environment.")
            
            if scratchpad: # This is the agent's previous scratchpad
                system_prompt_content += f"\n\n## Your Current Memory (from previous turn):\n{scratchpad}"
        
        elif self.history_strategy == HistoryManagement.OPTIMAL_SCRATCHPAD:
            # For optimal scratchpad, DO NOT add instructions to system prompt.
            # System prompt remains the base env.system_prompt.
            # Prepend the optimal scratchpad to the user's observation string.
            if scratchpad: # This is the optimal scratchpad from env.get_optimal_scratchpad()
                user_content_for_turn = f"## Your Current Memory (provided by the environment):\n{scratchpad}\n\n## Current Observation:\n{obs}"
            # No modification to system_prompt_content here

        # --- Assemble the final prompt list ---
        if self.history_strategy == HistoryManagement.FULL_HISTORY:
            # system_prompt_content for FULL_HISTORY is just the base env.system_prompt,
            # as the history should contain the initial system message if needed.
            prompt.extend(history)
            if not prompt: # If history is empty, ensure there's a system prompt.
                prompt.append({"role": "system", "content": env.system_prompt})
        else:
            # For NO_HISTORY, SCRATCHPAD, OPTIMAL_SCRATCHPAD:
            # system_prompt_content was modified for SCRATCHPAD.
            # For OPTIMAL_SCRATCHPAD and NO_HISTORY, system_prompt_content is the base env.system_prompt.
            prompt.append({"role": "system", "content": system_prompt_content})
        
        prompt.append({"role": "user", "content": user_content_for_turn}) # Use the (potentially modified) user_content_for_turn
        return prompt