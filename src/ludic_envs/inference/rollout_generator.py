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
            logger.debug(f"Seed for GRPO group {g_idx}: {seed}")
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

        logger.info(f"Starting rollout collection for {batch_size} groups, {self.group_size} clones/group, total {n_total_envs} environments.")

        for step_idx in range(self.max_steps):
            active_env_indices = [i for i, d_flag in enumerate(done) if not d_flag]

            if not active_env_indices:
                logger.info("All %d environments completed. Ending rollout early at step %d.", n_total_envs, step_idx)
                break
            
            logger.debug(f"Step {step_idx + 1}/{self.max_steps}. Active environments: {len(active_env_indices)} / {n_total_envs}")

            active_prompts: List[List[Dict[str, str]]] = []
            # Store mapping to original env index to correctly update states later
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
            
            if not active_prompts: # Should only happen if all envs were done at the start of the loop
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
                assistant_reply = active_replies_txt[i] # Raw LLM string output
                raw_reply_data = active_replies_raw[i]  # Raw generation object from vLLM/sample
                prompt_this_turn = active_prompts[i]    # The prompt that led to this reply

                try:
                    if self.history_strategy == HistoryManagement.SCRATCHPAD:
                        try:
                            # Update scratchpad memory for this specific environment
                            scratchpads[original_env_idx] = extract_tag_value(assistant_reply, "scratchpad")
                            logger.debug(f"Env {original_env_idx} updated scratchpad: \"{scratchpads[original_env_idx][:50]}...\"")
                        except ValueError:
                            logger.warning(f"Scratchpad tag missing/malformed for env {original_env_idx} (Group {group_ids[original_env_idx]}). Last scratchpad preserved. Reply: \"{assistant_reply[:100].replace(os.linesep, ' ')}...\"")
                            # No 'pass' needed, scratchpads[original_env_idx] remains unchanged.
                    
                    # Pass the raw assistant_reply directly to the environment's step method.
                    # The environment is now responsible for parsing and handling parsing errors.
                    next_obs, reward, current_env_done_flag, info = envs[original_env_idx].step(assistant_reply)

                except Exception as e: 
                    # This catch block is for *UNEXPECTED* errors from env.step().
                    # Parsing errors (ValueErrors) should be handled *within* env.step(),
                    # which should then return a valid 4-tuple (obs, rew, done, info)
                    # with an error message in the observation.
                    # If an exception reaches here, it means env.step() itself crashed.
                    logger.error(
                        f"UNEXPECTED CRITICAL ERROR in env.step() for env {original_env_idx} (Group {group_ids[original_env_idx]}) "
                        f"with raw LLM input: \"{assistant_reply[:200].replace(os.linesep, ' ')}...\". "
                        f"This likely indicates a bug in the environment's step() method that needs to be fixed.",
                        exc_info=True # This will log the full traceback of the exception 'e'
                    )
                    # Re-raise the original exception to halt the collect process,
                    # making bugs in environment implementations immediately visible.
                    raise 
                
                done[original_env_idx] = current_env_done_flag
                
                # Update history for FULL_HISTORY strategy
                if self.history_strategy == HistoryManagement.FULL_HISTORY:
                    if not histories[original_env_idx]: # Initialize with system prompt if first entry
                         histories[original_env_idx].append({"role": "system", "content": envs[original_env_idx].system_prompt})
                    histories[original_env_idx].extend([
                        {"role": "user", "content": obs_text[original_env_idx]}, # Observation that led to this action
                        {"role": "assistant", "content": assistant_reply},      # Raw LLM reply
                    ])
                
                trajectories[original_env_idx]["steps"].append({
                    "prompt": prompt_this_turn,    # The full prompt list of dicts sent to LLM
                    "assistant": assistant_reply, # The raw string reply from LLM
                    "reward": reward,
                    "raw": raw_reply_data.raw_response if hasattr(raw_reply_data, 'raw_response') else raw_reply_data,
                    "info_from_env": info if info else {} # Store info dict from environment
                })
                obs_text[original_env_idx] = next_obs # Update observation for the next turn

            if all(done):
                logger.info("All %d environments completed within max_steps (after step %d processing).", n_total_envs, step_idx + 1)
                break
        
        # Check if loop finished due to max_steps but not all envs were done
        # step_idx is 0-indexed. If loop completes fully, step_idx will be self.max_steps - 1.
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