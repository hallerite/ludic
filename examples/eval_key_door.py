import json
import logging
import sys
import os
import time
from functools import partial

import pygame
from ludic_envs.envs.pomdp.key_door import PygameRenderer

# --- Assumed Imports from your project structure ---
from ludic_envs.envs.pomdp.key_door import KeyDoorEnv
from ludic_envs.inference.rollout_generator import HistoryManagement, RolloutGenerator
from ludic_envs.inference.vllm_client import VLLMClient

try:
    from vllm import SamplingParams
except ImportError:
    print("vLLM is not installed. Please install it with `pip install vllm`.", file=sys.stderr)
    sys.exit(1)

# --- Basic logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("--- Starting KeyDoor Environment Evaluation ---")

    # --- CONFIGURATION ---
    EVAL_SIZE = 3
    EVAL_MAX_STEPS = 35
    # --- END CONFIGURATION ---

    # --- CONDITIONAL RENDERING LOGIC ---
    render_enabled = os.getenv("RENDER_EVAL") == "1"
    renderer = None
    env_class_to_use = KeyDoorEnv

    if render_enabled:
        logger.info(f"Rendering ENABLED for a {EVAL_SIZE}x{EVAL_SIZE} grid. Patching KeyDoorEnv...")
        
        renderer = PygameRenderer(size=EVAL_SIZE)
        
        # This is a cleaner way to set the env size without extra patching
        env_class_to_use = partial(KeyDoorEnv, size=EVAL_SIZE, max_steps=EVAL_MAX_STEPS)

        original_step = KeyDoorEnv.step
        original_reset = KeyDoorEnv.reset

        def _step_with_render(self, action):
            result = original_step(self, action)
            renderer.render(self)
            return result

        def _reset_with_render(self, *args, **kwargs):
            obs = original_reset(self, *args, **kwargs)
            renderer.render(self)
            return obs

        KeyDoorEnv.step = _step_with_render
        KeyDoorEnv.reset = _reset_with_render
    else:
        logger.info("Rendering is DISABLED. To enable, set RENDER_EVAL=1")
    # --- END OF PATCHING LOGIC ---

    try:
        client = VLLMClient()
    except ConnectionError as e:
        logger.error(f"Could not connect to vLLM server: {e}")
        sys.exit(1)

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=500)

    for strategy in HistoryManagement:
        logger.info(f"\n{'='*50}\nEVALUATING STRATEGY: {strategy.name}\n{'='*50}")

        rg = RolloutGenerator(
            max_steps=EVAL_MAX_STEPS,
            env_cls=env_class_to_use,
            # If not rendering, these values will be passed to the constructor
            # to override the defaults.
            env_kwargs={'size': EVAL_SIZE, 'max_steps': EVAL_MAX_STEPS},
            history_strategy=strategy,
            seed=42
        )

        trajectories = rg.collect(batch_size=1, model=client, sampling_params=sampling_params)

        # The rest of the script is unchanged...
        if trajectories:
            trajectory = trajectories[0]
            output_filename = f"trajectory_{strategy.name}.json"
            with open(output_filename, 'w') as f:
                json.dump(trajectory, f, indent=2)
            logger.info(f"✅ Trajectory saved to '{output_filename}'")
            
            total_prompt_tokens, total_completion_tokens, num_steps = 0, 0, 0
            if 'steps' in trajectory and isinstance(trajectory['steps'], list):
                num_steps = len(trajectory['steps'])
                for step in trajectory['steps']:
                    if isinstance(step, dict) and (raw_data := step.get('raw')) and isinstance(raw_data, dict):
                        if (usage_data := raw_data.get('usage')) and isinstance(usage_data, dict):
                            total_prompt_tokens += usage_data.get('prompt_tokens', 0)
                            total_completion_tokens += usage_data.get('completion_tokens', 0)
            
            logger.info(
                f"Token Stats for {strategy.name} ({num_steps} steps): "
                f"Input Tokens = {total_prompt_tokens}, "
                f"Output Tokens = {total_completion_tokens}, "
                f"Total Tokens = {total_prompt_tokens + total_completion_tokens}"
            )

            final_reward = trajectory['steps'][-1]['reward'] if num_steps > 0 else 0
            if final_reward == 1.0:
                logger.info(f"Outcome: Success! 🚪 Agent unlocked the door.")
            else:
                logger.info(f"Outcome: Failure. 😔 Agent did not unlock the door.")
        else:
            logger.error(f"❌ Failed to generate a trajectory for strategy: {strategy.name}")

    logger.info("\n--- Evaluation Complete ---")
    
    # If we rendered, close the pygame window
    if renderer:
        logger.info("Closing renderer. Final state will be visible for 2 seconds.")
        time.sleep(2) # Give a moment to see the final state
        renderer.close()