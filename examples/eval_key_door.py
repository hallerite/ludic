# run_evaluation.py

import json
import logging
import sys

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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("--- Starting KeyDoor Environment Evaluation ---")

    # 1. Initialize the VLLMClient.
    # This will check for a running server and raise a ConnectionError if not found.
    try:
        # Assuming the VLLMClient connects to http://localhost:8000 by default.
        client = VLLMClient()
    except ConnectionError as e:
        logger.error(f"Could not connect to vLLM server: {e}")
        logger.error("Please ensure the server is running.")
        sys.exit(1)

    # 2. Define sampling parameters for the generation.
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=150)

    # 3. Iterate through each history management strategy.
    for strategy in HistoryManagement:
        logger.info(f"\n{'='*50}\nEVALUATING STRATEGY: {strategy.name}\n{'='*50}")

        # Instantiate the RolloutGenerator for the current strategy.
        # A fixed seed ensures the environment starts identically for each strategy,
        # providing a fair comparison.
        rg = RolloutGenerator(
            env_cls=KeyDoorEnv,
            max_steps=20,
            history_strategy=strategy,
            seed=42  # Use a fixed seed for reproducibility
        )

        # 4. Collect one trajectory (batch_size=1).
        trajectories = rg.collect(
            batch_size=1,
            model=client,
            sampling_params=sampling_params
        )

        # 5. Save the collected trajectory to a file.
        if trajectories:
            trajectory = trajectories[0]
            output_filename = f"trajectory_{strategy.name}.json"

            with open(output_filename, 'w') as f:
                json.dump(trajectory, f, indent=2)

            logger.info(f"✅ Trajectory saved to '{output_filename}'")

            # --- TOKEN STATISTICS CALCULATION ---
            total_prompt_tokens = 0
            total_completion_tokens = 0
            num_steps = 0

            if 'steps' in trajectory and isinstance(trajectory['steps'], list):
                num_steps = len(trajectory['steps'])
                for step in trajectory['steps']:
                    if isinstance(step, dict):
                        raw_data = step.get('raw')
                        # The 'raw' field should contain the JSON response from the server
                        if raw_data and isinstance(raw_data, dict):
                            usage_data = raw_data.get('usage')
                            if usage_data and isinstance(usage_data, dict):
                                total_prompt_tokens += usage_data.get('prompt_tokens', 0)
                                total_completion_tokens += usage_data.get('completion_tokens', 0)
                            else:
                                logger.warning(f"Missing or invalid 'usage' data in step raw response for {strategy.name}")
                        else:
                            logger.warning(f"Missing or invalid 'raw' data in step for {strategy.name}")
            
            logger.info(
                f"Token Stats for {strategy.name} ({num_steps} steps): "
                f"Input Tokens = {total_prompt_tokens}, "
                f"Output Tokens = {total_completion_tokens}, "
                f"Total Tokens = {total_prompt_tokens + total_completion_tokens}"
            )
            # --- END TOKEN STATISTICS CALCULATION ---

            final_reward = trajectory['steps'][-1]['reward'] if num_steps > 0 else 0
            if final_reward == 1.0:
                logger.info(f"Outcome: Success! 🚪 Agent unlocked the door.")
            else:
                logger.info(f"Outcome: Failure. 😔 Agent did not unlock the door.")
        else:
            logger.error(f"❌ Failed to generate a trajectory for strategy: {strategy.name}")

    logger.info("\n--- Evaluation Complete ---")