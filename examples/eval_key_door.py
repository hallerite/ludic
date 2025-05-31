# run_evaluation.py

import json
import logging
import sys

# --- Assumed Imports from your project structure ---
# Make sure these paths are correct relative to where you run the script.
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

            # Log the final outcome for quick inspection.
            final_reward = trajectory['steps'][-1]['reward']
            if final_reward == 1.0:
                logger.info(f"Outcome: Success! 🚪 Agent unlocked the door.")
            else:
                logger.info(f"Outcome: Failure. 😔 Agent did not unlock the door.")
        else:
            logger.error(f"❌ Failed to generate a trajectory for strategy: {strategy.name}")

    logger.info("\n--- Evaluation Complete ---")