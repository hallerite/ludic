import json
import logging
import sys
import os
import time
from functools import partial
from typing import Optional, List, Dict, Any

import pygame
from ludic_envs.envs.pomdp.key_door import KeyDoorEnv, PygameRenderer

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

class VLLMClientWithLogging(VLLMClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_prompts_sent: Optional[List[List[Dict[str, str]]]] = None
        self.last_responses_received: Optional[List[Any]] = None

    def chat(self, messages: List[List[Dict[str, str]]], **kwargs) -> List[Any]:
        self.last_prompts_sent = messages
        responses = super().chat(messages, **kwargs)
        self.last_responses_received = responses
        return responses

if __name__ == "__main__":
    logger.info("--- Starting KeyDoor Environment Evaluation ---")

    EVAL_SIZE = 4
    EVAL_MAX_STEPS = 50
    render_enabled = os.getenv("RENDER_EVAL") == "1"
    renderer: Optional[PygameRenderer] = None
    client_for_rollout: VLLMClient
    shared_render_context = {"last_env_for_render": None}

    try:
        client_class = VLLMClientWithLogging if render_enabled else VLLMClient
        client_for_rollout = client_class()
    except Exception as e:
        logger.error(f"Could not connect to vLLM server or initialize client: {e}")
        sys.exit(1)

    if render_enabled:
        logger.info(f"Rendering ENABLED for a {EVAL_SIZE}x{EVAL_SIZE} grid. Patching KeyDoorEnv...")
        renderer = PygameRenderer(grid_size=EVAL_SIZE, log_width=600, cell_size=200)
        
        original_step = KeyDoorEnv.step
        original_reset = KeyDoorEnv.reset

        def _step_with_render(self_env: KeyDoorEnv, action_from_rg: str):
            if renderer and isinstance(client_for_rollout, VLLMClientWithLogging):
                if client_for_rollout.last_prompts_sent and client_for_rollout.last_responses_received:
                    prompt_list = client_for_rollout.last_prompts_sent[0]
                    user_prompt = next((m.get("content", "") for m in reversed(prompt_list) if m.get("role") == "user"), "N/A")
                    completion = getattr(client_for_rollout.last_responses_received[0].outputs[0], 'text', "N/A")
                    renderer.add_log_entry(user_prompt, completion)
            
            result = original_step(self_env, action_from_rg)
            shared_render_context["last_env_for_render"] = self_env
            if renderer:
                renderer.render(self_env)
            return result

        def _reset_with_render(self_env: KeyDoorEnv, *args, **kwargs):
            # This is called by RolloutGenerator. The log is now cleared in the main loop.
            if renderer:
                renderer.add_log_entry("--- NEW GAME STARTED ---", f"Seed: {kwargs.get('seed') or 'N/A'}")
            
            obs = original_reset(self_env, *args, **kwargs)
            shared_render_context["last_env_for_render"] = self_env
            if renderer:
                renderer.render(self_env)
            return obs

        KeyDoorEnv.step = _step_with_render
        KeyDoorEnv.reset = _reset_with_render
    else:
        logger.info("Rendering is DISABLED. To enable, set RENDER_EVAL=1")

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

    for strategy in HistoryManagement:
        if strategy == HistoryManagement.SCRATCHPAD:
            continue
        logger.info(f"\n{'='*50}\nEVALUATING STRATEGY: {strategy.name}\n{'='*50}")

        if renderer:
            renderer.clear_log()
            renderer.show_transition_screen(f"Next Up: {strategy.name}", duration_sec=3)
            renderer.set_title(f"Strategy: {strategy.name}")

        rg = RolloutGenerator(
            max_steps=EVAL_MAX_STEPS,
            env_cls=partial(KeyDoorEnv, size=EVAL_SIZE, max_steps=EVAL_MAX_STEPS),
            env_kwargs={'size': EVAL_SIZE, 'max_steps': EVAL_MAX_STEPS},
            history_strategy=strategy,
            seed=42
        )
        trajectories = rg.collect(batch_size=1, model=client_for_rollout, sampling_params=sampling_params)

        if trajectories:
            trajectory = trajectories[0]
            output_filename = f"trajectory_{strategy.name}.json"
            with open(output_filename, 'w') as f: json.dump(trajectory, f, indent=2)
            logger.info(f"✅ Trajectory saved to '{output_filename}'")
            
            # Token counting logic remains the same
            num_steps = len(trajectory['steps']) if 'steps' in trajectory else 0
            final_reward = trajectory['steps'][-1]['reward'] if num_steps > 0 else 0
            logger.info(f"Outcome: {'Success!' if final_reward == 1.0 else 'Failure.'}")
        else:
            logger.error(f"❌ Failed to generate a trajectory for strategy: {strategy.name}")
        
        if render_enabled and renderer: 
            logger.info(f"Strategy {strategy.name} finished. Displaying final state for 4s.")
            if shared_render_context["last_env_for_render"]:
                 renderer.render(shared_render_context["last_env_for_render"]) 
            pygame.display.flip()
            time.sleep(4)

    logger.info("\n--- Evaluation Complete ---")
    
    if renderer:
        logger.info("All strategies evaluated. Close window to exit.")
        renderer.set_title("Evaluation Finished")
        renderer.render(shared_render_context.get("last_env_for_render"))
        
        running_pygame = True
        while running_pygame:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_pygame = False
                if event.type == pygame.MOUSEWHEEL:
                    renderer.handle_scroll_event(event)
                    renderer.render(shared_render_context.get("last_env_for_render"))
            pygame.time.Clock().tick(30)
        logger.info("Closing renderer.")
        renderer.close()