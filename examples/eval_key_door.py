import json
import logging
import sys
import os
import time
from functools import partial
from typing import Optional, List, Dict, Any

import pygame
from ludic_envs.envs.pomdp.key_door import PygameRenderer

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

    EVAL_SIZE = 3
    EVAL_MAX_STEPS = 35
    
    render_enabled = os.getenv("RENDER_EVAL") == "1"
    renderer: Optional[PygameRenderer] = None
    client_for_rollout: VLLMClient 
    
    # Use a mutable dictionary for shared state that needs to be modified by nested functions
    shared_render_context = {"last_env_for_render": None}

    try:
        if render_enabled:
            logger.info("Render enabled, using VLLMClientWithLogging.")
            client_for_rollout = VLLMClientWithLogging()
        else:
            client_for_rollout = VLLMClient()
    except ConnectionError as e:
        logger.error(f"Could not connect to vLLM server: {e}")
        sys.exit(1)
    except Exception as e: 
        logger.error(f"Error initializing VLLMClient: {e}")
        sys.exit(1)

    env_class_to_use = KeyDoorEnv
    
    if render_enabled:
        logger.info(f"Rendering ENABLED for a {EVAL_SIZE}x{EVAL_SIZE} grid. Patching KeyDoorEnv...")
        renderer = PygameRenderer(grid_size=EVAL_SIZE, log_width=400, cell_size=300)
        
        env_class_to_use = partial(KeyDoorEnv, size=EVAL_SIZE, max_steps=EVAL_MAX_STEPS)

        original_step = KeyDoorEnv.step
        original_reset = KeyDoorEnv.reset

        def _step_with_render(self_env: KeyDoorEnv, action_from_rg: str):
            if renderer and client_for_rollout and isinstance(client_for_rollout, VLLMClientWithLogging):
                if client_for_rollout.last_prompts_sent and client_for_rollout.last_responses_received:
                    current_prompt_msg_list = client_for_rollout.last_prompts_sent[0] 
                    raw_completion_obj = client_for_rollout.last_responses_received[0]

                    user_prompt_content = "P: User Obs N/A"
                    for msg_dict in reversed(current_prompt_msg_list):
                        if msg_dict.get("role") == "user":
                            user_prompt_content = msg_dict.get("content", "User Obs N/A")
                            break
                    
                    assistant_reply_text = getattr(raw_completion_obj.outputs[0], 'text', "Completion N/A")
                    display_prompt = user_prompt_content[:250] + "..." if len(user_prompt_content) > 250 else user_prompt_content
                    display_completion = assistant_reply_text[:300] + "..." if len(assistant_reply_text) > 300 else assistant_reply_text
                    
                    renderer.add_log_entry(display_prompt, display_completion)
                else:
                    renderer.add_log_entry("Prompt: (Not captured this step)", "Completion: (Not captured this step)")
            
            result = original_step(self_env, action_from_rg)
            shared_render_context["last_env_for_render"] = self_env
            if renderer:
                renderer.render(self_env)
            return result

        def _reset_with_render(self_env: KeyDoorEnv, *args, **kwargs):
            if renderer:
                renderer.clear_log()
                base_sys_prompt = self_env.system_prompt
                display_sys_prompt = base_sys_prompt[:250] + "..." if len(base_sys_prompt) > 250 else base_sys_prompt
                renderer.add_log_entry(f"SYSTEM: {display_sys_prompt}", "(Environment Reset)")
            
            obs = original_reset(self_env, *args, **kwargs)
            shared_render_context["last_env_for_render"] = self_env
            if renderer:
                renderer.render(self_env)
            return obs

        KeyDoorEnv.step = _step_with_render
        KeyDoorEnv.reset = _reset_with_render
    else:
        logger.info("Rendering is DISABLED. To enable, set RENDER_EVAL=1")

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=500)

    for strategy in HistoryManagement:
        logger.info(f"\n{'='*50}\nEVALUATING STRATEGY: {strategy.name}\n{'='*50}")

        rg = RolloutGenerator(
            max_steps=EVAL_MAX_STEPS,
            env_cls=env_class_to_use,
            env_kwargs={'size': EVAL_SIZE, 'max_steps': EVAL_MAX_STEPS},
            history_strategy=strategy,
            seed=42
        )
        trajectories = rg.collect(batch_size=1, model=client_for_rollout, sampling_params=sampling_params)

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
        
        if render_enabled and renderer: 
            logger.info(f"Strategy {strategy.name} finished. Final state shown for 2s.")
            if shared_render_context["last_env_for_render"]:
                 renderer.render(shared_render_context["last_env_for_render"]) 
            pygame.display.flip()
            time.sleep(2)

    logger.info("\n--- Evaluation Complete ---")
    
    if renderer:
        logger.info("All strategies evaluated. Renderer will stay open for log inspection.")
        logger.info("Scroll log with mouse wheel. Close window to exit.")
        
        running_pygame = True
        while running_pygame:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_pygame = False
                if event.type == pygame.MOUSEWHEEL:
                    renderer.handle_scroll_event(event)
                    # Re-render with the last known environment state
                    current_last_env = shared_render_context["last_env_for_render"]
                    if current_last_env:
                         renderer.render(current_last_env) 
                    else:
                         renderer.render() # Render log even if no game state
            
            pygame.time.Clock().tick(30) 

        logger.info("Closing renderer.")
        renderer.close()