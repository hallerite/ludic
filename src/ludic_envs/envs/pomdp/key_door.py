from __future__ import annotations
import random
from typing import Dict, Tuple, Any

from ludic_envs.envs.env import Env
from ludic_envs.parsers import extract_tag_value

class KeyDoorEnv(Env):
    """
    A simple POMDP environment where an agent must explore an n x n grid to
    find a key and use it to unlock a door.

    This environment is 'pure' and knows nothing about agent memory systems
    like scratchpads. It only processes simple action commands.
    """
    VALID_ACTIONS = (
        "move north", "move south", "move east", "move west", "interact"
    )

    # --- Attributes are now non-optional for cleaner code ---
    agent_pos: Tuple[int, int]
    key_pos: Tuple[int, int]
    door_pos: Tuple[int, int]
    
    def __init__(self, size: int = 3, max_steps: int = 15):
        super().__init__()
        if not isinstance(size, int) or size < 2:
            raise ValueError("Size must be an integer of at least 2.")
        
        self.size = size
        self.max_steps = max_steps

        self.system_prompt = (
            "You are an agent in a grid-world maze. Your goal is to find the "
            "key, use it to unlock the door, and exit."
        )

        # --- State variables are initialized with placeholder values ---
        # These are immediately overwritten by reset(), but this satisfies the
        # non-optional type hints and provides a predictable default state.
        self.agent_pos = (0, 0)
        self.key_pos = (0, 0)
        self.door_pos = (0, 0)
        self.has_key = False
        self.done = False
        self.current_step = 0

    def reset(self, seed: int | None = None) -> str:
        if seed is not None:
            random.seed(seed)
        positions = random.sample(
            [(r, c) for r in range(self.size) for c in range(self.size)], 3
        )
        self.agent_pos = positions[0]
        self.key_pos = positions[1]
        self.door_pos = positions[2]
        self.has_key = False
        self.done = False
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self) -> str:
        if self.done:
            return "The game is over. Call reset()."
        obs = f"You are at position {self.agent_pos} in a {self.size}x{self.size} grid. "
        obs += f"You are {'carrying the key' if self.has_key else 'not carrying a key'}. "
        if self.agent_pos == self.key_pos and not self.has_key:
            obs += "You see a key on the floor. "
        elif self.agent_pos == self.door_pos:
            obs += "You see a large, locked door. "
        else:
            obs += "The room is empty. "
        return obs + "What is your next move?"

    def parse_action(self, action_str: str) -> str:
        try:
            action = extract_tag_value(action_str, "action").lower().strip()
        except ValueError:
            raise ValueError("The <action> tag was not found in the response.")

        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Choose from: {self.VALID_ACTIONS}")
            
        return action

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Game has ended. Call reset().")

        self.current_step += 1
        reward = 0.0
        info: Dict[str, Any] = {"action_taken": action}

        if action.startswith("move"):
            direction = action.split(" ")[1]
            r, c = self.agent_pos
            if direction == "north": r = max(0, r - 1)
            elif direction == "south": r = min(self.size - 1, r + 1)
            elif direction == "west": c = max(0, c - 1)
            elif direction == "east": c = min(self.size - 1, c + 1)
            self.agent_pos = (r, c)
            info["new_pos"] = self.agent_pos
        elif action == "interact":
            if self.agent_pos == self.key_pos and not self.has_key:
                self.has_key = True
                reward = 0.1
                info["event"] = "key_picked_up"
            elif self.agent_pos == self.door_pos and self.has_key:
                self.done = True
                reward = 1.0
                info["event"] = "door_unlocked"
            elif self.agent_pos == self.door_pos and not self.has_key:
                info["event"] = "interact_locked_door_fail"
            else:
                info["event"] = "interact_nothing"

        if self.current_step >= self.max_steps and not self.done:
            self.done = True
            reward = -1.0
            info["event"] = "max_steps_reached"

        if self.done:
            obs = "Congratulations! You unlocked the door." if reward == 1.0 else "You failed to escape."
        else:
            obs = self._get_obs()

        return obs, reward, self.done, info

# --- Interactive Test Block ---
if __name__ == "__main__":
    env = KeyDoorEnv(size=3, max_steps=20)
    obs = env.reset(seed=42)
    
    print("--- Key-Door Maze Interactive Test ---")
    print("This tests the pure environment logic.")
    print(f"DEBUG: Key is at {env.key_pos}, Door is at {env.door_pos}")
    print("-" * 20)
    
    print(f"INITIAL OBSERVATION: {obs}")

    while not env.done:
        try:
            # 1. Get a clean action from the user
            action_input = input(f"Enter action ({', '.join(env.VALID_ACTIONS)}): ").lower().strip()
            
            # 2. Simulate the raw LLM output that the RolloutGenerator will process
            #    (The environment itself doesn't use the scratchpad tag, but this
            #    tests that the parser can ignore it correctly).
            simulated_llm_output = f"<scratchpad>This is a test thought.</scratchpad><action>{action_input}</action>"

            # 3. Parse the action using the environment's method
            parsed_action = env.parse_action(simulated_llm_output)

            # 4. Step the environment with the clean, parsed action
            obs, reward, done, info = env.step(parsed_action)
            
            print("\n" + "="*20)
            print(f"ACTION TAKEN: '{info.get('action_taken')}'")
            print(f"INFO: {info}")
            print(f"REWARD: {reward}")
            print(f"OBSERVATION: {obs}")
            print("="*20 + "\n")
        
        except ValueError as e:
            print(f"\n--- PARSE/ACTION ERROR: {e} ---\n")
            continue

    print("\n--- GAME OVER ---")
    if reward == 1.0:
        print("You won!")
    else:
        print("You lost or ran out of steps.")