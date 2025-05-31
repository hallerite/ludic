from __future__ import annotations
import random
import textwrap
from typing import Dict, Tuple, Any

from ludic_envs.envs.env import Env
from ludic_envs.parsers import extract_tag_value

import tkinter as tk
from functools import partial

SYSTEM_PROMPT = "You are an agent in a grid-world maze. Your goal is to find the key, use it to unlock the door, and exit."


SCRATCHPAD_INSTR = textwrap.dedent("""

    ## Your Task & Memory Instructions:
    You are an agent solving a maze. You have a short-term observation and a long-term memory scratchpad.

    ### How to Use Your Memory Scratchpad:
    1.  **Purpose:** Your scratchpad is for storing critical FACTS you discover, not your immediate plans.
    2.  **What to Store (Examples):**
        - Location of the key once you find it.
        - Location of the door once you find it.
        - Your status (e.g., `Status: I have the key`).
    3.  **What NOT to Store (CRITICAL):
        - DO NOT store your current position. It is always in the 'Current Observation'.
        - DO NOT store your plan for the next move (e.g., 'I will move north').
    4.  **Behavior:** Your new scratchpad output will COMPLETELY REPLACE the old one.

    ### Response Format:
    First, think in a `<scratchpad>` tag, updating your memory with new facts. Then, output your final action in an `<action>` tag.

    ### Good Example:
    <scratchpad>Fact: Key is at (x1,y1). Fact: Door is at (x2,y2). Status: I have the key. Goal: Get to door at (2,2).</scratchpad>
    <action>move south</action>
""")

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

        self.system_prompt = SYSTEM_PROMPT
        self.scratchpad_instr = SCRATCHPAD_INSTR

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
        """
        Generates the observation string for the agent based on the current state.
        This method now focuses solely on describing the state.
        """
        # This method is called when the game is NOT done.
        # If self.done is True, the step method will handle the final message.

        obs_parts = []
        obs_parts.append(f"You are at position {self.agent_pos} in a {self.size}x{self.size} grid. ")
        obs_parts.append(f"You are {'carrying the key' if self.has_key else 'not carrying a key'}. ")
        
        # Describe what's in the current cell
        if self.agent_pos == self.key_pos and not self.has_key:
            obs_parts.append("You see a key on the floor. ")
        elif self.agent_pos == self.door_pos:
            obs_parts.append("You see a large, locked door. ")
        else:
            obs_parts.append("The room is empty. There is no locked door here. ")
            
        current_observation = "".join(obs_parts)
        return current_observation + "What is your next move?"


    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self.done:
            # This should ideally not be reached if the game loop stops on done,
            # but it's good practice to keep it.
            return "The game has already ended. Please call reset().", 0.0, True, {"event": "action_after_done"}

        self.current_step += 1
        reward = 0.0
        info: Dict[str, Any] = {"action_taken": action}
        
        action_feedback = "" # To store messages like "You bumped into a wall."

        if action.startswith("move"):
            direction = action.split(" ")[1]
            r_old, c_old = self.agent_pos # Store position before moving
            r, c = self.agent_pos

            # Update based on your chosen action space (north/south or up/down)
            # Assuming "north", "south", "east", "west" for now as per your latest code.
            # If you switched to "up", "down", "left", "right", adjust these conditions.
            if direction == "north": r = max(0, r - 1)
            elif direction == "south": r = min(self.size - 1, r + 1)
            elif direction == "west": c = max(0, c - 1)
            elif direction == "east": c = min(self.size - 1, c + 1)
            
            self.agent_pos = (r, c)
            info["new_pos"] = self.agent_pos

            # Check if the agent tried to move but its position didn't change (hit a wall)
            if (r_old, c_old) == self.agent_pos:
                action_feedback = f"You tried to move {direction} but bumped into a wall. "
            else:
                action_feedback = f"You moved {direction}. "

        elif action == "interact":
            action_feedback = "You chose to interact. " # Base feedback for interact
            if self.agent_pos == self.key_pos and not self.has_key:
                self.has_key = True
                reward = 0.1 # Small reward for picking up the key
                info["event"] = "key_picked_up"
                action_feedback += "You picked up the key! "
            elif self.agent_pos == self.door_pos and self.has_key:
                self.done = True
                reward = 1.0 # Max reward for unlocking the door
                info["event"] = "door_unlocked"
                action_feedback += "You unlocked the door! "
            elif self.agent_pos == self.door_pos and not self.has_key:
                info["event"] = "interact_locked_door_fail"
                action_feedback += "The door is locked, and you don't have the key. "
            else:
                info["event"] = "interact_nothing"
                action_feedback += "There was nothing to interact with here. "
        else:
            # This case should ideally be caught by parse_action, but as a fallback:
            action_feedback = f"The action '{action}' is not recognized. "
            info["event"] = "unknown_action"


        # Check for game end due to max steps
        if not self.done and self.current_step >= self.max_steps:
            self.done = True
            reward = -0.5 # Negative reward for running out of time without solving
            info["event"] = "max_steps_reached"
            action_feedback += "You ran out of time. "


        # Construct the final observation message
        if self.done:
            if info.get("event") == "door_unlocked":
                final_message = "Congratulations, you escaped the maze!"
            elif info.get("event") == "max_steps_reached":
                final_message = "Game over."
            else: # Generic failure if done for other reasons (should not happen with current logic)
                final_message = "The game has ended."
            obs = action_feedback + final_message
        else:
            # Prepend the action feedback to the standard observation
            obs = action_feedback + self._get_obs()

        return obs, reward, self.done, info

    def parse_action(self, action_str: str) -> str:
        try:
            action = extract_tag_value(action_str, "action").lower().strip()
        except ValueError:
            raise ValueError("The <action> tag was not found in the response.")

        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Choose from: {self.VALID_ACTIONS}")
            
        return action

    def render(self):
        """
        Renders the current state of the grid to the console.
        - 'A' for Agent
        - 'K' for Key
        - 'D' for Door
        - '.' for empty space
        """
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        
        # Place the door
        grid[self.door_pos[0]][self.door_pos[1]] = "D"
        
        # Place the key if it hasn't been picked up
        if not self.has_key:
            grid[self.key_pos[0]][self.key_pos[1]] = "K"
        
        # Place the agent
        grid[self.agent_pos[0]][self.agent_pos[1]] = "A"

        # Print the grid
        print("\n--- Grid View ---")
        for row in grid:
            print(" ".join(row))
        print("-----------------\n")

def _graphical_render_patch(env):
    """The new render function that will replace the old one."""
    if not hasattr(env, 'window') or env.window is None:
        return  # Do nothing if the window isn't initialized

    # Helper to convert grid coords to pixel coords
    def get_canvas_coords(pos: Tuple[int, int]):
        r, c = pos
        return (c * env.cell_size + env.cell_size // 2, 
                r * env.cell_size + env.cell_size // 2)

    # Place door and key
    env.canvas.coords(env.door_obj, get_canvas_coords(env.door_pos))
    env.canvas.coords(env.key_obj, get_canvas_coords(env.key_pos))
    
    # Show/hide key based on state
    env.canvas.itemconfig(env.key_obj, state='hidden' if env.has_key else 'normal')
    
    # Move agent
    x, y = get_canvas_coords(env.agent_pos)
    radius = env.cell_size // 3
    env.canvas.coords(env.agent_obj, x - radius, y - radius, x + radius, y + radius)
    
    env.window.update()

def _close_window_patch(env):
    """The new close function to manage the window's lifecycle."""
    if hasattr(env, 'window') and env.window:
        print("\nGame Over. Close the graphical window to exit.")
        env.window.mainloop()

def patch_in_renderer(env: KeyDoorEnv):
    """
    Dynamically adds tkinter rendering capabilities to an existing KeyDoorEnv instance.
    This is the core of the "patching" logic.
    """
    env.cell_size = 60
    env.window = tk.Tk()
    env.window.title("Key-Door Maze (Patched)")
    canvas_size = env.size * env.cell_size
    env.canvas = tk.Canvas(env.window, width=canvas_size, height=canvas_size, bg='white')
    env.canvas.pack()

    for i in range(1, env.size):
        env.canvas.create_line(i * env.cell_size, 0, i * env.cell_size, canvas_size, fill='gray')
        env.canvas.create_line(0, i * env.cell_size, canvas_size, i * env.cell_size, fill='gray')

    env.agent_obj = env.canvas.create_oval(0, 0, 0, 0, fill='blue', outline='blue')
    env.key_obj = env.canvas.create_text(0, 0, text="🔑", font=("Arial", 24))
    env.door_obj = env.canvas.create_text(0, 0, text="🚪", font=("Arial", 24))
    
    # Replace the instance's original render and add a close method
    env.render = partial(_graphical_render_patch, env)
    env.close = partial(_close_window_patch, env)


# --- Interactive Test Block ---
if __name__ == "__main__":
    # 1. Create the environment as usual
    env = KeyDoorEnv(size=4, max_steps=25)
    
    # 2. Patch the graphical renderer into the instance
    patch_in_renderer(env)
    
    # 3. Reset the environment and render the initial state
    obs = env.reset(seed=42)
    env.render() # This now calls the patched graphical renderer
    
    print("--- Key-Door Maze Interactive Test ---")
    print("This tests the pure environment logic.")
    print(f"DEBUG: Key is at {env.key_pos}, Door is at {env.door_pos}")
    print("-" * 20)
    print(f"INITIAL OBSERVATION: {obs}")

    while not env.done:
        try:
            action_input = input(f"Enter action ({', '.join(env.VALID_ACTIONS)}): ").lower().strip()
            if not action_input:
                continue
            
            simulated_llm_output = f"<scratchpad>This is a test thought.</scratchpad><action>{action_input}</action>"
            parsed_action = env.parse_action(simulated_llm_output)
            obs, reward, done, info = env.step(parsed_action)

            # 4. The render call now updates the GUI
            env.render()
            
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
    
    # 5. Call the new close method to keep the window open until the user closes it
    if hasattr(env, 'close'):
        env.close()