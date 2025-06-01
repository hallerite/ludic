from __future__ import annotations
import random
import textwrap
import os
import pygame
from typing import Dict, Tuple, Any

from ludic_envs.envs.env import Env
from ludic_envs.parsers import extract_tag_value

from functools import partial

SYSTEM_PROMPT = "You are an agent in a grid-world maze. Your goal is to find the key, use it to unlock the door, and exit."


SCRATCHPAD_INSTR = textwrap.dedent("""
    ## Your Task & Memory Instructions:
    You are an agent solving a maze. You have a short-term observation and a long-term memory scratchpad.

    ### Your Strategy:
    1.  **Explore Systematically:** Your primary goal is to explore the maze cell by cell to find the key and the door. A good strategy is to explore row by row or column by column.
    2.  **Build a Map:** Use your scratchpad to build a map of the world as you explore. Record what you find in each room. This is the most important part of your memory.

    ### How to Use Your Memory Scratchpad:
    1.  **Purpose:** Your scratchpad is for storing critical FACTS you discover, not your immediate plans or current position.
    2.  **What to Store (Examples):**
        - Your status: `Status: I do not have the key.`
        - The location of the key, once found: `Key is at (x1, y1).`
        - The location of the door, once found: `Door is at (x2, y2).`
        - Your map of explored rooms: `Already visited empty rooms: (x1,y2), (x2,y1)\n.`
    3.  **What NOT to Store (CRITICAL):**
        - **DO NOT** store your current position (e.g., `I am at (0,0)`). It is always provided in the 'Current Observation'.
        - **DO NOT** store your plans or intentions (e.g., `I will move north next`). Your final decision goes in the <action> tag.
    4.  **Behavior:** Your new scratchpad output will **COMPLETELY REPLACE** the old one. You must re-state all known facts, including your updated map, on every turn.

    ### Response Format:
    First, update your memory of the world in a `<scratchpad>` tag. Then, output your final action in an `<action>` tag.

    ### Good Example of a turn:
    <scratchpad>Status: I do not have the key. Empty Rooms: (x1,y1), (x2,y2) Room with door:(x1,y2).</scratchpad>
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
    
    def __init__(self, size: int = 4, max_steps: int = 35):
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

class PygameRenderer:
    def __init__(self, size: int, cell_size: int = 60):
        pygame.init()
        self.cell_size = cell_size
        self.screen_size = size * cell_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Key-Door Maze (Pygame)")

        # Load sprites
        res_dir = "res"
        sprite_size = int(self.cell_size * 0.8)
        try:
            self.font = pygame.font.SysFont("Arial", 24)
        except pygame.error:
            print("Default font 'Arial' not found, using pygame's default.")
            self.font = pygame.font.Font(None, 24) # Fallback to default font

        # Load sprites (logic is unchanged)
        res_dir = "res"
        sprite_size = int(self.cell_size * 0.8)
        try:
            agent_img = pygame.image.load(os.path.join(res_dir, "robot_face.png")).convert_alpha()
            key_img = pygame.image.load(os.path.join(res_dir, "key.png")).convert_alpha()
            door_img = pygame.image.load(os.path.join(res_dir, "door.png")).convert_alpha()

            self.agent_sprite = pygame.transform.scale(agent_img, (sprite_size, sprite_size))
            self.key_sprite = pygame.transform.scale(key_img, (sprite_size, sprite_size))
            self.door_sprite = pygame.transform.scale(door_img, (sprite_size, sprite_size))
        except pygame.error as e:
            print(f"Error loading sprites: {e}")
            print("Creating fallback shapes.")
            self.agent_sprite = self.key_sprite = self.door_sprite = None


    def render(self, env: KeyDoorEnv):
        self.screen.fill(pygame.Color("white")) # Clear screen
        self._draw_grid()

        # Get sprite rectangles and blit (draw) them (logic is unchanged)
        if self.agent_sprite:
            key_rect = self.key_sprite.get_rect(center=self._get_pixel_coords(env.key_pos))
            door_rect = self.door_sprite.get_rect(center=self._get_pixel_coords(env.door_pos))
            agent_rect = self.agent_sprite.get_rect(center=self._get_pixel_coords(env.agent_pos))

            if not env.has_key:
                self.screen.blit(self.key_sprite, key_rect)
            self.screen.blit(self.door_sprite, door_rect)
            self.screen.blit(self.agent_sprite, agent_rect)

        # --- NEW: Render and draw the turn counter ---
        turn_text = f"Turn: {env.current_step}/{env.max_steps}"
        text_surface = self.font.render(turn_text, True, pygame.Color("black"))
        # Position text in the top-right corner with a 10px margin
        text_rect = text_surface.get_rect(topright=(self.screen_size - 10, 10))
        self.screen.blit(text_surface, text_rect)
        # --- End of new code ---

        pygame.display.flip() # Update the full display

    def _draw_grid(self):
        for i in range(1, self.screen_size // self.cell_size):
            pygame.draw.line(self.screen, pygame.Color("gray"), (i * self.cell_size, 0), (i * self.cell_size, self.screen_size))
            pygame.draw.line(self.screen, pygame.Color("gray"), (0, i * self.cell_size), (self.screen_size, i * self.cell_size))

    def _get_pixel_coords(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        r, c = pos
        return (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2)

    def close(self):
        pygame.quit()

# --- Interactive Test Block ---
if __name__ == "__main__":
    env = KeyDoorEnv(size=4, max_steps=25)
    renderer = PygameRenderer(size=env.size) # Initialize renderer

    obs = env.reset(seed=42)
    renderer.render(env) # Initial render

    print("--- Key-Door Maze Interactive Test (Pygame) ---")
    print(f"DEBUG: Key is at {env.key_pos}, Door is at {env.door_pos}")
    print("-" * 20)
    print(f"INITIAL OBSERVATION: {obs}")

    # Main game loop
    while not env.done:
        # Pygame windows must process events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.done = True # Exit the loop if window is closed

        if env.done:
            continue

        try:
            action_input = input(f"Enter action ({', '.join(env.VALID_ACTIONS)}): ").lower().strip()
            if not action_input: continue
            
            simulated_llm_output = f"<action>{action_input}</action>"
            parsed_action = env.parse_action(simulated_llm_output)
            obs, reward, done, info = env.step(parsed_action)

            renderer.render(env) # Re-render the screen after every move
            
            print("\n" + "="*20)
            print(f"ACTION TAKEN: '{info.get('action_taken')}'"); print(f"INFO: {info}")
            print(f"REWARD: {reward}"); print(f"OBSERVATION: {obs}")
            print("="*20 + "\n")
        
        except ValueError as e:
            print(f"\n--- PARSE/ACTION ERROR: {e} ---\n")
            continue

    print("\n--- GAME OVER ---")
    renderer.close()