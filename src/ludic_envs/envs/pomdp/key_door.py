from __future__ import annotations
import random
import textwrap
import os
import pygame
from typing import Dict, Tuple, Any

from ludic_envs.envs.env import Env
from ludic_envs.parsers import extract_tag_value

from functools import partial

SYSTEM_PROMPT = (
    "You are an agent in a grid-world maze."
    "Your goal is to find the key, pick it up, find the door and use the key to unlock the door."
    "Note: you have to actively pick up the key and unlock the door by using `interact`, when you see see the key or door. The key is not automatically picked up and the door is not automatically unlocked."
    )


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


    def step(self, action_str: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self.done:
            return "The game has already ended. Please call reset().", 0.0, True, {"event": "action_after_done", "raw_input": action_str}

        # Initialize info with the raw input string
        info: Dict[str, Any] = {"raw_input": action_str}
        parsed_action: str  # This will hold the successfully parsed action command

        try:
            parsed_action = self.parse_action(action_str)
            info["action_taken"] = parsed_action
        except ValueError as e:
            self.current_step += 1
            
            error_feedback = f"Your input was invalid: {str(e)}. "
            
            if self.current_step >= self.max_steps:
                self.done = True
                current_game_obs_summary = self._get_obs().replace(" What is your next move?", "")
                final_message = error_feedback + current_game_obs_summary + " You also ran out of time."
                info.update({"event": "max_steps_reached_on_invalid", "error": str(e)})
                return final_message, -0.5, True, info # Penalize for ending due to invalid actions
            
            # If not max_steps, return current observation with error feedback
            obs_with_error = error_feedback + self._get_obs()
            info.update({"illegal_move_or_parse_error": True, "error": str(e)})
            return obs_with_error, 0.0, False, info

        # If parsing succeeded, proceed with game logic using 'parsed_action'
        self.current_step += 1
        reward = 0.0
        action_feedback = "" # For augmenting the obs given to the agent.

        if parsed_action.startswith("move"):
            direction = parsed_action.split(" ")[1]
            r_old, c_old = self.agent_pos
            r, c = self.agent_pos

            if direction == "north": r = max(0, r - 1)
            elif direction == "south": r = min(self.size - 1, r + 1)
            elif direction == "west": c = max(0, c - 1)
            elif direction == "east": c = min(self.size - 1, c + 1)
            
            self.agent_pos = (r, c)
            info["new_pos"] = self.agent_pos

            if (r_old, c_old) == self.agent_pos:
                action_feedback = f"You tried to move {direction} but bumped into a wall. "
            else:
                action_feedback = f"You moved {direction}. " # Feedback for successful move

        elif parsed_action == "interact":
            action_feedback = "You chose to interact. "
            if self.agent_pos == self.key_pos and not self.has_key:
                self.has_key = True
                reward = 0.1
                info["event"] = "key_picked_up"
                action_feedback += "You picked up the key! "
            elif self.agent_pos == self.door_pos and self.has_key:
                self.done = True
                reward = 1.0
                info["event"] = "door_unlocked"
                action_feedback += "You unlocked the door! "
            elif self.agent_pos == self.door_pos and not self.has_key:
                info["event"] = "interact_locked_door_fail"
                action_feedback += "The door is locked, and you don't have the key. "
            else:
                info["event"] = "interact_nothing"
                action_feedback += "There was nothing to interact with here. "
        # The 'else' case for an unrecognized 'parsed_action' is implicitly handled because
        # self.parse_action should raise a ValueError if 'action_str' doesn't map to a valid command.

        if not self.done and self.current_step >= self.max_steps:
            self.done = True
            reward = -0.5 
            info["event"] = "max_steps_reached"
            # Prepend to existing feedback, or set if action_feedback was for a non-eventful last action
            if action_feedback and not action_feedback.isspace():
                 action_feedback += "Additionally, you ran out of time. "
            else:
                action_feedback = "You ran out of time. "


        if self.done:
            # Get a summary of the current state for the final message, without the "What is your next move?" part.
            current_game_obs_summary = self._get_obs().replace(" What is your next move?", "").strip()
            
            final_status_message = ""
            if info.get("event") == "door_unlocked":
                final_status_message = "Congratulations, you escaped the maze!"
            elif info.get("event") == "max_steps_reached":
                # action_feedback already contains "You ran out of time."
                final_status_message = "Game over."
            # Add other specific terminal conditions if necessary
            else: 
                final_status_message = "The game has ended." # Generic fallback for other done states

            # Combine action feedback (what just happened) with current state summary and final status
            obs = f"{action_feedback.strip()} {current_game_obs_summary} {final_status_message}".strip()
        else:
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
    def __init__(self, grid_size: int, cell_size: int = 500, log_width: int = 800, log_font_size: int = 14):
        pygame.init()
        self.cell_size = cell_size
        self.grid_size = grid_size

        self.game_area_width = grid_size * cell_size
        self.log_width = log_width
        self.screen_width = self.game_area_width + self.log_width
        self.screen_height = grid_size * cell_size
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Key-Door Maze (with Conversation Log)")

        # Fonts
        try:
            self.game_font = pygame.font.SysFont("Arial", 20) # For turn counter
            self.log_font = pygame.font.SysFont("monospace", log_font_size)
        except pygame.error:
            print("Default font 'Arial' or 'monospace' not found, using pygame's default.", file=sys.stderr)
            self.game_font = pygame.font.Font(None, 24) # Fallback for game font
            self.log_font = pygame.font.Font(None, log_font_size) # Fallback for log font
        self.line_height = self.log_font.get_linesize()

        # Log area
        self.log_area_rect = pygame.Rect(self.game_area_width, 0, self.log_width, self.screen_height)
        self.conversation_log_lines: List[str] = [] 
        self.scroll_offset_y_pixels = 0 # How many pixels the log content is scrolled up
        self.log_padding = 5

        # Sprite loading
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        res_dir_relative = "res" # Path relative to this file

        sprite_size = int(self.cell_size * 0.8)
        try:
            agent_img = pygame.image.load(os.path.join(res_dir_relative, "robot_face.png")).convert_alpha()
            key_img = pygame.image.load(os.path.join(res_dir_relative, "key.png")).convert_alpha()
            door_img = pygame.image.load(os.path.join(res_dir_relative, "door.png")).convert_alpha()

            self.agent_sprite = pygame.transform.scale(agent_img, (sprite_size, sprite_size))
            self.key_sprite = pygame.transform.scale(key_img, (sprite_size, sprite_size))
            self.door_sprite = pygame.transform.scale(door_img, (sprite_size, sprite_size))
        except pygame.error as e:
            print(f"Error loading sprites from '{os.path.abspath(res_dir_relative)}': {e}", file=sys.stderr)
            print("Displaying fallback shapes if sprites fail to load.", file=sys.stderr)
            self.agent_sprite = self.key_sprite = self.door_sprite = None # Fallback

    def clear_log(self):
        self.conversation_log_lines = []
        self.scroll_offset_y_pixels = 0

    def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> List[str]:
        lines = []
        for paragraph in text.split('\n'): # Handle existing newlines
            words = paragraph.split(' ')
            current_line = ""
            for word in words:
                test_line = current_line + word + " "
                if font.size(test_line)[0] <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line.strip())
                    current_line = word + " "
            lines.append(current_line.strip())
        return lines

    def add_log_entry(self, prompt_str: str, completion_str: str):
        max_text_width = self.log_width - 2 * self.log_padding
        
        self.conversation_log_lines.extend(self._wrap_text(f"{prompt_str}", self.log_font, max_text_width))
        self.conversation_log_lines.extend(self._wrap_text(f"{completion_str}", self.log_font, max_text_width))
        separator_line = "─" * (max_text_width // self.log_font.size("─")[0] if self.log_font.size("─")[0] > 0 else 20)
        self.conversation_log_lines.append(separator_line)

        # --- Auto-scroll to make the newest entry visible ---
        total_content_pixel_height = len(self.conversation_log_lines) * self.line_height
        visible_log_area_pixel_height = self.log_area_rect.height - (2 * self.log_padding)

        if total_content_pixel_height > visible_log_area_pixel_height:
            # Scroll so the bottom of the content is at the bottom of the visible area
            self.scroll_offset_y_pixels = total_content_pixel_height - visible_log_area_pixel_height
        else:
            # If content is shorter than the view, no scroll needed (top of content at top of view)
            self.scroll_offset_y_pixels = 0
        # --- End of auto-scroll logic ---

    def handle_scroll_event(self, event):
        if event.type == pygame.MOUSEWHEEL:
            # Adjust scroll based on mouse wheel direction and speed
            # event.y is typically 1 for scroll up, -1 for scroll down
            self.scroll_offset_y_pixels -= event.y * self.line_height * 2 # Scroll 2 lines at a time

            # Clamp scroll_offset_y_pixels to valid range
            total_content_pixel_height = len(self.conversation_log_lines) * self.line_height
            visible_log_area_pixel_height = self.log_area_rect.height - (2 * self.log_padding)
            
            # Max scroll offset (how much the content can be scrolled upwards)
            max_scroll_offset = total_content_pixel_height - visible_log_area_pixel_height
            if max_scroll_offset < 0: 
                max_scroll_offset = 0 # Content is shorter than visible area, no scroll possible
            
            self.scroll_offset_y_pixels = max(0, min(self.scroll_offset_y_pixels, max_scroll_offset))


    def _render_log_content(self, surface: pygame.Surface):
        surface.fill(pygame.Color("gainsboro")) # Log background
        
        y_pos = self.log_padding - self.scroll_offset_y_pixels 

        for line_text in self.conversation_log_lines:
            # Only render lines that might be visible
            if y_pos + self.line_height > 0 and y_pos < self.log_area_rect.height:
                text_surface = self.log_font.render(line_text, True, pygame.Color("black"))
                surface.blit(text_surface, (self.log_padding, y_pos))
            y_pos += self.line_height
            if y_pos > self.log_area_rect.height: # Optimization: stop if past visible area
                break
        
    def render(self, env: Optional[KeyDoorEnv] = None):
        self.screen.fill(pygame.Color("darkgrey")) # Overall background for borders

        # Game Area
        game_area_surface = self.screen.subsurface(pygame.Rect(0, 0, self.game_area_width, self.screen_height))
        game_area_surface.fill(pygame.Color("white")) 
        
        if env:
            self._draw_grid(game_area_surface)

            if self.agent_sprite: # Check if sprites loaded correctly
                key_r = self.key_sprite.get_rect(center=self._get_pixel_coords(env.key_pos))
                door_r = self.door_sprite.get_rect(center=self._get_pixel_coords(env.door_pos))
                agent_r = self.agent_sprite.get_rect(center=self._get_pixel_coords(env.agent_pos))

                if not env.has_key:
                    game_area_surface.blit(self.key_sprite, key_r)
                game_area_surface.blit(self.door_sprite, door_r)
                game_area_surface.blit(self.agent_sprite, agent_r)
            else: # Fallback rendering if sprites failed
                self._draw_fallback_shapes(game_area_surface, env)


            turn_text = f"Turn: {env.current_step}/{env.max_steps}"
            text_surf = self.game_font.render(turn_text, True, pygame.Color("black"))
            text_r = text_surf.get_rect(topright=(self.game_area_width - 10, 5))
            game_area_surface.blit(text_surf, text_r)

        # Log Area
        log_display_surface = self.screen.subsurface(self.log_area_rect)
        self._render_log_content(log_display_surface)
        
        pygame.display.flip()

    def _draw_fallback_shapes(self, surface: pygame.Surface, env: KeyDoorEnv):
        # Simple colored squares as fallback
        agent_color = (0,0,255) # Blue
        key_color = (255,255,0) # Yellow
        door_color = (139,69,19) # Brown

        s = self.cell_size // 2
        ax, ay = self._get_pixel_coords(env.agent_pos)
        pygame.draw.rect(surface, agent_color, (ax - s//2, ay - s//2, s, s))
        
        if not env.has_key:
            kx, ky = self._get_pixel_coords(env.key_pos)
            pygame.draw.rect(surface, key_color, (kx-s//2, ky-s//2, s,s))
        
        dx, dy = self._get_pixel_coords(env.door_pos)
        pygame.draw.rect(surface, door_color, (dx-s//2, dy-s//2, s,s))


    def _draw_grid(self, surface: pygame.Surface):
        for i in range(1, self.grid_size):
            pygame.draw.line(surface, pygame.Color("gray"), (i * self.cell_size, 0), (i * self.cell_size, self.screen_height))
            pygame.draw.line(surface, pygame.Color("gray"), (0, i * self.cell_size), (self.game_area_width, i * self.cell_size))

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