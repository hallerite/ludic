from __future__ import annotations
import random
import textwrap
import os
import re
import sys
import pygame
from typing import Dict, Tuple, Any, List, Optional

from ludic_envs.envs.env import Env
from ludic_envs.parsers import extract_tag_value

from functools import partial

SYSTEM_PROMPT = (
    "You are an agent in a grid-world maze."
    "Your goal is to find the key, pick it up, find the door and use the key to unlock the door."
    "Your available moves are 'up', 'down', 'left', 'right', and you can 'interact' with objects in your current location."
    "You must wrap your moves in <action>...</action>, like so <action>interact</action>"
    "Note: you have to actively pick up the key and unlock the door by using `interact`, when you see the key or door. The key is not automatically picked up and the door is not automatically unlocked."
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
        - **DO NOT** store your plans or intentions (e.g., `I will go up next`). Your final decision goes inside the <action></action> tags.
    4.  **Behavior:** Your new scratchpad output will **COMPLETELY REPLACE** the old one. You must re-state all known facts, including your updated map, on every turn.

    ### Response Format:
    First, update your memory of the world in a `<scratchpad>` tag. Then, output your final action wrapped inside <action></action> like so <action>interact</action>.

    ### Good Example of a turn:
    <scratchpad>Status: I do not have the key. Empty Rooms: (x1,y1), (x2,y2) Room with door:(x1,y2).</scratchpad>
    <action>down</action>
""")

class KeyDoorEnv(Env):
    """
    A simple POMDP environment where an agent must explore an `n \\times n` grid to
    find a key and use it to unlock a door.
    This version includes logic to track agent knowledge for an optimal scratchpad.
    """
    VALID_ACTIONS = ("up", "down", "left", "right", "interact")

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
        
        # State to track agent's discoveries
        self.visited_cells: Dict[Tuple[int, int], str] = {}

    def reset(self, seed: int | None = None) -> str:
        if seed is not None:
            random.seed(seed)
        positions = random.sample([(r, c) for r in range(self.size) for c in range(self.size)], 3)
        self.agent_pos, self.key_pos, self.door_pos = positions[0], positions[1], positions[2]
        self.has_key = False
        self.done = False
        self.current_step = 0
        
        # Reset and initialize knowledge
        self.visited_cells.clear()
        self._update_knowledge()

        return self._get_obs()

    def _update_knowledge(self):
        """Internal method to update the agent's discovered knowledge."""
        if self.agent_pos not in self.visited_cells:
            if self.agent_pos == self.key_pos:
                self.visited_cells[self.agent_pos] = "key"
            elif self.agent_pos == self.door_pos:
                self.visited_cells[self.agent_pos] = "door"
            else:
                self.visited_cells[self.agent_pos] = "empty"
    
    def get_optimal_scratchpad(self) -> str:
        """
        Generates a perfect, concise scratchpad based on the ground-truth
        of what the agent has discovered.
        """
        parts = []
        parts.append(f"Status: {'I have the key' if self.has_key else 'I do not have the key'}.")

        key_loc_str = "unknown"
        door_loc_str = "unknown"
        empty_rooms = []

        # If agent has the key, it knows the key's location is no longer relevant on the map
        key_pos_if_known = self.key_pos if not self.has_key else None

        for pos, item in self.visited_cells.items():
            if pos == key_pos_if_known and item == "key":
                key_loc_str = str(pos)
            elif item == "door":
                door_loc_str = str(pos)
            elif item == "empty":
                empty_rooms.append(str(pos))
        
        parts.append(f"Key location: {key_loc_str}.")
        parts.append(f"Door location: {door_loc_str}.")
        
        if empty_rooms:
            parts.append(f"Visited empty rooms: {', '.join(sorted(map(str, empty_rooms)))}.")
        
        return " ".join(parts)

    def _get_obs(self) -> str:
        # This method is unchanged
        obs_parts = [
            f"You are at position {self.agent_pos} in a {self.size}x{self.size} grid. ",
            f"You are {'carrying the key' if self.has_key else 'not carrying a key'}. "
        ]
        if self.agent_pos == self.key_pos and not self.has_key:
            obs_parts.append("You see a key on the floor. ")
        elif self.agent_pos == self.door_pos:
            obs_parts.append("You see a large, locked door. ")
        else:
            obs_parts.append("The room you are currently in is empty. ")
        
        r, c = self.agent_pos
        available_actions_description = []
        if r > 0:
            available_actions_description.append(f"up to ({r-1}, {c})")
        if r < self.size - 1:
            available_actions_description.append(f"down to ({r+1}, {c})")
        if c > 0:
            available_actions_description.append(f"left to ({r}, {c-1})")
        if c < self.size - 1:
            available_actions_description.append(f"right to ({r}, {c+1})")
        available_actions_description.append("interact")
        
        obs_parts.append(f"Your available actions are: {', '.join(available_actions_description)}. ")
        return "".join(obs_parts) + "What is your next move?"

    def step(self, action_str: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self.done:
            return "The game has already ended. Please call reset().", 0.0, True, {"event": "action_after_done", "raw_input": action_str}

        info: Dict[str, Any] = {"raw_input": action_str}
        try:
            parsed_action = self.parse_action(action_str)
            info["action_taken"] = parsed_action
        except ValueError as e:
            self.current_step += 1
            error_feedback = f"Your input was invalid: {str(e)}. "
            if self.current_step >= self.max_steps:
                self.done = True
                final_message = error_feedback + self._get_obs().replace(" What is your next move?", "") + " You also ran out of time."
                info.update({"event": "max_steps_reached_on_invalid", "error": str(e)})
                return final_message, -0.5, True, info
            return error_feedback + self._get_obs(), 0.0, False, {"illegal_move_or_parse_error": True, "error": str(e)}

        self.current_step += 1
        reward = 0.0
        action_feedback = ""
        r_old, c_old = self.agent_pos

        if parsed_action in ("up", "down", "left", "right"):
            r, c = self.agent_pos
            if parsed_action == "up": r = max(0, r - 1)
            elif parsed_action == "down": r = min(self.size - 1, r + 1)
            elif parsed_action == "left": c = max(0, c - 1)
            elif parsed_action == "right": c = min(self.size - 1, c + 1)
            self.agent_pos = (r, c)
            info["new_pos"] = self.agent_pos
            action_feedback = f"You tried to move {parsed_action} but bumped into a wall. " if (r_old, c_old) == self.agent_pos else f""
            self._update_knowledge() # NEW: Update knowledge after moving
        elif parsed_action == "interact":
            action_feedback = "You chose to interact. "
            if self.agent_pos == self.key_pos and not self.has_key:
                self.has_key = True; reward = 0.1; info["event"] = "key_picked_up"; action_feedback += "You picked up the key! "
                # Now that the key is picked up, where the key *was* is now an empty room.
                self.visited_cells[self.agent_pos] = "empty"
            elif self.agent_pos == self.door_pos and self.has_key:
                self.done = True; reward = 1.0; info["event"] = "door_unlocked"; action_feedback += "You unlocked the door! "
            elif self.agent_pos == self.door_pos and not self.has_key:
                info["event"] = "interact_locked_door_fail"; action_feedback += "The door is locked, and you don't have the key. "
            else:
                info["event"] = "interact_nothing"; action_feedback += "There was nothing to interact with here. "

        if not self.done and self.current_step >= self.max_steps:
            self.done = True; reward = -0.5; info["event"] = "max_steps_reached"; action_feedback += "You ran out of time. "
        
        if self.done:
            obs = f"{action_feedback.strip()} {self._get_obs().replace(' What is your next move?', '').strip()} {'Congratulations, you escaped the maze!' if info.get('event') == 'door_unlocked' else 'Game over.'}".strip()
        else:
            obs = action_feedback + self._get_obs()
        return obs, reward, self.done, info

    def parse_action(self, action_str: str) -> str:
        # This method is unchanged
        try:
            action = extract_tag_value(action_str, "action").lower().strip()
        except ValueError:
            match = re.search(r'<action>(.*?)(?:</action>|$)', action_str, re.DOTALL)
            if match:
                action = match.group(1).lower().strip()
                print(f"INFO: Used lenient parsing for malformed action tag. Extracted '{action}' from '{action_str.strip()}'")
            else:
                raise ValueError("You need to wrap your action in <action>...</action>!")
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Choose from: {self.VALID_ACTIONS}")
        return action

class PygameRenderer:
    def __init__(self, grid_size: int, cell_size: int = 120, log_width: int = 400, log_font_size: int = 14):
        pygame.init()
        self.cell_size = cell_size
        self.grid_size = grid_size
        
        self.top_panel_height = 60 

        self.game_area_width = grid_size * cell_size
        self.game_area_height = grid_size * cell_size
        self.log_width = log_width
        self.screen_width = self.game_area_width + self.log_width
        self.screen_height = self.game_area_height + self.top_panel_height

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Key-Door Maze Evaluation")

        self.top_panel_rect = pygame.Rect(0, 0, self.game_area_width, self.top_panel_height)
        self.game_area_rect = pygame.Rect(0, self.top_panel_height, self.game_area_width, self.game_area_height)
        self.log_area_rect = pygame.Rect(self.game_area_width, 0, self.log_width, self.screen_height)

        try:
            self.game_font = pygame.font.SysFont("Arial", 20)
            self.log_font = pygame.font.SysFont("monospace", log_font_size)
            self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
            self.transition_font = pygame.font.SysFont("Arial", 40, bold=True)
        except pygame.error:
            self.game_font = pygame.font.Font(None, 24)
            self.log_font = pygame.font.Font(None, log_font_size + 2)
            self.title_font = pygame.font.Font(None, 30)
            self.transition_font = pygame.font.Font(None, 46)

        self.line_height = self.log_font.get_linesize()
        self.conversation_log_lines: List[str] = []
        self.scroll_offset_y_pixels = 0
        self.log_padding = 5
        self.title = ""
        self.agent_sprite = self.key_sprite = self.door_sprite = None
        self._load_sprites()

    def set_title(self, new_title: str):
        self.title = new_title

    def show_transition_screen(self, main_text: str, duration_sec: int = 3):
        # Draw the transition message only in the main game area
        game_area_surface = self.screen.subsurface(self.game_area_rect)
        game_area_surface.fill(pygame.Color("midnightblue"))
        main_surf = self.transition_font.render(main_text, True, pygame.Color("white"))
        main_rect = main_surf.get_rect(center=(self.game_area_width // 2, self.game_area_height // 2))
        game_area_surface.blit(main_surf, main_rect)

        # Keep the top panel and log visible during the transition
        top_panel_surface = self.screen.subsurface(self.top_panel_rect)
        top_panel_surface.fill(pygame.Color("dimgrey"))
        transition_title_surf = self.title_font.render("Preparing Next Game...", True, pygame.Color("white"))
        transition_title_rect = transition_title_surf.get_rect(center=(self.game_area_width // 2, self.top_panel_height // 2))
        top_panel_surface.blit(transition_title_surf, transition_title_rect)

        log_display_surface = self.screen.subsurface(self.log_area_rect)
        self._render_log_content(log_display_surface)
        
        pygame.display.flip()
        
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < duration_sec * 1000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.time.Clock().tick(30)

    def render(self, env: Optional[KeyDoorEnv] = None):
        self.screen.fill(pygame.Color("darkgrey"))

        # --- 1. Render Top Panel ---
        top_panel_surface = self.screen.subsurface(self.top_panel_rect)
        top_panel_surface.fill(pygame.Color("dimgrey"))

        if self.title:
            title_surf = self.title_font.render(self.title, True, pygame.Color("white"))
            title_rect = title_surf.get_rect(centery=self.top_panel_height // 2, left=20)
            top_panel_surface.blit(title_surf, title_rect)

        if env:
            turn_text = f"Turn: {env.current_step}/{env.max_steps}"
            text_surf = self.game_font.render(turn_text, True, pygame.Color("lightgray"))
            text_r = text_surf.get_rect(centery=self.top_panel_height // 2, right=self.game_area_width - 20)
            top_panel_surface.blit(text_surf, text_r)

        # --- 2. Render Game Area ---
        game_area_surface = self.screen.subsurface(self.game_area_rect)
        game_area_surface.fill(pygame.Color("white"))
        
        if env:
            # These functions draw onto the 'game_area_surface' subsurface
            self._draw_grid(game_area_surface)
            if self.agent_sprite: self._draw_sprites(game_area_surface, env)
            else: self._draw_fallback_shapes(game_area_surface, env)

        # --- 3. Render Log Area ---
        log_display_surface = self.screen.subsurface(self.log_area_rect)
        self._render_log_content(log_display_surface)
        
        pygame.display.flip()

    def _load_sprites(self):
        try:
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            res_dir = "res"
            sprite_size = int(self.cell_size * 0.8)
            agent_img = pygame.image.load(os.path.join(res_dir, "robot_face.png")).convert_alpha()
            key_img = pygame.image.load(os.path.join(res_dir, "key.png")).convert_alpha()
            door_img = pygame.image.load(os.path.join(res_dir, "door.png")).convert_alpha()
            self.agent_sprite = pygame.transform.scale(agent_img, (sprite_size, sprite_size))
            self.key_sprite = pygame.transform.scale(key_img, (sprite_size, sprite_size))
            self.door_sprite = pygame.transform.scale(door_img, (sprite_size, sprite_size))
        except (pygame.error, FileNotFoundError) as e:
            print(f"Warning: Could not load sprites. Using fallback shapes. Error: {e}", file=sys.stderr)
    def clear_log(self):
        self.conversation_log_lines = []; self.scroll_offset_y_pixels = 0
    def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> List[str]:
        lines = []
        for paragraph in text.split('\n'):
            words = paragraph.split(' ')
            current_line = ""
            for word in words:
                test_line = current_line + word + " "
                if font.size(test_line)[0] <= max_width: current_line = test_line
                else: lines.append(current_line.strip()); current_line = word + " "
            lines.append(current_line.strip())
        return lines
    def add_log_entry(self, prompt_str: str, completion_str: str):
        max_text_width = self.log_width - 2 * self.log_padding
        self.conversation_log_lines.extend(self._wrap_text(f"OBS: {prompt_str}", self.log_font, max_text_width))
        self.conversation_log_lines.extend(self._wrap_text(f"AGENT: {completion_str}", self.log_font, max_text_width))
        separator_line = "─" * (max_text_width // self.log_font.size("─")[0] if self.log_font.size("─")[0] > 0 else 20)
        self.conversation_log_lines.append(separator_line)
        total_content_height = len(self.conversation_log_lines) * self.line_height
        visible_log_height = self.log_area_rect.height - (2 * self.log_padding)
        if total_content_height > visible_log_height:
            self.scroll_offset_y_pixels = total_content_height - visible_log_height
    def handle_scroll_event(self, event):
        if event.type == pygame.MOUSEWHEEL:
            self.scroll_offset_y_pixels -= event.y * self.line_height * 2
            total_content_height = len(self.conversation_log_lines) * self.line_height
            visible_log_height = self.log_area_rect.height - (2 * self.log_padding)
            max_scroll = max(0, total_content_height - visible_log_height)
            self.scroll_offset_y_pixels = max(0, min(self.scroll_offset_y_pixels, max_scroll))
    def _render_log_content(self, surface: pygame.Surface):
        surface.fill(pygame.Color("gainsboro"))
        y_pos = self.log_padding - self.scroll_offset_y_pixels
        for line_text in self.conversation_log_lines:
            if y_pos + self.line_height > 0 and y_pos < self.log_area_rect.height:
                text_surface = self.log_font.render(line_text, True, pygame.Color("black"))
                surface.blit(text_surface, (self.log_padding, y_pos))
            y_pos += self.line_height
            if y_pos > self.log_area_rect.height: break
    def _draw_sprites(self, surface: pygame.Surface, env: KeyDoorEnv):
        agent_r = self.agent_sprite.get_rect(center=self._get_pixel_coords(env.agent_pos))
        door_r = self.door_sprite.get_rect(center=self._get_pixel_coords(env.door_pos))
        surface.blit(self.door_sprite, door_r)
        if not env.has_key:
            key_r = self.key_sprite.get_rect(center=self._get_pixel_coords(env.key_pos))
            surface.blit(self.key_sprite, key_r)
        surface.blit(self.agent_sprite, agent_r)
    def _draw_fallback_shapes(self, surface: pygame.Surface, env: KeyDoorEnv):
        s = self.cell_size // 2
        ax, ay = self._get_pixel_coords(env.agent_pos)
        pygame.draw.rect(surface, (0,0,255), (ax - s//2, ay - s//2, s, s))
        if not env.has_key:
            kx, ky = self._get_pixel_coords(env.key_pos)
            pygame.draw.rect(surface, (255,255,0), (kx-s//2, ky-s//2, s,s))
        dx, dy = self._get_pixel_coords(env.door_pos)
        pygame.draw.rect(surface, (139,69,19), (dx-s//2, dy-s//2, s,s))
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
    env = KeyDoorEnv(size=4, max_steps=35)
    renderer = PygameRenderer(grid_size=env.size)

    obs = env.reset(seed=42)
    renderer.render(env)

    print("--- Key-Door Maze Interactive Test (Pygame) ---")
    print(f"DEBUG: Key is at {env.key_pos}, Door is at {env.door_pos}")
    print("-" * 20)
    print(f"INITIAL OBSERVATION:\n{obs}")

    is_running = True
    while not env.done and is_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
            renderer.handle_scroll_event(event)
        
        renderer.render(env) # Keep rendering to process events

        try:
            action_input = input(f"\nEnter action ({', '.join(env.VALID_ACTIONS)}): ").lower().strip()
            if not action_input: continue
            if not is_running: break
            
            simulated_llm_output = f"<action>{action_input}</action>"
            
            obs, reward, done, info = env.step(simulated_llm_output)

            renderer.add_log_entry(info['raw_input'], obs)
            renderer.render(env)
            
            print("\n" + "="*20)
            print(f"ACTION TAKEN: '{info.get('action_taken')}'"); print(f"INFO: {info}")
            print(f"REWARD: {reward}"); print(f"OBSERVATION:\n{obs}")
            print("="*20 + "\n")
        
        except (ValueError, KeyboardInterrupt) as e:
            print(f"\n--- INPUT ERROR / INTERRUPT: {e} ---\n")
            is_running = False
            continue

    print("\n--- GAME OVER ---")
    renderer.close()
    sys.exit()