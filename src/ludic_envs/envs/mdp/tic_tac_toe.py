from __future__ import annotations
import random
import re
from typing import Dict, List, Tuple, Optional
from ludic_envs.envs.env import Env
from ludic_envs.parsers import extract_tag_value
from pydantic import BaseModel, Field
from enum import Enum

# --- Semantic Action Space using Enums ---
class Height(str, Enum):
    TOP = "Top"
    CENTER = "Center"
    BOTTOM = "Bottom"

class Width(str, Enum):
    LEFT = "Left"
    CENTER = "Center"
    RIGHT = "Right"

class Action(BaseModel):
    """Validated player move, combining height and width."""
    height: Height
    width: Width

class TicTacToe(Env):
    """
    ─────────────────────────
    • Board is a flat list[Optional[str]] with indexes 0-8.
    • Agent action is defined by Height (Top/Center/Bottom) and Width (Left/Center/Right).
    • Agent side is random at every reset.
    • Opponent is purely random but legal.
    • Granular rewards for parsing action format correctly.
    ─────────────────────────
    """

    SUGGESTED_SYSPROMPT = (
        'You are the Tic-Tac-Toe player using "{mark}". '
        'Given the board, reply with your move by picking a height and width, '
        'in this format: <move>Top Center</move>'
    )

    # --- Mappings for Observation and Action Formats ---
    POSITION_NAMES: Tuple[str, ...] = (
        "Top Left", "Top Center", "Top Right",
        "Center Left", "Center Center", "Center Right",
        "Bottom Left", "Bottom Center", "Bottom Right"
    )
    
    MOVE_MAP: Dict[Tuple[Height, Width], int] = {
        (Height.TOP,    Width.LEFT):   0, (Height.TOP,    Width.CENTER): 1, (Height.TOP,    Width.RIGHT):  2,
        (Height.CENTER, Width.LEFT):   3, (Height.CENTER, Width.CENTER): 4, (Height.CENTER, Width.RIGHT):  5,
        (Height.BOTTOM, Width.LEFT):   6, (Height.BOTTOM, Width.CENTER): 7, (Height.BOTTOM, Width.RIGHT):  8,
    }
    
    INDEX_TO_NAME: Dict[int, str] = {i: name for i, name in enumerate(POSITION_NAMES)}

    WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    )

    def __init__(self) -> None:
        super().__init__()
        self.action_space = Action
        self.board: List[Optional[str]] = [None] * 9
        self.agent_mark: str = 'X'
        self.opponent_mark: str = 'O'
        self.done: bool = False

    def reset(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)
        self.board = [None] * 9
        self.done = False
        self.agent_mark, self.opponent_mark = random.choice([('X', 'O'), ('O', 'X')])
        self.system_prompt = self.SUGGESTED_SYSPROMPT.format(mark=self.agent_mark)
        if self.agent_mark == 'O':
            self._place(random.choice(self._empty_cells()), self.opponent_mark)
        return self._obs()

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Game has ended. Call reset().")

        reward = 0.0
        info = {"illegal_move": False}

        # 1. Reward for <move> tag
        try:
            move_content = extract_tag_value(action, "move")
            if move_content is None or not move_content.strip():
                raise ValueError("Tag is missing or empty.")
            reward += 0.1
        except Exception:
            obs = f"Your action was malformed. Please wrap your move in a <move> tag.\n{self._obs()}"
            info.update({"illegal_move": True, "error": "Malformed or missing <move> tag."})
            return obs, 0.0, False, info

        # 2. Split move and validate Height and Width
        parts = move_content.strip().split()
        if len(parts) != 2:
            obs = f"Your move '{move_content}' is malformed. Use format 'Height Width', e.g., 'Top Center'.\n{self._obs()}"
            info.update({"illegal_move": True, "error": "Move must have two parts: Height and Width."})
            return obs, reward, False, info

        # 3. Reward for valid 'height'
        height_str = parts[0]
        try:
            parsed_height = Height(height_str.capitalize())
            reward += 0.1
        except ValueError:
            obs = f"Invalid height '{height_str}'. Use one of {[h.value for h in Height]}.\n{self._obs()}"
            info.update({"illegal_move": True, "error": f"Invalid height value: {height_str}"})
            return obs, reward, False, info

        # 4. Reward for valid 'width'
        width_str = parts[1]
        try:
            parsed_width = Width(width_str.capitalize())
            reward += 0.1
        except ValueError:
            obs = f"Invalid width '{width_str}'. Use one of {[w.value for w in Width]}.\n{self._obs()}"
            info.update({"illegal_move": True, "error": f"Invalid width value: {width_str}"})
            return obs, reward, False, info

        # 5. Check move legality (cell must be empty)
        pos_idx = self.MOVE_MAP[(parsed_height, parsed_width)]
        if self.board[pos_idx] is not None:
            valid_moves = self._empty_cell_names()
            obs = (f"Your move to {self.INDEX_TO_NAME[pos_idx]} is illegal (cell is occupied).\n"
                   f"Valid moves are: {valid_moves}\n{self._obs()}")
            info.update({"illegal_move": True, "error": "Cell occupied"})
            return obs, reward, False, info

        # 6. Execute legal move and proceed with game
        self._place(pos_idx, self.agent_mark)
        terminal_rew = self._terminal_reward()
        if terminal_rew is not None:
            self.done = True
            return self._obs(), float(terminal_rew), True, {}

        if self._empty_cells():
            self._place(random.choice(self._empty_cells()), self.opponent_mark)
        
        terminal_rew = self._terminal_reward()
        if terminal_rew is not None:
            self.done = True
            return self._obs(), float(terminal_rew), True, {}

        return self._obs(), reward, False, {}

    def pretty_print(self) -> str:
        return "\n".join(
            f"{self.POSITION_NAMES[i]}: {self.board[i] or ' '}" for i in range(9)
        )

    def _obs(self) -> str:
        board_state = "\n".join(
            f"{self.POSITION_NAMES[i]}: {self.board[i] or '_'}" for i in range(9)
        )
        prompt = (
            "\n\nWhat will be your next move? "
            "Pick a height (Top, Center, Bottom) and a width (Left, Center, Right)."
        )
        return board_state + prompt

    def _place(self, pos: int, mark: str) -> None:
        self.board[pos] = mark

    def _empty_cells(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v is None]

    def _empty_cell_names(self) -> List[str]:
        return sorted([self.INDEX_TO_NAME[i] for i in self._empty_cells()])

    def _terminal_reward(self) -> Optional[int]:
        for a, b, c in self.WIN_LINES:
            if self.board[a] and self.board[a] == self.board[b] == self.board[c]:
                return 1 if self.board[a] == self.agent_mark else -1
        return 0 if all(cell is not None for cell in self.board) else None

if __name__ == "__main__":
    env = TicTacToe()
    obs = env.reset(seed=43)

    print("You're playing as:", env.agent_mark)
    print(env.pretty_print())
    print("______________________\n")

    while not env.done:
        try:
            h_in = input(f"Enter height ({'/'.join([h.value for h in Height])}): ")
            w_in = input(f"Enter width ({'/'.join([w.value for w in Width])}): ")
            action_str = f"<move>{h_in} {w_in}</move>"
            
            obs, reward, done, info = env.step(action_str)

            print("\n" + env.pretty_print())
            print(f"Reward: {reward:.1f}")
            print("______________________\n")

            if info.get("illegal_move"):
                print(f"INFO: {info.get('error')}")
                if "Valid moves" in obs:
                    print(obs.splitlines()[1]) # Print the line with valid moves

        except (ValueError, TypeError) as e:
            print(f"Invalid input: {e}")
            continue

        if env.done:
            if reward == 1:
                print("You win! 🎉")
            elif reward == -1:
                print("You lose. 😔")
            else:
                print("It's a draw. 🤝")