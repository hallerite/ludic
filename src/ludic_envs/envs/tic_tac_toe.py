from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional
from ludic_envs.envs.env import Env
from ludic_envs.parsers import extract_tag_value
from pydantic import BaseModel, Field

class Action(BaseModel):
    """Validated player move.

    Attributes
    ----------
    pos : int
        Board position 1-9, arranged left-to-right, top-to-bottom:

            1 | 2 | 3
            ---------
            4 | 5 | 6
            ---------
            7 | 8 | 9
    """
    pos: int = Field(..., ge=1, le=9, description="Board position 1-9")

class TicTacToe(Env):
    """
    ─────────────────────────
    • Board is a flat list[Optional[str]] with indexes 0-8.
    • Marks are string literals: 'X', 'O', None.
    • Agent side is random at every reset.
    • Opponent is purely random but legal.
    ─────────────────────────
    TBD:
        - allow for choosing opponent playstyle
        - add flag for intermediate rewards
        - allow for self-play
    """

    SUGGESTED_SYSPROMPT = (
        'You are the Tic-Tac-Toe player using "{mark}". '
        'Given the board, reply with your move in this format: <move>5</move>'
    )

    # Winning triples, expressed once and reused
    WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),           # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),           # cols
        (0, 4, 8), (2, 4, 6)                       # diags
    )

    def __init__(self) -> None:
        super().__init__()
        self.action_space = Action

        self.board: List[Optional[str]] = [None] * 9

        self.agent_mark: str = 'X'                 # set in reset()
        self.opponent_mark: str = 'O'

        self.done: bool = False

    def parse_action(self, action_str: str) -> Action:
        try:
            pos = int(extract_tag_value(action_str, "move"))
        except (ValueError, TypeError):
            raise ValueError("Invalid move: <move> tag must contain an integer between 1 and 9.")

        if not (1 <= pos <= 9):
            raise ValueError(f"Invalid move: position {pos} is out of bounds (1-9).")

        pos_idx = pos - 1
        if self.board[pos_idx] is not None:
            raise ValueError(f"Invalid move: position {pos} is already occupied.")

        return Action(pos=pos)

        

    # ──────────────────────────────────────────────────────────────
    # RL interface
    # ──────────────────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        self.board = [None] * 9
        self.done = False

        # Coin-flip for sides
        self.agent_mark, self.opponent_mark = random.choice([('X', 'O'),
                                                             ('O', 'X')])
        
        self.system_prompt = self.SUGGESTED_SYSPROMPT.format(mark=self.agent_mark)

        # If agent is O, let opponent (as X) open with a random legal move
        if self.agent_mark == 'O':
            self._place(random.choice(self._empty_cells()), self.opponent_mark)

        return self._obs()

    def step(self, action: str) -> Tuple[str, int, bool, Dict]:
        if self.done:
            raise RuntimeError("Game has ended. Call reset().")
        
        try:
            act = self.parse_action(action)
        except Exception as e:
            # Don't end game, just return current obs + info
            obs = "\nYour last move was illegal. Try again." + self._obs()
            return obs, 0, False, {"illegal_move": True, "error": str(e)}

        pos_idx: int = act.pos - 1

        self._place(pos_idx, self.agent_mark)

        # Check win/draw for agent move
        reward = self._terminal_reward()
        if reward is not None:
            self.done = True
            return self._obs(), reward, True, {}

        # Opponent's random reply
        self._place(random.choice(self._empty_cells()), self.opponent_mark)

        reward = self._terminal_reward()
        if reward is not None:
            self.done = True
            return self._obs(), reward, True, {}

        return self._obs(), 0, False, {}

    # ──────────────────────────────────────────────────────────────
    # Convenience for humans
    # ──────────────────────────────────────────────────────────────
    def pretty_print(self) -> str:
        """Return an ASCII board with grid lines."""
        symbols = [m if m is not None else str(i+1) for i, m in enumerate(self.board)]
        template = (
            " {0} | {1} | {2} \n"
            "-----------\n"
            " {3} | {4} | {5} \n"
            "-----------\n"
            " {6} | {7} | {8} "
        )
        return template.format(*symbols)

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────
    def _obs(self) -> str:
        """Three-line minimal string for an LLM: e.g. 'XO.\n..O\nX..'"""
        to_char = lambda m: m if m is not None else '.'
        rows = ["".join(to_char(self.board[i + j]) for j in range(3))
                for i in (0, 3, 6)]
        return "\n".join(rows) + "\nWhat will be your next move?"

    def _place(self, pos: int, mark: str) -> None:
        if not (0 <= pos < 9):
            raise ValueError("pos must be 0-8")
        if self.board[pos] is not None:
            raise ValueError("Illegal move: square occupied")
        self.board[pos] = mark

    def _empty_cells(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v is None]

    def _terminal_reward(self) -> Optional[int]:
        """Return +1 (agent win), -1 (agent loss), 0 (draw) or None (ongoing)."""
        # Someone wins?
        import random
        return random.randint(0, 10) 
        for a, b, c in self.WIN_LINES:
            line = {self.board[a], self.board[b], self.board[c]}
            if line == {self.agent_mark}:
                return 1
            if line == {self.opponent_mark}:
                return -1
        # Draw?
        if all(cell is not None for cell in self.board):
            return 0
        return None

if __name__ == "__main__":
    env = TicTacToe()
    obs = env.reset(seed=42)

    print("You're playing as:", env.agent_mark)
    print(env.pretty_print())
    print("___________")

    
    while not env.done:
        try:
            move = int(input("Enter your move (1-9): "))
            obs, reward, done, _ = env.step({'pos': move})
        except Exception as e:
            print("Invalid move:", e)
            continue

        print(env.pretty_print())
        print("___________")

        if env.done:
            if reward == 1:
                print("You win!")
            elif reward == -1:
                print("You lose.")
            else:
                print("It's a draw.")