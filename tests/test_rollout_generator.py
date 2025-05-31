from __future__ import annotations

import os
import random
import re
import multiprocessing as mp
from typing import Dict, Tuple, Optional, Any

import pytest

# -----------------------------------------------------------------------------
# vLLM multiprocessing boilerplate (GPU integration test) ----------------------
# -----------------------------------------------------------------------------
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
mp.set_start_method("spawn", force=True)

# -----------------------------------------------------------------------------
# Under‑test plumbing ----------------------------------------------------------
# -----------------------------------------------------------------------------
from ludic_envs.envs.env import Env
from ludic_envs.inference.rollout_generator import RolloutGenerator
from ludic_envs.parsers import extract_tag_value

# -----------------------------------------------------------------------------
# Environment: three‑attempt digit guesser -------------------------------------
# -----------------------------------------------------------------------------
from ludic_envs.envs.env import Env # Assuming this is the correct path to your Env ABC
from ludic_envs.parsers import extract_tag_value # Assuming this is available

class NumberGuessEnv(Env):
    """
    Guess a hidden single digit in up to `max_attempts` tries.
    The agent should reply with <move>N</move>, where N is its guess.
    The observation explicitly reveals the hidden digit to test instruction following.
    """

    max_attempts: int = 3
    
    # system_prompt is now set in __init__ to conform to the Env ABC,
    # though it could remain a class variable if preferred and super().__init__() is called.
    # For this game, the system prompt is static.

    def __init__(self) -> None:
        super().__init__() # Call the parent class's __init__
        self.system_prompt: str = (
            f"You are playing a guessing game. I have a hidden digit. "
            f"You have up to {self.max_attempts} attempts to guess it. "
            "The hidden digit will be shown in my message. "
            "Reply **only** with your guess in the format <move>N</move> where N is the digit."
        )
        self._number: Optional[int] = None
        self._attempt: int = 0

    def reset(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)
        self._number = random.randint(0, 9)
        self._attempt = 0
        # The system_prompt is already set in __init__
        return self._obs()

    def parse_action(self, action_str: str) -> int:
        """
        Parses the LLM's string output (e.g., "<move>7</move>") into an integer guess.
        Raises ValueError if parsing fails or the guess is invalid.
        """
        try:
            guess_str = extract_tag_value(action_str, "move")
            if guess_str is None:
                raise ValueError("The <move> tag was not found in your response.")
            
            guess = int(guess_str)
            if not (0 <= guess <= 9):
                raise ValueError("Your guess must be a single digit between 0 and 9.")
            return guess
        except ValueError as e: # Catches int conversion errors or specific ValueErrors raised
            raise ValueError(f"Invalid move. {e}")
        except Exception as e: # Catch other potential errors from extract_tag_value
            # This helps debug unexpected issues with the parser or action string
            raise ValueError(f"Error parsing action string '{action_str}': {e}")

    def step(self, action_str: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Processes the agent's action string, updates the game state, and returns the outcome.
        """
        self._attempt += 1
        
        try:
            guess = self.parse_action(action_str)
        except ValueError as e:
            # If parsing fails, penalize slightly or give specific feedback.
            # An attempt is still counted.
            error_message = f"⚠️ Your action was invalid: {e}."
            if self._attempt >= self.max_attempts:
                obs = f"{error_message} You have no more attempts. The number was {self._number}."
                return obs, 0.0, True, {"error": str(e), "final_state": "invalid_action_loss"}
            else:
                obs = f"{error_message} Try again. {self._obs()}"
                return obs, 0.0, False, {"error": str(e)}

        # Proceed with a valid guess
        correct = (guess == self._number)
        done = False
        info: Dict[str, Any] = {}

        if correct:
            reward = 1.0
            done = True
            obs = f"⭐ Correct! The number was {self._number}. Well done!"
            info["final_state"] = "win"
        elif self._attempt >= self.max_attempts:
            reward = 0.0
            done = True
            obs = f"❌ Wrong. That was your last attempt. The number was {self._number}."
            info["final_state"] = "loss_attempts"
        else: # Wrong, but still have attempts
            reward = 0.0
            done = False
            obs = f"❌ Wrong guess. Try again."
        
        # If the game is not done, append the standard observation prompt
        if not done:
            obs = f"{obs} {self._obs()}"
            
        return obs, reward, done, info

    def _obs(self) -> str:
        """
        Generates the observation string for the agent.
        This version explicitly tells the agent the hidden number, making it a test
        of instruction following and extraction rather than pure guessing.
        """
        # Ensure _number is not None; it should be set by reset()
        if self._number is None:
            # This case should ideally not be reached if reset() is always called first.
            # Handle it defensively, perhaps by calling reset or raising an error.
            # For now, we'll make a placeholder, but this indicates a state issue.
            return "Error: Game not initialized. Call reset()." 
            
        return (
            f"I'm thinking of a digit. Hidden digit is: {self._number}. "
            f"Attempt {self._attempt + 1}/{self.max_attempts}. What is your move?"
        )


# -----------------------------------------------------------------------------
# Deterministic stub sampler (unit‑tests) --------------------------------------
# -----------------------------------------------------------------------------

def _echo_number_sampler(model, prompts: List[List[Dict]], sampling_params):  # noqa: D401
    """Return the digit embedded in the latest observation as <move>n</move>."""

    def _latest_obs(chat: List[Dict]) -> str:  # noqa: D401
        return chat[-1]["content"]

    replies: List[str] = []
    for chat in prompts:
        m = re.search(r"(\d)", _latest_obs(chat))
        digit = m.group(1) if m else "?"
        replies.append(f"<move>{digit}</move>")
    return replies, [None] * len(replies)


# -----------------------------------------------------------------------------
# Autouse monkey‑patch: route to echo‑sampler in CPU tests ---------------------
# -----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_sampler(monkeypatch, request):  # noqa: D401
    """Patch both sample references unless the test is GPU‑enabled."""

    if request.node.get_closest_marker("requires_gpu"):
        # GPU tests want the real `sample`.
        yield
        return

    monkeypatch.setattr(
        "ludic_envs.inference.sample.sample",
        _echo_number_sampler,
        raising=True,
    )
    monkeypatch.setattr(
        "ludic_envs.inference.rollout_generator.sample",
        _echo_number_sampler,
        raising=True,
    )
    yield


# -----------------------------------------------------------------------------
# Unit tests (CPU‑only) --------------------------------------------------------
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("batch,groups", [(3, 1), (2, 2)])
def test_collect_basic(batch: int, groups: int):  # noqa: D401
    """Smoke‑test: echo‑sampler should solve on first or later try."""

    rg = RolloutGenerator(NumberGuessEnv, max_steps=3, group_size=groups)
    trajs = rg.collect(batch, model=None, sampling_params={})

    assert len(trajs) == batch * groups
    for t in trajs:
        assert 1 <= len(t["steps"]) <= 3
        last = t["steps"][-1]
        assert last["reward"] in (0.0, 1.0)
        assert extract_tag_value(last["assistant"], "move").isdigit()


def test_remember_history(monkeypatch):  # noqa: D401
    """Validate `remember_history=True` actually threads conversation."""

    class FixedNumberEnv(NumberGuessEnv):
        """Env with digit fixed to 5 so sampler is always wrong."""

        def reset(self, seed: int | None = None) -> str:  # noqa: D401
            self._number = 5
            self._attempt = 0
            return self._obs()

    def _always_wrong_sampler(model, prompts, params):  # noqa: D401
        return ["<move>0</move>"] * len(prompts), [None] * len(prompts)

    monkeypatch.setattr(
        "ludic_envs.inference.sample.sample",
        _always_wrong_sampler,
        raising=True,
    )
    monkeypatch.setattr(
        "ludic_envs.inference.rollout_generator.sample",
        _always_wrong_sampler,
        raising=True,
    )

    rg = RolloutGenerator(FixedNumberEnv, max_steps=3, remember_history=True)
    trajs = rg.collect(batch_size=1, model=None, sampling_params={})

    steps = trajs[0]["steps"]
    assert len(steps) == 3  # exhausted all attempts

    expected_lengths = [2, 4, 6]  # system+user, then +2 history etc.
    for step, exp_len in zip(steps, expected_lengths):
        assert len(step["prompt"]) == exp_len

    assert steps[1]["prompt"][0]["role"] != steps[1]["prompt"][1]["role"]


# -----------------------------------------------------------------------------
# GPU integration tests (real vLLM + Qwen‑2 0.5B) ------------------------------
# -----------------------------------------------------------------------------
CUDA_AVAILABLE = False
try:
    import torch  # noqa: WPS433

    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    pass

RUN_REAL_TESTS = os.environ.get("RUN_REAL_VLLM_TESTS") == "1"

requires_gpu = pytest.mark.skipif(
    not (CUDA_AVAILABLE and RUN_REAL_TESTS),
    reason="CUDA not available or RUN_REAL_VLLM_TESTS not set",
)

@requires_gpu  # type: ignore[misc]
def test_collect_with_qwen():  # noqa: D401
    """Exercise RolloutGenerator.collect (≤3 steps) with Qwen‑2 on GPU."""

    from vllm import LLM, SamplingParams  # noqa: WPS433, WPS440

    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.40,
    )

    rg = RolloutGenerator(NumberGuessEnv, max_steps=3)
    sampling_params = SamplingParams(max_tokens=8, temperature=0)

    trajs = rg.collect(batch_size=3, model=llm, sampling_params=sampling_params)

    assert len(trajs) == 3
    for tr in trajs:
        steps = tr["steps"]
        assert 1 <= len(steps) <= 3
        assert steps[-1]["reward"] in (0.0, 1.0)
