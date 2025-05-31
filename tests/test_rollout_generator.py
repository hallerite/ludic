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
from ludic_envs.inference.rollout_generator import RolloutGenerator, HistoryManagement
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

def _echo_number_sampler_for_tests(model, prompts: List[List[Dict]], sampling_params, current_strategy: HistoryManagement):
    """
    Mock sampler modified for testing.
    If SCRATCHPAD, produces <scratchpad> and <action> tags.
    Otherwise, produces only the content expected by NumberGuessEnv (<move>N</move>).
    """
    def _latest_obs(chat: List[Dict]) -> str:
        return chat[-1]["content"]

    replies: List[str] = []
    for chat in prompts:
        m = re.search(r"Hidden digit is: (\d)", _latest_obs(chat)) # More specific regex
        digit = m.group(1) if m else "?"
        
        action_content = f"<move>{digit}</move>"

        if current_strategy == HistoryManagement.SCRATCHPAD:
            # For SCRATCHPAD, the RolloutGenerator expects both tags.
            # The inner <action> tag will contain what NumberGuessEnv.parse_action expects.
            replies.append(f"<scratchpad>Mock scratchpad content for digit {digit}.</scratchpad><action>{action_content}</action>")
        else:
            # For NO_HISTORY or FULL_HISTORY, RolloutGenerator passes the full reply to env.parse_action.
            # So, the reply should be directly what NumberGuessEnv.parse_action expects.
            replies.append(action_content)
            
    return replies, [None] * len(replies)


# Store the original sample function to restore it
_original_sample_func = None

@pytest.fixture(autouse=True)
def _patch_sampler(monkeypatch, request):
    """
    Patches the 'sample' function. The mock sampler's behavior now depends
    on the history_strategy of the RolloutGenerator instance being tested.
    This is tricky because the fixture doesn't directly know the rg instance.
    We'll patch it globally and make the mock adaptable or specific per test if needed.
    For now, we'll make it adaptable if we can access the strategy.
    However, a simpler way for this fixture is to assume a default mock behavior
    and let specific tests re-patch if they need different mock behavior.
    
    Let's make the default mock work for NO_HISTORY and FULL_HISTORY.
    Tests for SCRATCHPAD will need a specialized mock or re-patch.
    """
    global _original_sample_func
    if _original_sample_func is None: # Ensure we only store it once
        import ludic_envs.inference.sample as sample_module
        _original_sample_func = sample_module.sample

    if request.node.get_closest_marker("requires_gpu"):
        yield
        return

    # This default mock is for NO_HISTORY and FULL_HISTORY
    # The NumberGuessEnv.parse_action expects "<move>N</move>" directly.
    # The RolloutGenerator, for these strategies, passes the full assistant_reply
    # to env.parse_action.
    def mock_sampler_default(model, prompts: List[List[Dict]], sampling_params):
        def _latest_obs(chat: List[Dict]) -> str:
            return chat[-1]["content"]
        replies_list: List[str] = []
        for chat_prompt in prompts:
            m = re.search(r"Hidden digit is: (\d)", _latest_obs(chat_prompt))
            digit = m.group(1) if m else "?"
            replies_list.append(f"<move>{digit}</move>")
        return replies_list, [None] * len(replies_list)

    monkeypatch.setattr("ludic_envs.inference.sample.sample", mock_sampler_default)
    monkeypatch.setattr("ludic_envs.inference.rollout_generator.sample", mock_sampler_default)
    
    yield
    
    # Restore original sample function after test run
    if _original_sample_func is not None:
        monkeypatch.setattr("ludic_envs.inference.sample.sample", _original_sample_func)
        monkeypatch.setattr("ludic_envs.inference.rollout_generator.sample", _original_sample_func)


# -----------------------------------------------------------------------------
# Unit tests (CPU‑only) --------------------------------------------------------
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("batch,groups", [(3, 1), (2, 2)])
def test_collect_basic_no_history(batch: int, groups: int):
    """Test with NO_HISTORY (default)."""
    # Uses default history_strategy = HistoryManagement.NO_HISTORY
    rg = RolloutGenerator(NumberGuessEnv, max_steps=3, group_size=groups)
    trajs = rg.collect(batch, model=None, sampling_params={})

    assert len(trajs) == batch * groups
    for t in trajs:
        assert 1 <= len(t["steps"]) <= 3
        last_step = t["steps"][-1]
        # For NO_HISTORY/FULL_HISTORY, assistant_reply is "<move>N</move>"
        # and NumberGuessEnv.parse_action directly parses that.
        assert extract_tag_value(last_step["assistant"], "move").isdigit() 
        assert last_step["reward"] in (0.0, 1.0)


def test_full_history(monkeypatch):
    """Validate FULL_HISTORY strategy."""

    class FixedNumberEnv(NumberGuessEnv):
        def reset(self, seed: int | None = None) -> str:
            self._number = 5 # Always 5
            self._attempt = 0
            return self._obs()

    # This specialized mock is for _always_wrong_sampler with FULL_HISTORY
    def _always_wrong_sampler_for_full_history(model, prompts, params):
        # NumberGuessEnv.parse_action needs "<move>N</move>"
        return ["<move>0</move>"] * len(prompts), [None] * len(prompts)
    
    monkeypatch.setattr("ludic_envs.inference.sample.sample", _always_wrong_sampler_for_full_history)
    monkeypatch.setattr("ludic_envs.inference.rollout_generator.sample", _always_wrong_sampler_for_full_history)

    rg = RolloutGenerator(FixedNumberEnv, max_steps=3, history_strategy=HistoryManagement.FULL_HISTORY)
    trajs = rg.collect(batch_size=1, model=None, sampling_params={})

    steps = trajs[0]["steps"]
    assert len(steps) == 3

    # System prompt + user obs, then +2 (user, assistant) for each subsequent turn
    # First prompt: System + User_Obs (length 2)
    # Second prompt: System + User_Obs + Assistant_Reply_1 + User_Obs_2 (length 4)
    # Third prompt: System + User_Obs + Assistant_Reply_1 + User_Obs_2 + Assistant_Reply_2 + User_Obs_3 (length 6)
    expected_lengths = [2, 4, 6]
    for step, exp_len in zip(steps, expected_lengths):
        assert len(step["prompt"]) == exp_len
        # Check that system prompt is the first message
        assert step["prompt"][0]["role"] == "system"
        if exp_len > 2: # For prompts with history
             # Check alternating roles after system prompt
            roles = [msg["role"] for msg in step["prompt"][1:]]
            for i in range(len(roles) - 1):
                 assert roles[i] != roles[i+1]


def test_scratchpad_history(monkeypatch):
    """Validate SCRATCHPAD strategy."""

    class ScratchpadTestEnv(NumberGuessEnv): # Can reuse NumberGuess for simplicity
        def __init__(self):
            super().__init__()
            # System prompt for NumberGuessEnv is fine, RolloutGenerator appends SCRATCHPAD instructions
    
    # Mock sampler specifically for SCRATCHPAD mode
    def _scratchpad_aware_sampler(model, prompts: List[List[Dict]], sampling_params):
        replies_list: List[str] = []
        for chat_prompt in prompts:
            def _latest_obs(chat: List[Dict]) -> str: # Helper to get current observation
                for msg in reversed(chat):
                    if msg["role"] == "user":
                        return msg["content"]
                return "" # Should not happen

            current_obs = _latest_obs(chat_prompt)
            m = re.search(r"Hidden digit is: (\d)", current_obs)
            digit = m.group(1) if m else "0" # Default to 0 if not found

            # Reply format for SCRATCHPAD mode
            scratchpad_content = f"My guess for {digit} based on obs."
            action_content = f"<move>{digit}</move>" 
            replies_list.append(f"<scratchpad>{scratchpad_content}</scratchpad><action>{action_content}</action>")
        return replies_list, [None] * len(replies_list)

    monkeypatch.setattr("ludic_envs.inference.sample.sample", _scratchpad_aware_sampler)
    monkeypatch.setattr("ludic_envs.inference.rollout_generator.sample", _scratchpad_aware_sampler)

    rg = RolloutGenerator(ScratchpadTestEnv, max_steps=1, history_strategy=HistoryManagement.SCRATCHPAD)
    trajs = rg.collect(batch_size=1, model=None, sampling_params={})
    
    assert len(trajs) == 1
    assert len(trajs[0]["steps"]) == 1
    step1 = trajs[0]["steps"][0]

    # Check prompt structure for scratchpad
    prompt1 = step1["prompt"]
    assert prompt1[0]["role"] == "system"
    assert "## Your Current Memory/Scratchpad:" not in prompt1[0]["content"] # No scratchpad on first turn
    assert "<scratchpad>" in prompt1[0]["content"] # Instructions should be there
    assert "<action>" in prompt1[0]["content"]
    assert prompt1[1]["role"] == "user"

    # Check assistant reply parsing
    assert "My guess for" in step1["assistant"]
    
    # Test with a second step to see if scratchpad is used
    rg_multistep = RolloutGenerator(ScratchpadTestEnv, max_steps=2, history_strategy=HistoryManagement.SCRATCHPAD)
    trajs_multistep = rg_multistep.collect(batch_size=1, model=None, sampling_params={})
    
    assert len(trajs_multistep[0]["steps"]) == 2 # Should have 2 steps
    step2_prompt = trajs_multistep[0]["steps"][1]["prompt"]
    assert step2_prompt[0]["role"] == "system"
    assert "## Your Current Memory/Scratchpad:\nMy guess for" in step2_prompt[0]["content"]


# -----------------------------------------------------------------------------
# GPU integration tests (real vLLM + Qwen‑2 0.5B) ------------------------------
# -----------------------------------------------------------------------------
CUDA_AVAILABLE = False
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    pass

RUN_REAL_TESTS = os.environ.get("RUN_REAL_VLLM_TESTS") == "1"

requires_gpu = pytest.mark.skipif(
    not (CUDA_AVAILABLE and RUN_REAL_TESTS),
    reason="CUDA not available or RUN_REAL_VLLM_TESTS not set",
)

@requires_gpu
def test_collect_with_qwen_no_history():
    """Exercise RolloutGenerator with Qwen‑2 on GPU (NO_HISTORY)."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen2-0.5B-Instruct",
        dtype="float16",
        enforce_eager=True, # Easier for small tests
        gpu_memory_utilization=0.50, # Increased slightly
    )
    # Default is NO_HISTORY
    rg = RolloutGenerator(NumberGuessEnv, max_steps=3, history_strategy=HistoryManagement.NO_HISTORY)
    # Temperature 0 for deterministic output from LLM, max_tokens increased for safety
    sampling_params = SamplingParams(max_tokens=20, temperature=0) 

    trajs = rg.collect(batch_size=1, model=llm, sampling_params=sampling_params)

    assert len(trajs) == 1
    tr = trajs[0]
    steps = tr["steps"]
    assert 1 <= len(steps) <= 3
    
    # NumberGuessEnv.parse_action expects <move>N</move>
    # For NO_HISTORY, RolloutGenerator passes full assistant_reply to parse_action
    # LLM for NumberGuessEnv should ideally output *only* <move>N</move>
    # If it outputs more, parse_action in NumberGuessEnv might fail or misinterpret
    # This tests if the LLM follows the simple instruction format
    parsed_action_content = extract_tag_value(steps[-1]["assistant"], "move")
    assert parsed_action_content.isdigit()
    assert steps[-1]["reward"] in (0.0, 1.0)

@requires_gpu
def test_collect_with_qwen_scratchpad():
    """Exercise RolloutGenerator with Qwen-2.5-7B-Instruct on GPU (SCRATCHPAD)."""
    from vllm import LLM, SamplingParams
    from ludic_envs.envs.pomdp.key_door import KeyDoorEnv

    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.50,
    )
    
    # KeyDoorEnv's system prompt doesn't include scratchpad/action format instructions.
    # RolloutGenerator._build_prompt for SCRATCHPAD mode adds these instructions.
    rg = RolloutGenerator(KeyDoorEnv, max_steps=5, history_strategy=HistoryManagement.SCRATCHPAD, seed=42)
    sampling_params = SamplingParams(max_tokens=60, temperature=0.1, top_p=0.9) # Allow more diverse output

    trajs = rg.collect(batch_size=1, model=llm, sampling_params=sampling_params)

    assert len(trajs) == 1
    tr = trajs[0]
    steps = tr["steps"]
    assert len(steps) > 0 # Should take at least one step

    # For SCRATCHPAD, assistant_reply should contain <scratchpad> and <action>
    # The RolloutGenerator extracts content of <action> and passes to KeyDoorEnv.parse_action
    # KeyDoorEnv.parse_action expects the inner content of <action> to be one of its VALID_ACTIONS
    
    first_step_reply = steps[0]["assistant"]
    assert "<scratchpad>" in first_step_reply
    assert "</scratchpad>" in first_step_reply
    assert "<action>" in first_step_reply
    assert "</action>" in first_step_reply
    
    action_tag_content = extract_tag_value(first_step_reply, "action")
    assert action_tag_content.strip() in KeyDoorEnv.VALID_ACTIONS