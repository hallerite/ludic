import os
import random
import re
import multiprocessing as mp
from typing import Dict, List, Tuple

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
class NumberGuessEnv(Env):
    """Guess a hidden single digit in ≤ 3 tries using <move>N</move>."""

    max_attempts: int = 3
    system_prompt: str = (
        "You are playing a guessing game. You have up to 3 attempts. "
        "Reply **only** with <move>N</move> where N is the hidden digit shown "
        "in the user's last message."
    )

    def __init__(self) -> None:  # noqa: D401
        self._number: int | None = None
        self._attempt: int = 0

    # Gym‑style API -----------------------------------------------------------
    def reset(self, seed: int | None = None) -> str:  # noqa: D401
        if seed is not None:
            random.seed(seed)
        self._number = random.randint(0, 9)
        self._attempt = 0
        return self._obs()

    def step(self, action: Dict) -> Tuple[str, float, bool, Dict]:  # noqa: D401
        self._attempt += 1
        guess = action.get("pos")
        correct = (guess == self._number)

        if correct:
            reward, done, obs = 1.0, True, "⭐ Correct — well done!"
        elif self._attempt >= self.max_attempts:
            reward, done, obs = 0.0, True, "❌ Wrong. Out of attempts."
        else:
            reward, done, obs = 0.0, False, "❌ Wrong. Try again."

        if not done:
            obs = f"{obs} {self._obs()}"
        return obs, reward, done, {}

    # Helper ------------------------------------------------------------------
    def _obs(self) -> str:  # noqa: D401
        return (
            f"Hidden digit: {self._number}. Attempt {self._attempt+1}/"
            f"{self.max_attempts}. What is your move?"
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

    expected_lengths = [2, 3, 5]  # system+user, then +2 history etc.
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
