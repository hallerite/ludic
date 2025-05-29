from __future__ import annotations
from typing import Type, List, Dict

import random
import logging

from ludic_envs.envs.env import Env
from ludic_envs.inference.sample import sample
from ludic_envs.parsers import extract_tag_value   # noqa: F401  (import kept for callers)

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class RolloutGenerator:
    """Collect trajectories as *prompt/assistant/reward* triples (dialog-centric).

    The generator makes **no assumptions** about tokenisation or model internals;
    it only records the *exact* prompt forwarded to the LLM, the assistant's
    reply, and the reward returned by the environment.
    """

    def __init__(
        self,
        env_cls: Type[Env],
        max_steps: int,
        *,
        remember_history: bool = False,
        group_size: int = 1,
        seed: int | None = None,
    ) -> None:
        self.env_cls = env_cls
        self.max_steps = max_steps
        self.remember_history = remember_history
        self.group_size = group_size          # FIXME: assumes GRPO!
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def collect(self, batch_size: int, model, sampling_params):
        """Roll out *batch_size × group_size* episodes and return dialogs.

        Each element in the returned list corresponds to **one environment** and
        has the structure::

            {
                "group": int,        # GRPO clone‑set ID
                "steps": [
                    {
                        "prompt": List[Dict[str, str]],   # chat format fed to LLM
                        "assistant": str | Dict,          # model reply as given
                        "reward": float,                  # env‑returned reward
                        "raw": Any,                       # model metadata (optional)
                    },
                    ...
                ]
            }
        """

        # 1) Spawn envs in seed‑synchronised groups
        envs: List[Env] = []
        obs_text: List[str] = []
        group_ids: List[int] = []

        for g in range(batch_size):
            seed = self.rng.randint(0, 2**32 - 1)
            logger.debug("Seed for group %d: %s", g, seed)
            for _ in range(self.group_size):
                env = self.env_cls()
                envs.append(env)
                obs_text.append(env.reset(seed=seed))
                group_ids.append(g)

        n_envs = len(envs)
        done = [False] * n_envs
        histories: List[List[Dict[str, str]]] = [[] for _ in envs]

        trajectories = [{"group": gid, "steps": []} for gid in group_ids]

        # 2) Roll-out loop ──────────────────────────────────────────────
        for step_idx in range(self.max_steps):
            # ── Inject system prompt once per env (no mutation in helper) ──
            if self.remember_history and step_idx == 0:
                for i, env in enumerate(envs):
                    histories[i].append(
                        {"role": "system", "content": env.system_prompt}
                    )

            # ── Build prompts & sample the model ────────────────────────
            prompts = [
                self._build_prompt(envs[i], obs_text[i], histories[i])
                for i in range(n_envs)
            ]
            replies_txt, replies_raw = sample(model, prompts, sampling_params)

            # ── Step each env with the assistant reply ──────────────────
            for i, env in enumerate(envs):
                if done[i]:
                    continue

                assistant_reply = replies_txt[i]
                next_obs, reward, done[i], _ = env.step(assistant_reply)

                if self.remember_history:
                    histories[i].extend([
                        {"role": "user", "content": obs_text[i]},
                        {"role": "assistant", "content": assistant_reply},
                    ])

                trajectories[i]["steps"].append({
                    "prompt": prompts[i],
                    "assistant": assistant_reply,
                    "reward": reward,
                    "raw": replies_raw[i],
                })

                obs_text[i] = next_obs

            if all(done):
                break

        return trajectories

    # ------------------------------------------------------------------
    # Prompt helper
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        env: Env,
        obs: str,
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Return a fresh prompt list without mutating *history*."""
        if self.remember_history:
            base = history
        else:
            base = [{"role": "system", "content": env.system_prompt}]

        # `+` creates a shallow copy so the caller can’t accidentally
        # mutate the original list.
        return base + [{"role": "user", "content": obs}]
