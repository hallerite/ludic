from __future__ import annotations

from typing import Dict, Tuple


def split_parser_reward(
    *,
    parser_reward: float,
    action_target: str,
    parse_error: bool,
) -> Tuple[float, Dict[str, float], float, Dict[str, float]]:
    """
    Split parser reward between agent and env steps.

    - For env-targeted, successfully parsed actions:
      move the parser reward onto the env step to avoid double counting.
    - For parse errors or internal steps:
      keep the parser reward on the agent step only.
    """
    agent_reward = float(parser_reward)
    agent_components: Dict[str, float] = {"parser": float(parser_reward)}
    env_parser_reward = 0.0
    env_components: Dict[str, float] = {}

    if action_target == "env" and not parse_error:
        agent_reward = 0.0
        agent_components = {}
        env_parser_reward = float(parser_reward)
        env_components = {"parser": float(parser_reward)}

    return agent_reward, agent_components, env_parser_reward, env_components
