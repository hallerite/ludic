from __future__ import annotations
import asyncio
from typing import Optional, Dict, List, Set

from ludic.envs.env import LudicEnv
from ludic.agents.base_agent import Agent
from ludic.types import Rollout, Step, StepOutcome
from ludic.inference.request import InferenceSpec
from .base import InteractionProtocol
from .step_collector import TraceCollector
from .info import merge_step_info


class MultiAgentProtocol(InteractionProtocol):
    """
    Handles an interaction with multiple agents.

    This protocol queries the environment for which agents are
    'active' at each step and only gathers actions from them.

    Agents continue independently: when one agent terminates, others
    keep playing until they also terminate or max_steps is hit.

    It returns a list of Rollouts (one per managed agent) suitable
    for training.

    Parser failures:
      If a managed agent's parser returns ParseResult.action=None, that
      agent is omitted from env.step(actions). A synthetic Step is still
      logged for the failing agent (reward=parse_result.reward, info
      includes parse_error=True). The failing agent's context is updated
      with the synthetic observation for the next turn.
    """

    def __init__(self, agents: Dict[str, Agent]):
        """
        Initializes the protocol with a mapping of Agent IDs
        from the environment (e.g., "player_1") to the
        actual Agent objects that will fill those roles.
        """
        self.agent_map = agents

    async def run(
        self,
        *,
        env: LudicEnv,
        max_steps: int,
        env_seed: Optional[int] = None,
        sampling_seed: Optional[int] = None,
        inference: Optional[InferenceSpec] = None,
        timeout_s: Optional[float] = None,
    ) -> List[Rollout]:
        inf = inference or InferenceSpec()

        # Initialize the collector to track separate histories per agent
        collector = TraceCollector(
            protocol=self.__class__.__name__,
            env_name=env.__class__.__name__,
            agents=list(self.agent_map.keys()),
        )

        # Track which managed agents have finished (terminated or truncated by env)
        finished_agents: Set[str] = set()

        # 1. --- Reset Env and all managed Agents ---
        obs_info_dict = env.reset(seed=env_seed)
        sys_prompt = getattr(env, "suggested_sysprompt", None)

        for agent_id, agent in self.agent_map.items():
            obs, info = obs_info_dict.get(agent_id, (None, {}))
            if obs is not None:
                agent.reset(system_prompt=sys_prompt)
                agent.on_env_reset(obs, info)

        # Track current observations for all managed agents
        current_obs = {k: v[0] for k, v in obs_info_dict.items()}

        # 2. --- Run Interaction Loop ---
        for t in range(max_steps):

            # --- A. Identify active agents ---
            active_ids = env.active_agents

            # Find which of the active agents are managed by this protocol
            # and haven't already finished
            agents_to_poll = {
                agent_id: self.agent_map[agent_id]
                for agent_id in active_ids
                if agent_id in self.agent_map and agent_id not in finished_agents
            }

            # --- B. Handle cases with no managed agents to poll ---
            if not agents_to_poll:
                if not active_ids:
                    # No active agents at all implies the game is effectively over
                    break

                # Check if all our managed agents have finished
                if len(finished_agents) >= len(self.agent_map):
                    # All managed agents done, no need to continue
                    break

                # Active agents exist but are not managed by us (e.g. internal bots/NPCs).
                # We must step the environment with an empty action set to let it proceed.
                outcomes_dict = env.step({})

                # Update observations for our agents (they might see what the bot did)
                for aid, out in outcomes_dict.items():
                    current_obs[aid] = out.obs
                    # We don't log a Step for our agents here (they didn't act),
                    # but we MUST update their context so they have the latest state.
                    if aid in self.agent_map and aid not in finished_agents:
                        self.agent_map[aid].on_after_step(out.obs, out.info)

                continue

            # --- C. Gather actions in parallel ---
            tasks = [
                agent.act(inference=inf, sampling_seed=sampling_seed, timeout_s=timeout_s)
                for agent in agents_to_poll.values()
            ]
            results = await asyncio.gather(*tasks)

            actions_to_take: Dict[str, str] = {}

            # Temporary storage to hold data needed for logging after the env step
            step_context_cache: Dict[str, Dict] = {}

            for agent_id, (parse_result, raw, info, token_trace) in zip(
                agents_to_poll.keys(), results
            ):
                # Cache context for this specific agent's step
                step_context_cache[agent_id] = {
                    "prev_obs": current_obs.get(agent_id, ""),
                    "raw_action": raw,
                    "client_info": info,
                    "parse_result": parse_result,
                    "token_trace": token_trace,
                }

                if parse_result.action is not None:
                    actions_to_take[agent_id] = parse_result.action

            # --- D. Step the environment ---
            # actions_to_take only contains VALID actions from managed agents.
            # Invalid ones are excluded here but handled below via parser_failures.
            env_outcomes: Dict[str, StepOutcome] = {}
            if actions_to_take:
                env_outcomes = env.step(actions_to_take)

            # --- E. Process Results & Log Steps ---
            for agent_id, ctx in step_context_cache.items():
                parse_result = ctx["parse_result"]
                client_info = ctx["client_info"]
                token_trace = ctx.get("token_trace")

                # Did parsing fail?
                if parse_result.action is None:
                    # Create synthetic outcome for parser failure
                    synthetic_obs = parse_result.obs or "Invalid action."
                    outcome = StepOutcome(
                        obs=synthetic_obs,
                        reward=parse_result.reward,
                        truncated=False,
                        terminated=False,
                        info=merge_step_info(
                            client_info=client_info,
                            extra={"parse_error": True},
                        ),
                        trace=token_trace,
                    )
                else:
                    # Normal environment outcome
                    # (It should exist in env_outcomes if we sent an action)
                    raw_outcome = env_outcomes.get(agent_id)
                    if not raw_outcome:
                        # Should theoretically not happen if logic is sound
                        continue

                    # Combine parser reward (usually 0.0) + Env reward
                    total_reward = raw_outcome.reward + parse_result.reward

                    outcome = StepOutcome(
                        obs=raw_outcome.obs,
                        reward=total_reward,
                        truncated=raw_outcome.truncated,
                        terminated=raw_outcome.terminated,
                        info=merge_step_info(
                            client_info=client_info,
                            env_info=raw_outcome.info,
                        ),
                        trace=token_trace,
                    )

                # Track if this agent has finished
                if outcome.terminated or outcome.truncated:
                    finished_agents.add(agent_id)

                # 1. Log the CLEAN single-agent step to the collector
                step = Step(
                    index=t,
                    prev_obs=ctx["prev_obs"],
                    action=ctx["raw_action"],
                    next_obs=(
                        outcome.obs
                        if not (outcome.terminated or outcome.truncated)
                        else None
                    ),
                    reward=outcome.reward,
                    truncated=outcome.truncated,
                    terminated=outcome.terminated,
                    info=outcome.info,
                    trace=outcome.trace,
                )
                collector.add(agent_id, step)

                # 2. Update Agent Context for the next turn
                # (Only if the episode isn't over for them)
                if not (outcome.terminated or outcome.truncated):
                    self.agent_map[agent_id].on_after_step(outcome.obs, outcome.info)
                    current_obs[agent_id] = outcome.obs

            # --- F. Check if all managed agents are done ---
            if len(finished_agents) >= len(self.agent_map):
                break

        # --- G. Handle Time-Limit Truncation ---
        # For agents who haven't finished yet (didn't terminate or get truncated by env),
        # mark their last step as truncated due to max_steps.
        for agent_id, steps in collector._traces.items():
            if not steps:
                continue

            last_step = steps[-1]

            # If this agent already terminated or was truncated by env, leave as-is
            if last_step.terminated or last_step.truncated:
                continue

            # This agent hit max_steps without finishing - mark as truncated
            last_step.truncated = True
            # Only clear next_obs if it wasn't a parser failure (which has synthetic obs).
            if not last_step.info.get("parse_error"):
                last_step.next_obs = None
            last_step.info = {**last_step.info, "truncation_reason": "max_steps"}

        # Extract rollouts with per-agent truncation info derived from their last step
        return collector.extract_rollouts_with_per_agent_status()
