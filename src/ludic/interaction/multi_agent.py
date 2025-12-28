from __future__ import annotations
import asyncio
import uuid
from typing import Optional, Dict, List, Set, Any

from ludic.envs.env import LudicEnv
from ludic.agents.base_agent import Agent
from ludic.types import Rollout, Step, StepOutcome, AgentStep, EnvironmentStep
from ludic.inference.request import InferenceSpec
from .base import InteractionProtocol
from .step_collector import TraceCollector
from .info import merge_step_info
from .rewards import split_parser_reward


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
        step_indices = {agent_id: 0 for agent_id in self.agent_map}
        turn_ids = {agent_id: str(uuid.uuid4()) for agent_id in self.agent_map}
        turn_agent_step_ids: Dict[str, List[str]] = {
            agent_id: [] for agent_id in self.agent_map
        }

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
            step_context_cache: Dict[str, Dict[str, Any]] = {}

            for agent_id, act_result in zip(agents_to_poll.keys(), results):
                prev_obs = current_obs.get(agent_id, "")
                for act_step in act_result.steps:
                    parse_result = act_step.parse_result
                    # parse_result is None for external tool calls (not final actions)
                    parse_error = parse_result is not None and parse_result.action is None
                    extra_info = {
                        "parse_error": parse_error,
                        "action_target": act_step.action_target,
                    }
                    if parse_error and parse_result is not None and parse_result.obs is not None:
                        extra_info["parse_feedback_obs"] = parse_result.obs
                    if act_step.tool_calls is not None:
                        extra_info["tool_calls"] = act_step.tool_calls
                    if act_step.tool_results is not None:
                        extra_info["tool_results"] = act_step.tool_results
                    step_info = merge_step_info(
                        client_info=act_step.info,
                        extra=extra_info,
                    )
                    parser_reward = float(parse_result.reward) if parse_result else 0.0
                    agent_reward, agent_reward_components, _, _ = split_parser_reward(
                        parser_reward=parser_reward,
                        action_target=act_step.action_target,
                        parse_error=parse_error,
                    )
                    agent_step = AgentStep(
                        index=step_indices[agent_id],
                        prompt_messages=act_step.prompt_messages,
                        action=act_step.action,
                        action_target=act_step.action_target,
                        loop_index=act_step.loop_index,
                        reward=agent_reward,
                        reward_components=agent_reward_components,
                        truncated=False,
                        terminated=False,
                        info=step_info,
                        trace=act_step.trace,
                        turn_id=turn_ids[agent_id],
                        tool_calls=act_step.tool_calls,
                        tool_results=act_step.tool_results,
                    )
                    collector.add(agent_id, agent_step)
                    step_indices[agent_id] += 1
                    turn_agent_step_ids[agent_id].append(agent_step.id)

                final_step = act_result.final_step
                if final_step.action_target != "env":
                    continue

                if final_step.parse_result.action is None:
                    synthetic_obs = final_step.parse_result.obs or "Invalid action."
                    info = merge_step_info(
                        client_info=final_step.info,
                        extra={
                            "parse_error": True,
                            "parse_feedback_obs": synthetic_obs,
                        },
                    )
                    self.agent_map[agent_id].on_after_step(synthetic_obs, info)
                    current_obs[agent_id] = synthetic_obs
                    turn_ids[agent_id] = str(uuid.uuid4())
                    turn_agent_step_ids[agent_id] = []
                    continue

                actions_to_take[agent_id] = final_step.parse_result.action
                step_context_cache[agent_id] = {
                    "prev_obs": prev_obs,
                    "final_step": final_step,
                    "agent_step_ids": list(turn_agent_step_ids[agent_id]),
                }

            # --- D. Step the environment ---
            # actions_to_take only contains VALID actions from managed agents.
            # Invalid ones are excluded here but handled below via parser_failures.
            env_outcomes: Dict[str, StepOutcome] = {}
            if actions_to_take:
                env_outcomes = env.step(actions_to_take)

            # --- E. Process Results & Log Steps ---
            for agent_id, ctx in step_context_cache.items():
                raw_outcome = env_outcomes.get(agent_id)
                if not raw_outcome:
                    continue

                final_step = ctx["final_step"]
                parsed_action = final_step.parse_result.action
                if parsed_action is None:
                    continue

                parser_reward = float(final_step.parse_result.reward)
                _, _, env_parser_reward, env_reward_components = split_parser_reward(
                    parser_reward=parser_reward,
                    action_target=final_step.action_target,
                    parse_error=False,
                )
                step_info = merge_step_info(
                    client_info=final_step.info,
                    env_info=raw_outcome.info,
                    extra={
                        "parsed_action": parsed_action,
                        "source_agent_step_id": turn_agent_step_ids[agent_id][-1],
                        "agent_step_ids": ctx["agent_step_ids"],
                    },
                )
                logged_next_obs = None
                if not (raw_outcome.terminated or raw_outcome.truncated):
                    logged_next_obs = raw_outcome.obs

                env_step = EnvironmentStep(
                    index=step_indices[agent_id],
                    prev_obs=ctx["prev_obs"],
                    action=final_step.action,
                    parsed_action=parsed_action,
                    next_obs=logged_next_obs,
                    source_agent_step_id=turn_agent_step_ids[agent_id][-1],
                    agent_step_ids=ctx["agent_step_ids"],
                    reward=float(raw_outcome.reward) + env_parser_reward,
                    reward_components={
                        "env": float(raw_outcome.reward),
                        **env_reward_components,
                    },
                    truncated=raw_outcome.truncated,
                    terminated=raw_outcome.terminated,
                    info=step_info,
                    trace=final_step.trace,
                    turn_id=turn_ids[agent_id],
                )
                collector.add(agent_id, env_step)
                step_indices[agent_id] += 1
                turn_ids[agent_id] = str(uuid.uuid4())
                turn_agent_step_ids[agent_id] = []

                if raw_outcome.terminated or raw_outcome.truncated:
                    finished_agents.add(agent_id)
                else:
                    self.agent_map[agent_id].on_after_step(raw_outcome.obs, step_info)
                    current_obs[agent_id] = raw_outcome.obs

            # --- F. Check if all managed agents are done ---
            if len(finished_agents) >= len(self.agent_map):
                break

        # --- G. Handle Time-Limit Truncation ---
        # For agents who haven't finished yet (didn't terminate or get truncated by env),
        # mark their last step as truncated due to max_steps.
        for agent_id, steps in collector._traces.items():
            if not steps:
                continue

            last_env_step = next((s for s in reversed(steps) if s.kind == "env"), None)
            if last_env_step is None:
                last_step = steps[-1]
                if last_step.terminated or last_step.truncated:
                    continue
                last_step.truncated = True
                last_step.info = {**last_step.info, "truncation_reason": "max_steps"}
                continue

            if last_env_step.terminated or last_env_step.truncated:
                continue

            last_env_step.truncated = True
            last_env_step.next_obs = None
            last_env_step.info = {**last_env_step.info, "truncation_reason": "max_steps"}

        # Extract rollouts with per-agent truncation info derived from their last step
        return collector.extract_rollouts_with_per_agent_status()
