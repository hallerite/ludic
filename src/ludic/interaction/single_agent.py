from __future__ import annotations
from typing import Any, Callable, Dict, Optional, List
import logging
import uuid

from ludic.envs.env import LudicEnv
from ludic.agents.base_agent import Agent
from ludic.types import Rollout, Step, AgentStep, EnvironmentStep
from ludic.inference.request import InferenceSpec
from .base import InteractionProtocol
from .info import merge_step_info
from .rewards import split_parser_reward

logger = logging.getLogger(__name__)

# Type for external tool handlers: takes tool_calls, returns list of result dicts
# Each result dict should have: tool_call_id, tool_name, content
ExternalToolHandler = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]


class SingleAgentSyncProtocol(InteractionProtocol):
    """
    Implements the standard single-agent, synchronous interaction loop.

    This protocol consumes a LudicEnv but ASSUMES it has exactly
    one agent and that this agent is active every step.

    It works perfectly with any env inheriting from SingleAgentEnv.

    Action targets:
      - "internal": Agent handled internally, loop continues automatically.
      - "external": Agent returns to protocol. If external_tool_handler is set,
        protocol handles it and feeds result back to agent. Otherwise, error.
      - "env": Final action for environment. Parsed and sent to env.step().

    Parser failures:
      If the agent's parser returns ParseResult.action=None, the protocol
      does not call env.step(). Instead it logs a synthetic Step with
      reward=parse_result.reward, next_obs=parse_result.obs (or
      "Invalid action."), and info including parse_error=True. The
      synthetic observation is fed back into the agent context for the
      next turn. Optionally, the protocol can terminate immediately on the
      first parse error.

    System prompt resolution:
      Uses the context's default_system_prompt if present; otherwise falls
      back to env.suggested_sysprompt. There is no protocol-level prompt
      override.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        stop_on_parse_error: bool = False,
        external_tool_handler: Optional[ExternalToolHandler] = None,
    ):
        """
        Initializes the protocol.

        Args:
            agent: A fully-configured Agent instance.
            stop_on_parse_error:
                If True, terminate the episode after the first parser failure.
                If False (default), continue and feed the synthetic
                observation back to the agent.
            external_tool_handler:
                Optional handler for external tool calls. If provided, when
                agent returns with action_target="external", this handler is
                called with the tool_calls and the result is fed back to the
                agent's context. The agent then continues reasoning.
        """
        self.agent = agent
        self.stop_on_parse_error = stop_on_parse_error
        self.external_tool_handler = external_tool_handler

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
        
        agent = self.agent
        inf = inference or InferenceSpec()

        # 1. --- Validate Env and get Agent ID ---
        agent_ids = env.agent_ids
        if len(agent_ids) != 1:
            raise ValueError(
                f"SingleAgentSyncProtocol requires a LudicEnv with "
                f"exactly one agent, but found {len(agent_ids)}."
            )
        agent_id = agent_ids[0]

        # 2. --- Reset Env ---
        # env.reset() returns a dict
        obs_info_dict = env.reset(seed=env_seed)
        obs, info = obs_info_dict[agent_id]
        
        # 3. --- Reset Agent & Feed First Obs ---
        # Choose system prompt: prefer the context's default if set, else env suggestion.
        ctx_default_prompt = getattr(getattr(agent, "_ctx", None), "default_system_prompt", None)
        env_prompt = getattr(env, "suggested_sysprompt", None)
        sys_prompt = ctx_default_prompt or env_prompt

        agent.reset(system_prompt=sys_prompt)
        agent.on_env_reset(obs, info)
        
        # Accumulate steps locally first
        steps: List[Step] = []
        step_index = 0
        turn_id = str(uuid.uuid4())
        turn_agent_step_ids: List[str] = []
        ended_by_max_steps = True

        # 4. --- Run Interaction Loop ---
        for t in range(max_steps):
            parse_halt = False
            
            # Check that our agent is the one expected to act
            active_agents = env.active_agents
            if agent_id not in active_agents:
                # This env is not a simple single-agent env, stop.
                break 

            current_obs_for_step = obs
            
            # --- A. Call the Agent ---
            act_result = await agent.act(
                inference=inf,
                sampling_seed=sampling_seed,
                timeout_s=timeout_s
            )
            # --- B. Record Agent Steps ---
            for act_step in act_result.steps:
                parse_result = act_step.parse_result
                # parse_result is None for external tool calls (not final actions)
                parse_error = parse_result is not None and parse_result.action is None
                terminated_on_parse = bool(
                    parse_error and act_step.action_target == "env" and self.stop_on_parse_error
                )
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
                    index=step_index,
                    prompt_messages=act_step.prompt_messages,
                    action=act_step.action,
                    action_target=act_step.action_target,
                    loop_index=act_step.loop_index,
                    reward=agent_reward,
                    reward_components=agent_reward_components,
                    truncated=False,
                    terminated=terminated_on_parse,
                    info=step_info,
                    trace=act_step.trace,
                    turn_id=turn_id,
                    tool_calls=act_step.tool_calls,
                    tool_results=act_step.tool_results,
                )
                steps.append(agent_step)
                turn_agent_step_ids.append(agent_step.id)
                step_index += 1

            # --- C. Handle External Tool Calls ---
            final_step = act_result.final_step

            while final_step.action_target == "external":
                if self.external_tool_handler is None:
                    raise RuntimeError(
                        "Agent returned external tool call but no external_tool_handler "
                        "is configured. Either provide an external_tool_handler to the "
                        "protocol or remove external_tools from the agent."
                    )

                # Call the external handler with the tool calls
                tool_results = self.external_tool_handler(final_step.tool_calls or [])

                # Feed results back to agent's context
                for result in tool_results:
                    agent._ctx.add_tool_result(
                        result["tool_call_id"],
                        result["tool_name"],
                        result["content"],
                    )

                # Continue agent reasoning
                act_result = await agent.act(
                    inference=inf,
                    sampling_seed=sampling_seed,
                    timeout_s=timeout_s,
                )

                # Record the new agent steps
                for act_step in act_result.steps:
                    parse_result = act_step.parse_result
                    parse_error = parse_result is not None and parse_result.action is None
                    terminated_on_parse = bool(
                        parse_error and act_step.action_target == "env" and self.stop_on_parse_error
                    )
                    extra_info = {
                        "parse_error": parse_error,
                        "action_target": act_step.action_target,
                    }
                    if parse_error and parse_result.obs is not None:
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
                        index=step_index,
                        prompt_messages=act_step.prompt_messages,
                        action=act_step.action,
                        action_target=act_step.action_target,
                        loop_index=act_step.loop_index,
                        reward=agent_reward,
                        reward_components=agent_reward_components,
                        truncated=False,
                        terminated=terminated_on_parse,
                        info=step_info,
                        trace=act_step.trace,
                        turn_id=turn_id,
                        tool_calls=act_step.tool_calls,
                        tool_results=act_step.tool_results,
                    )
                    steps.append(agent_step)
                    turn_agent_step_ids.append(agent_step.id)
                    step_index += 1

                # Update final_step for next iteration check
                final_step = act_result.final_step

            # --- D. Handle Env Action ---
            if final_step.action_target != "env":
                # Internal action - agent handled internally, continue to next env step
                continue

            parse_result = final_step.parse_result
            if parse_result.action is None:
                synthetic_obs = parse_result.obs or "Invalid action."
                parse_halt = bool(self.stop_on_parse_error)
                obs = synthetic_obs
                info = merge_step_info(
                    client_info=final_step.info,
                    extra={
                        "parse_error": True,
                        "parse_feedback_obs": synthetic_obs,
                    },
                )
                agent.on_after_step(obs, info)
                turn_id = str(uuid.uuid4())
                turn_agent_step_ids = []
                if parse_halt:
                    ended_by_max_steps = False
                    break
                continue

            parsed_action = parse_result.action
            parser_reward = float(parse_result.reward)
            _, _, env_parser_reward, env_reward_components = split_parser_reward(
                parser_reward=parser_reward,
                action_target=final_step.action_target,
                parse_error=False,
            )
            actions_dict = {agent_id: parsed_action}
            outcomes_dict = env.step(actions_dict)
            env_outcome = outcomes_dict[agent_id]

            step_info = merge_step_info(
                client_info=final_step.info,
                env_info=env_outcome.info,
                extra={
                    "parsed_action": parsed_action,
                    "source_agent_step_id": steps[-1].id,
                    "agent_step_ids": list(turn_agent_step_ids),
                },
            )
            logged_next_obs = None
            if not (env_outcome.terminated or env_outcome.truncated):
                logged_next_obs = env_outcome.obs

            env_step = EnvironmentStep(
                index=step_index,
                prev_obs=current_obs_for_step,
                action=final_step.action,
                parsed_action=parsed_action,
                next_obs=logged_next_obs,
                source_agent_step_id=steps[-1].id,
                agent_step_ids=list(turn_agent_step_ids),
                reward=float(env_outcome.reward) + env_parser_reward,
                reward_components={
                    "env": float(env_outcome.reward),
                    **env_reward_components,
                },
                truncated=env_outcome.truncated,
                terminated=env_outcome.terminated,
                info=step_info,
                trace=final_step.trace,
                turn_id=turn_id,
            )
            steps.append(env_step)
            step_index += 1

            obs = env_outcome.obs
            info = step_info
            if env_outcome.terminated or env_outcome.truncated:
                ended_by_max_steps = False
                break

            agent.on_after_step(obs, info)
            turn_id = str(uuid.uuid4())
            turn_agent_step_ids = []

        # --- E. Handle Time-Limit Truncation ---
        # If we exited the loop without env termination/truncation, we hit max_steps.
        # This is a time-limit truncation and must be recorded for correct RL semantics.
        episode_truncated = False
        truncation_reason = None

        if steps:
            last_env_step = next((s for s in reversed(steps) if s.kind == "env"), None)
            if last_env_step is not None:
                if last_env_step.terminated:
                    pass
                elif last_env_step.truncated:
                    episode_truncated = True
                    truncation_reason = "env"
                elif ended_by_max_steps:
                    episode_truncated = True
                    truncation_reason = "max_steps"
                    last_env_step.truncated = True
                    last_env_step.next_obs = None
                    last_env_step.info = {**last_env_step.info, "truncation_reason": "max_steps"}
            else:
                last_step = steps[-1]
                if ended_by_max_steps and not last_step.terminated and not last_step.truncated:
                    episode_truncated = True
                    truncation_reason = "max_steps"
                    last_step.truncated = True
                    last_step.info = {**last_step.info, "truncation_reason": "max_steps"}

        rollout = Rollout(
            steps=steps,
            meta={
                "agent_name": getattr(agent, "name", "unknown"),
                "env_name": env.__class__.__name__,
                "episode_truncated": episode_truncated,
                "truncation_reason": truncation_reason,
            }
        )
        return [rollout]
