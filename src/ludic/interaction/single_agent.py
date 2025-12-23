from __future__ import annotations
from typing import Optional, List

from ludic.envs.env import LudicEnv
from ludic.agents.base_agent import Agent
from ludic.types import Rollout, Step, StepOutcome
from ludic.inference.request import InferenceSpec
from .base import InteractionProtocol
from .info import merge_step_info

class SingleAgentSyncProtocol(InteractionProtocol):
    """
    Implements the standard single-agent, synchronous interaction loop.
    
    This protocol consumes a LudicEnv but ASSUMES it has exactly
    one agent and that this agent is active every step.
    
    It works perfectly with any env inheriting from SingleAgentEnv.

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
    ):
        """
        Initializes the protocol.
        
        Args:
            agent: A fully-configured Agent instance.
            stop_on_parse_error:
                If True, terminate the episode after the first parser failure.
                If False (default), continue and feed the synthetic
                observation back to the agent.
        """
        self.agent = agent
        self.stop_on_parse_error = stop_on_parse_error

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
            parse_result, raw_action, client_info, token_trace = await agent.act(
                inference=inf,
                sampling_seed=sampling_seed,
                timeout_s=timeout_s
            )
            # --- B. Handle Parser Failure ---
            if parse_result.action is None:
                synthetic_obs = parse_result.obs or "Invalid action."
                parser_reward = parse_result.reward
                terminated_on_parse = bool(self.stop_on_parse_error)
                parse_halt = terminated_on_parse

                # Create a synthetic outcome
                outcome = StepOutcome(
                    obs=synthetic_obs,
                    reward=parser_reward,
                    truncated=False,
                    terminated=terminated_on_parse,
                    info=merge_step_info(
                        client_info=client_info,
                        extra={"parse_error": True},
                    ),
                    trace=token_trace,
                )
                
                # Log this failure step
                steps.append(Step(
                    index=t,
                    prev_obs=current_obs_for_step,
                    action=raw_action,
                    next_obs=synthetic_obs,
                    reward=parser_reward,
                    truncated=outcome.truncated,
                    terminated=outcome.terminated,
                    info=outcome.info,
                    trace=outcome.trace,
                ))
                # When stopping on parse error, we still want to feed the synthetic
                # observation back into the context before exiting.

            # --- C. Handle Parser Success (Step Env) ---
            else:
                parsed_action = parse_result.action
                parser_reward = parse_result.reward

                # Send action to env in the required dict format
                actions_dict = {agent_id: parsed_action}
                outcomes_dict = env.step(actions_dict)

                # Unwrap the outcome for our agent
                env_outcome = outcomes_dict[agent_id]
                
                # Combine parser and env rewards
                total_reward = env_outcome.reward + parser_reward
                
                # Merge info dicts (protect reserved training keys from collisions)
                step_info = merge_step_info(
                    client_info=client_info,
                    env_info=env_outcome.info,
                    extra={"parsed_action": parsed_action},
                )

                # Create the final, combined outcome
                outcome = StepOutcome(
                    obs=env_outcome.obs,
                    reward=total_reward,
                    truncated=env_outcome.truncated,
                    terminated=env_outcome.terminated,
                    info=step_info,
                    trace=token_trace,
                )

                # Log this success step
                logged_next_obs = None
                if not (outcome.terminated or outcome.truncated):
                    logged_next_obs = outcome.obs
                
                steps.append(Step(
                    index=t,
                    prev_obs=current_obs_for_step,
                    action=raw_action,
                    next_obs=logged_next_obs,
                    reward=total_reward,
                    truncated=outcome.truncated,
                    terminated=outcome.terminated,
                    info=step_info,
                    trace=outcome.trace,
                ))

            # --- D. Update state for next loop ---
            obs = outcome.obs
            info = outcome.info
            if parse_halt:
                agent.on_after_step(obs, info)
            if outcome.terminated or outcome.truncated:
                break # Exit loop
            else:
                # Feed the new observation to the agent
                agent.on_after_step(obs, info)

        # --- E. Handle Time-Limit Truncation ---
        # If we exited the loop without env termination/truncation, we hit max_steps.
        # This is a time-limit truncation and must be recorded for correct RL semantics.
        episode_truncated = False
        truncation_reason = None

        if steps:
            last_step = steps[-1]
            if last_step.terminated:
                # Normal termination from env
                pass
            elif last_step.truncated:
                # Env-initiated truncation
                episode_truncated = True
                truncation_reason = "env"
            else:
                # We completed max_steps without env ending the episode
                # This is a time-limit truncation
                episode_truncated = True
                truncation_reason = "max_steps"
                last_step.truncated = True
                # Only clear next_obs if it wasn't a parser failure (which has synthetic obs).
                # Parser failures have parse_error=True in info.
                if not last_step.info.get("parse_error"):
                    last_step.next_obs = None
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
