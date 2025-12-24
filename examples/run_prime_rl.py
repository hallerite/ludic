# run_prime_rl.py
import asyncio
import logging

from prime_rl.utils.pydantic_config import parse_argv

from environments.tic_tac_toe import TicTacToeEnv
from ludic.agent import Agent
from ludic.context import FullDialog
from ludic.inference import InferenceSpec, ReturnSpec, SamplingParams, VLLMChatClient
from ludic.interaction import SingleAgentSyncProtocol
from ludic.parsers import xml_tag_parser
from ludic.training import (
    EnvSpec,
    GroupNormalizedReturn,
    GRPORequestStrategy,
    ProtocolSpec,
    RolloutBatchSource,
    RolloutEngine,
    RolloutRequest,
)
from ludic.training.prime_rl import LudicConfig, PrimeOrchestrator


# -------------------------------------------------------------------------
# 1. Environment registry
# -------------------------------------------------------------------------

env_registry = {
    "tictactoe_agent_starts": lambda **kw: TicTacToeEnv(agent_starts=True, **kw),
    "tictactoe_opp_starts": lambda **kw: TicTacToeEnv(agent_starts=False, **kw),
}


# -------------------------------------------------------------------------
# 2. Protocol registry
# -------------------------------------------------------------------------

def create_single_agent_protocol(**kwargs):
    """
    Factory for the SingleAgentSyncProtocol used by Ludic's RolloutEngine.

    NOTE:
    - enable_weight_updates=False because Prime's admin_clients handle weight sync.
    - The prompt below is just a placeholder; you likely want to combine
      `TicTacToeEnv().suggested_sysprompt` with stricter XML instructions.
    """
    client = VLLMChatClient(
        host="127.0.0.1",
        port=8000,
        enable_weight_updates=False,
    )

    base_prompt = TicTacToeEnv().suggested_sysprompt or ""
    prompt = base_prompt + "\n\nOutput your move as a single XML tag, e.g., <move>A1</move>."

    agent = Agent(
        client=client,
        model="Qwen/Qwen2.5-7B-Instruct",
        ctx=FullDialog(system_prompt=prompt),
        parser=xml_tag_parser("move", exact=True, success_reward=0.0, error_reward=-1.0),
    )

    return SingleAgentSyncProtocol(agent=agent)


protocol_registry = {"single_agent_xml": create_single_agent_protocol}


# -------------------------------------------------------------------------
# 3. Main orchestration
# -------------------------------------------------------------------------

async def main():
    # 3.1 Parse combined Prime+Ludic config (from TOML / CLI)
    #
    # Expected TOML (rough sketch):
    # [orchestrator]
    # env_kind = "tictactoe_agent_starts"
    # protocol_kind = "single_agent_xml"
    # max_steps_per_episode = 5
    # rollout_concurrency = 64
    #
    # [orchestrator.sampling]
    # temperature = 0.7
    # max_tokens = 64
    #
    # and standard Prime RL fields (output_dir, num_train_workers, rollouts_per_example, ...)
    config = parse_argv(LudicConfig)
    if config.rollouts_per_example <= 0:
        raise ValueError("rollouts_per_example must be > 0.")
    if config.batch_size % config.rollouts_per_example != 0:
        raise ValueError("batch_size must be divisible by rollouts_per_example.")

    # 3.2 Build the Ludic rollout engine
    engine = RolloutEngine(
        env_registry=env_registry,
        protocol_registry=protocol_registry,
        jsonl_path=str(config.output_dir / "logs" / "rollouts.jsonl"),
    )

    # 3.3 Build credit assigner (GRPO-style group-normalized return)
    credit_assigner = GroupNormalizedReturn(
        group_size=config.rollouts_per_example,
        normalize_adv=True,
    )

    # 3.4 Define GRPO base requests: one per "group" / prompt
    train_inference = InferenceSpec(
        sampling=SamplingParams(
            temperature=config.sampling.temperature,
            max_tokens=config.sampling.max_tokens,
        ),
        # Prime expects per-token logprobs for the sampled completion tokens.
        return_=ReturnSpec.for_rl(),
    )

    def base_requests_fn():
        # Number of groups: each group will be expanded to G rollouts by GRPORequestStrategy,
        # where G = config.rollouts_per_example (must divide batch_size).
        num_groups = config.batch_size // config.rollouts_per_example

        # Use some reproducible base seed if available; otherwise 0
        base_seed = getattr(config, "seed", 0)

        return [
            RolloutRequest(
                env=EnvSpec(kind=config.env_kind, kwargs=config.env_kwargs),
                protocol=ProtocolSpec(
                    kind=config.protocol_kind,
                    kwargs=config.protocol_kwargs,
                ),
                num_episodes=1,
                env_seed=base_seed + i,
                sampling_seed=base_seed + i * config.rollouts_per_example,
                inference=train_inference,
            )
            for i in range(num_groups)
        ]

    # 3.5 Choose a BatchSource (GRPO here; RolloutBatchSource would also work)
    def requests_fn():
        return GRPORequestStrategy(group_size=config.rollouts_per_example).expand(
            base_requests_fn()
        )

    max_steps_per_episode = getattr(config, "max_steps_per_episode", None)
    if max_steps_per_episode is None:
        # Backward-compat fallback: older configs overloaded max_steps.
        max_steps_per_episode = getattr(config, "max_steps", 1)
    if max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be > 0.")
    rollout_concurrency = getattr(config, "rollout_concurrency", None)
    if rollout_concurrency is None:
        rollout_concurrency = min(128, config.batch_size)
    if rollout_concurrency <= 0:
        raise ValueError("rollout_concurrency must be > 0.")

    batch_source = RolloutBatchSource(
        orchestrator=engine,
        credit_assigner=credit_assigner,
        requests_fn=requests_fn,
        max_steps=int(max_steps_per_episode),
        concurrency=int(rollout_concurrency),
        timeout_s=getattr(config, "rollout_timeout_s", None),
    )

    # 3.6 Build and run the PrimeOrchestrator bridge
    #
    # PrimeOrchestrator:
    #   - pulls SAWBatches from batch_source
    #   - syncs weights with Prime trainer via broadcast dir
    #   - converts to Prime MicroBatches and writes rank_i.pt files
    orchestrator = PrimeOrchestrator(config=config, batch_source=batch_source)
    await orchestrator.loop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main())
