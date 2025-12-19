from __future__ import annotations


def test_top_level_exports_import() -> None:
    # Smoke test: examples should be able to avoid deep-importing internals.
    from ludic.agent import Agent, ToolAgent, ReActAgent  # noqa: F401
    from ludic.agents import Agent as Agent2  # noqa: F401
    from ludic.context import ContextStrategy, FullDialog, TruncatedThinkingContext  # noqa: F401
    from ludic.envs import LudicEnv, SingleAgentEnv, DatasetQAEnv  # noqa: F401
    from ludic.inference import VLLMChatClient, start_vllm_server, wait_for_vllm_health  # noqa: F401
    from ludic.interaction import InteractionProtocol, SingleAgentSyncProtocol, MultiAgentProtocol, TraceCollector  # noqa: F401
    from ludic.parsers import boxed_parser, xml_tag_parser, compose_parsers, think_prefix_parser  # noqa: F401
    from ludic.distributed import create_vllm_publisher  # noqa: F401
    from ludic.types import Rollout, Step  # noqa: F401


def test_training_exports_import() -> None:
    from ludic.training import (  # noqa: F401
        RolloutEngine,
        RolloutBatchSource,
        Trainer,
        TrainerConfig,
        RLAlgorithm,
        Reducer,
        apply_reducers_to_records,
        RichLiveLogger,
        PrintLogger,
        make_dataset_queue_requests_fn,
        make_requests_fn_from_queue,
    )
