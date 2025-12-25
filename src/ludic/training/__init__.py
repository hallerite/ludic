from __future__ import annotations

# Training-facing convenience imports.
#
# Ludic is meant to be hackable, so everything remains reachable via the internal
# module paths. This file provides a curated "short path" API for common usage.

from typing import TYPE_CHECKING, Any

from .types import (
    EnvSpec,
    ProtocolSpec,
    RolloutRequest,
    SAWItem,
    SAWBatch,
    BatchSource,
    CreditAssigner,
    TokenizeFn,
    SampleFilter,
)
from .algorithm import (
    RLAlgorithm,
    make_reinforce,
    make_reinforce_baseline,
    make_grpo,
    make_gspo,
    make_cispo,
    make_sft,
)
from .credit_assignment import (
    GroupNormalizedReturn,
    MonteCarloReturn,
    PerStepReward,
    EpisodicReturn,
    ConstantCredit,
)
from .loss import (
    Loss,
    ReinforceLoss,
    ReinforceBaselineLoss,
    ClippedSurrogateLoss,
    TokenClippedSurrogateLoss,
    CISPOLoss,
    KLLoss,
    EntropyBonus,
    LossTerm,
    CompositeLoss,
    selective_log_softmax,
)
from .filters import drop_truncated, drop_parse_errors, drop_incomplete_completions
from .config import TrainerConfig
from .checkpoint import CheckpointConfig
from .batching import (
    RolloutEngine,
    RolloutBatchSource,
    OfflineBatchSource,
    PipelineBatchSource,
    run_pipeline_actor,
    RequestStrategy,
    IdentityStrategy,
    GRPORequestStrategy,
    RequestsExhausted,
    make_requests_fn_from_queue,
    make_dataset_queue_requests_fn,
    make_dataset_sequence_requests_fn,
    make_chat_template_step_to_item,
)
from .stats import Reducer, apply_reducers_to_records, default_reducers
from .loggers import TrainingLogger, PrintLogger, RichLiveLogger, TeeLogger, WandbLogger

if TYPE_CHECKING:  # pragma: no cover
    from .trainer import Trainer as Trainer

__all__ = [
    # Core training loop
    "Trainer",
    "TrainerConfig",
    "CheckpointConfig",
    # Algorithm composition
    "RLAlgorithm",
    "make_reinforce",
    "make_reinforce_baseline",
    "make_grpo",
    "make_gspo",
    "make_cispo",
    "make_sft",
    # Credit assignment
    "GroupNormalizedReturn",
    "MonteCarloReturn",
    "PerStepReward",
    "EpisodicReturn",
    "ConstantCredit",
    # Losses
    "Loss",
    "ReinforceLoss",
    "ReinforceBaselineLoss",
    "ClippedSurrogateLoss",
    "TokenClippedSurrogateLoss",
    "CISPOLoss",
    "KLLoss",
    "EntropyBonus",
    "LossTerm",
    "CompositeLoss",
    "selective_log_softmax",
    # Sample filters
    "drop_truncated",
    "drop_parse_errors",
    "drop_incomplete_completions",
    # Core data types
    "EnvSpec",
    "ProtocolSpec",
    "RolloutRequest",
    "SAWItem",
    "SAWBatch",
    "BatchSource",
    "CreditAssigner",
    "TokenizeFn",
    "SampleFilter",
    # Batching / rollout execution
    "RolloutEngine",
    "RolloutBatchSource",
    "OfflineBatchSource",
    "PipelineBatchSource",
    "run_pipeline_actor",
    "RequestStrategy",
    "IdentityStrategy",
    "GRPORequestStrategy",
    "RequestsExhausted",
    "make_requests_fn_from_queue",
    "make_dataset_queue_requests_fn",
    "make_dataset_sequence_requests_fn",
    "make_chat_template_step_to_item",
    # Stats + loggers
    "Reducer",
    "apply_reducers_to_records",
    "default_reducers",
    "TrainingLogger",
    "PrintLogger",
    "RichLiveLogger",
    "TeeLogger",
    "WandbLogger",
]


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "Trainer":
        from .trainer import Trainer as _Trainer

        return _Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
