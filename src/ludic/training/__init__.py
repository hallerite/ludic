from __future__ import annotations

# Training-facing convenience imports.
#
# Ludic is meant to be hackable, so everything remains reachable via the internal
# module paths. This file provides a curated "short path" API for common usage.

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
    make_spo,
)
from .credit_assignment import (
    GroupNormalizedReturn,
    MonteCarloReturn,
    PerStepReward,
    EpisodicReturn,
    SPOReturn,
)
from .value_tracker import ValueTracker
from .loss import (
    Loss,
    ReinforceLoss,
    ReinforceBaselineLoss,
    ClippedSurrogateLoss,
    KLLoss,
    EntropyBonus,
    LossTerm,
    CompositeLoss,
    selective_log_softmax,
)
from .filters import drop_truncated, drop_parse_errors, drop_incomplete_completions
from .trainer import Trainer
from .config import TrainerConfig
from .checkpoint import CheckpointConfig
from .batching import (
    RolloutEngine,
    RolloutBatchSource,
    PipelineBatchSource,
    run_pipeline_actor,
    RequestStrategy,
    IdentityStrategy,
    GRPORequestStrategy,
    RequestsExhausted,
    make_requests_fn_from_queue,
    make_dataset_queue_requests_fn,
    make_dataset_sequence_requests_fn,
)
from .stats import Reducer, apply_reducers_to_records
from .loggers import TrainingLogger, PrintLogger, RichLiveLogger

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
    "make_spo",
    # Credit assignment
    "GroupNormalizedReturn",
    "MonteCarloReturn",
    "PerStepReward",
    "EpisodicReturn",
    "SPOReturn",
    "ValueTracker",
    # Losses
    "Loss",
    "ReinforceLoss",
    "ReinforceBaselineLoss",
    "ClippedSurrogateLoss",
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
    "PipelineBatchSource",
    "run_pipeline_actor",
    "RequestStrategy",
    "IdentityStrategy",
    "GRPORequestStrategy",
    "RequestsExhausted",
    "make_requests_fn_from_queue",
    "make_dataset_queue_requests_fn",
    "make_dataset_sequence_requests_fn",
    # Stats + loggers
    "Reducer",
    "apply_reducers_to_records",
    "TrainingLogger",
    "PrintLogger",
    "RichLiveLogger",
]
