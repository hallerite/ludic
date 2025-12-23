"""
Offline batch source for SFT and offline RL.

Reads pre-collected rollouts from JSONL files and converts them to SAWBatches.
Uses PyTorch DataLoader with DistributedSampler for proper multi-GPU support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from ludic.types import Rollout, Step, AgentStep, EnvironmentStep, TokenTrace
from ludic.training.filters import default_step_selector
from ludic.training.types import (
    BatchSource,
    CreditAssigner,
    SAWBatch,
    SAWItem,
    SampleFilter,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


# Type for the step-to-item conversion function
StepToItemFn = Callable[[Rollout, Step, float], SAWItem]


# ---------------------------------------------------------------------------
# Reusable step_to_item factories
# ---------------------------------------------------------------------------


def make_chat_template_step_to_item(
    tokenizer: "PreTrainedTokenizerBase",
) -> StepToItemFn:
    """
    Create a step_to_item function that uses HuggingFace chat templates.

    This is the standard preprocessing for SFT when your data contains
    chat-formatted prompts and completions.

    Expects step.info to contain:
        - chat_prompt_messages: List of chat messages (system, user, etc.)
        - chat_completion: Dict with {"role": "assistant", "content": "..."}
          (optional - falls back to step.action if not provided)

    Returns:
        A function suitable for OfflineBatchSource.step_to_item that tokenizes
        using the chat template and builds proper action masks.

    Example data format in JSONL::

        {
            "id": "game_001",
            "steps": [{
                "index": 0,
                "prev_obs": "...",
                "action": "<think>...</think><move>5</move>",
                "reward": 1.0,
                "info": {
                    "chat_prompt_messages": [
                        {"role": "system", "content": "You are playing Tic-Tac-Toe..."},
                        {"role": "user", "content": "Current board: ..."}
                    ],
                    "chat_completion": {"role": "assistant", "content": "<think>...</think><move>5</move>"}
                }
            }]
        }
    """

    def step_to_item(rollout: Rollout, step: Step, weight: float) -> SAWItem:
        info = step.info or {}
        chat_messages: List[Dict[str, str]] = info.get("chat_prompt_messages", [])
        chat_completion: Dict[str, str] = info.get("chat_completion", {})

        if not chat_messages:
            raise ValueError(
                f"step.info missing 'chat_prompt_messages' for "
                f"rollout_id={rollout.id!r} step_index={step.index}"
            )

        # Build completion message
        if chat_completion and isinstance(chat_completion, dict):
            completion_msg = chat_completion
        else:
            completion_msg = {"role": "assistant", "content": step.action}

        # Tokenize prompt (up to assistant turn)
        prompt_ids: List[int] = tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=True,
        )

        # Tokenize full conversation including completion
        full_ids: List[int] = tokenizer.apply_chat_template(
            chat_messages + [completion_msg],
            add_generation_prompt=False,
            tokenize=True,
        )

        # Verify alignment
        prompt_ids = list(prompt_ids)
        full_ids = list(full_ids)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Chat template prompt is not a prefix of full_ids for "
                f"rollout_id={rollout.id!r} step_index={step.index}; cannot align completion tokens."
            )

        state_ids = prompt_ids
        action_ids = full_ids[len(prompt_ids):]

        if not action_ids:
            raise ValueError(
                f"Chat template produced empty completion tokens for "
                f"rollout_id={rollout.id!r} step_index={step.index}."
            )

        # Build SAWItem
        input_ids = state_ids + action_ids
        attention_mask = [1] * len(input_ids)
        action_mask = [0] * len(state_ids) + [1] * len(action_ids)

        prev_obs = step.prev_obs if isinstance(step, EnvironmentStep) else ""
        meta: Dict[str, Any] = {
            "rollout_id": rollout.id,
            "step_index": step.index,
            "reward": float(step.reward),
            "prev_obs": prev_obs,
            "action": step.action,
            "total_reward": rollout.total_reward,
            "completion_length": len(action_ids),
            "prompt_length": len(state_ids),
            "truncated": step.truncated,
            "terminated": step.terminated,
            "step_kind": step.kind,
            "turn_id": step.turn_id,
            **(rollout.meta),
        }
        # Merge step info (but don't overwrite our keys)
        for k, v in info.items():
            if k not in meta:
                meta[k] = v

        return SAWItem(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            weight=weight,
            meta=meta,
        )

    return step_to_item


def _load_rollouts_from_jsonl(path: Path) -> List[Rollout]:
    """Load rollouts from a JSONL file."""
    rollouts: List[Rollout] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            steps: List[Step] = []
            for s in data.get("steps", []):
                kind = s.get("kind")
                trace_data = s.get("trace")
                if trace_data is None:
                    raise ValueError("Missing trace in offline rollout step.")
                trace = TokenTrace.from_dict(trace_data)
                common = {
                    "index": s["index"],
                    "reward": float(s["reward"]),
                    "truncated": s.get("truncated", False),
                    "terminated": s.get("terminated", False),
                    "info": s.get("info", {}),
                    "reward_components": s.get("reward_components", {}),
                    "trace": trace,
                    "id": s.get("id") or str(uuid.uuid4()),
                    "ts_ns": s.get("ts_ns", 0),
                    "turn_id": s.get("turn_id"),
                    "parent_id": s.get("parent_id"),
                }
                if kind == "agent":
                    steps.append(
                        AgentStep(
                            prompt_messages=s.get("prompt_messages", []),
                            action=s.get("action", ""),
                            action_target=s.get("action_target", "env"),
                            loop_index=s.get("loop_index", 0),
                            tool_calls=s.get("tool_calls"),
                            tool_results=s.get("tool_results"),
                            **common,
                        )
                    )
                elif kind == "env":
                    steps.append(
                        EnvironmentStep(
                            prev_obs=s.get("prev_obs", ""),
                            action=s.get("action", ""),
                            parsed_action=s.get("parsed_action"),
                            next_obs=s.get("next_obs"),
                            source_agent_step_id=s.get("source_agent_step_id", ""),
                            agent_step_ids=s.get("agent_step_ids", []),
                            **common,
                        )
                    )
                else:
                    raise ValueError(f"Unknown step kind in offline data: {kind!r}")
            rollout = Rollout(
                id=data.get("id", ""),
                steps=steps,
                meta=data.get("meta", {}),
            )
            rollouts.append(rollout)
    return rollouts


class _SAWItemDataset(Dataset[SAWItem]):
    """Simple Dataset wrapper around a list of SAWItems."""

    def __init__(self, items: List[SAWItem]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> SAWItem:
        return self._items[idx]


def _saw_item_collate_fn(items: List[SAWItem]) -> List[SAWItem]:
    """Identity collate - we do actual tensor collation in the Trainer."""
    return items


@dataclass
class OfflineBatchSource(BatchSource):
    """
    Batch source that reads from pre-collected rollout data.

    This is the data source for SFT and offline RL. It:
    1. Loads rollouts from JSONL file(s)
    2. Applies credit assignment to compute per-step weights
    3. Converts steps to SAWItems via user-provided function
    4. Yields batches using PyTorch DataLoader with proper distributed sampling

    Unlike RolloutBatchSource, this does not generate new rollouts -
    it operates on a fixed dataset.

    In distributed mode (torchrun), each rank automatically receives
    different samples via DistributedSampler.

    Args:
        jsonl_paths: Path(s) to JSONL files containing rollouts.
        step_to_item: Function that converts (rollout, step, weight) -> SAWItem.
            This is where tokenization and any preprocessing happens.
        credit_assigner: Computes per-step weights.
        batch_size: Number of SAWItems per macro-batch (per rank in distributed mode).
        shuffle: Whether to shuffle samples each epoch.
        sample_filter: Optional filter to drop samples.
        seed: Random seed for shuffling.
        num_workers: DataLoader workers for prefetching (0 = main process).
        drop_last: Whether to drop the last incomplete batch.
        step_selector: Optional filter over rollout steps; defaults to env
            steps plus env-targeted parse errors.
    """

    jsonl_paths: List[Path]
    step_to_item: StepToItemFn
    credit_assigner: CreditAssigner
    batch_size: int = 32
    shuffle: bool = True
    sample_filter: Optional[SampleFilter] = None
    seed: int = 42
    num_workers: int = 0
    drop_last: bool = False
    step_selector: Optional[Callable[[Step], bool]] = None

    # Private fields (initialized in __post_init__)
    _items: List[SAWItem] = field(default_factory=list, init=False, repr=False)
    _rollouts: List[Rollout] = field(default_factory=list, init=False, repr=False)
    _dataloader: Optional[DataLoader[List[SAWItem]]] = field(default=None, init=False, repr=False)
    _iterator: Optional[Iterator[List[SAWItem]]] = field(default=None, init=False, repr=False)
    _sampler: Optional[DistributedSampler[int]] = field(default=None, init=False, repr=False)
    _epoch: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        # Load all rollouts upfront
        self._rollouts = []
        for path in self.jsonl_paths:
            self._rollouts.extend(_load_rollouts_from_jsonl(path))

        if not self._rollouts:
            raise ValueError(f"No rollouts loaded from {self.jsonl_paths}")

        # Compute credit weights once for all rollouts
        weights = self.credit_assigner.compute(self._rollouts)
        selector = self.step_selector or default_step_selector

        # Build all SAWItems upfront (they're small, just token IDs + metadata)
        self._items = []
        for rollout in self._rollouts:
            for step in rollout.steps:
                if not selector(step):
                    continue
                key = (rollout.id, step.index)
                weight = weights.get(key, 1.0)
                item = self.step_to_item(rollout, step, weight)
                # Apply filter if provided
                if self.sample_filter is None or self.sample_filter(item):
                    self._items.append(item)

        if not self._items:
            raise ValueError("No samples after filtering")

        # Create Dataset and DataLoader
        dataset = _SAWItemDataset(self._items)

        is_distributed = dist.is_available() and dist.is_initialized()

        if is_distributed:
            # DistributedSampler shards data across ranks
            self._sampler = DistributedSampler(
                dataset,
                shuffle=self.shuffle,
                seed=self.seed,
                drop_last=self.drop_last,
            )
            sampler: Any = self._sampler
        elif self.shuffle:
            # Single-GPU with shuffling
            g = torch.Generator()
            g.manual_seed(self.seed)
            sampler = RandomSampler(dataset, generator=g)
        else:
            # Single-GPU without shuffling
            sampler = SequentialSampler(dataset)

        self._dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=_saw_item_collate_fn,
            pin_memory=False,  # SAWItems are not tensors yet
            drop_last=self.drop_last,
        )

        self._iterator = None
        self._epoch = 0

    def _next_epoch(self) -> None:
        """Advance to the next epoch and reset the iterator."""
        self._epoch += 1
        if self._sampler is not None:
            # DistributedSampler needs epoch set for proper shuffling
            self._sampler.set_epoch(self._epoch)
        self._iterator = iter(self._dataloader)  # type: ignore[arg-type]

    async def next_batch(self) -> SAWBatch:
        """
        Return the next batch of SAWItems.

        Cycles through the dataset, reshuffling each epoch if shuffle=True.
        In distributed mode, each rank receives different samples.
        """
        if self._iterator is None:
            if self._sampler is not None:
                self._sampler.set_epoch(self._epoch)
            self._iterator = iter(self._dataloader)  # type: ignore[arg-type]

        try:
            items: List[SAWItem] = next(self._iterator)
        except StopIteration:
            self._next_epoch()
            items = next(self._iterator)  # type: ignore[arg-type]

        # Compute batch metadata
        completion_lengths = [it.meta.get("completion_length", 0) for it in items]
        prompt_lengths = [it.meta.get("prompt_length", 0) for it in items]

        meta: Dict[str, Any] = {
            "num_samples": len(items),
            "epoch": self._epoch,
            "avg_completion_length": (
                sum(completion_lengths) / len(completion_lengths)
                if completion_lengths else 0.0
            ),
            "avg_prompt_length": (
                sum(prompt_lengths) / len(prompt_lengths)
                if prompt_lengths else 0.0
            ),
            "source": "offline",
        }

        return SAWBatch(items=items, meta=meta)

    def __len__(self) -> int:
        """Total number of samples in the dataset (before sharding)."""
        return len(self._items)

    @property
    def num_batches_per_epoch(self) -> int:
        """
        Number of batches per epoch (accounts for distributed sharding).

        In distributed mode, this is the number of batches each rank will see.
        """
        if self._dataloader is None:
            # Fallback before __post_init__ completes
            return (len(self._items) + self.batch_size - 1) // self.batch_size
        return len(self._dataloader)

    @property
    def target_rollouts(self) -> int:
        """Number of rollouts loaded."""
        return len(self._rollouts)

    @property
    def epoch(self) -> int:
        """Current epoch number."""
        return self._epoch

    def iter_epoch(self) -> Iterator[SAWBatch]:
        """
        Synchronous iterator over one epoch.

        Useful for simple training loops that don't need async.
        """
        import asyncio

        # Reset iterator for a fresh epoch
        if self._sampler is not None:
            self._sampler.set_epoch(self._epoch)
        self._iterator = iter(self._dataloader)  # type: ignore[arg-type]

        for _ in range(len(self._dataloader)):  # type: ignore[arg-type]
            yield asyncio.run(self.next_batch())

        # Advance epoch counter after iteration
        self._epoch += 1
