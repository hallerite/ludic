from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import Tensor

from ludic.training.types import SAWItem, ActorTokenLogps, SampleAttachments


@dataclass(frozen=True)
class MicroBatch:
    tensors: Dict[str, Tensor]
    num_items: int


def collate_saw_items(
    items: List[SAWItem],
    *,
    pad_token_id: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    """
    Collate a list of SAWItem into a simple dense batch of tensors.

    - Left-aligns sequences and pads to max length in this batch.
    - Returns a dict suitable for RLAlgorithm.loss.compute():

          {
              "input_ids":      [B, T] long,
              "attention_mask": [B, T] long,
              "action_mask":    [B, T] float,
              "weight":         [B]    float,
          }
    """
    if not items:
        raise ValueError("Cannot collate empty list of SAWItems")

    lengths = [len(it.input_ids) for it in items]
    max_len = max(lengths)
    batch_size = len(items)

    input_ids = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
        device=device,
    )
    action_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.float32,
        device=device,
    )
    weights = torch.empty((batch_size,), dtype=torch.float32, device=device)

    for b, it in enumerate(items):
        L = len(it.input_ids)
        input_ids[b, :L] = torch.as_tensor(it.input_ids, dtype=torch.long, device=device)
        attention_mask[b, :L] = torch.as_tensor(it.attention_mask, dtype=torch.long, device=device)
        action_mask[b, :L] = torch.as_tensor(it.action_mask, dtype=torch.float32, device=device)
        weights[b] = float(it.weight)

    batch: Dict[str, Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "action_mask": action_mask,
        "weight": weights,
    }

    # actor_logps is optional; required for ratio objectives (PPO/GRPO).
    actors = [it.actor_logps for it in items]
    if any(actor is not None for actor in actors):
        if any(actor is None for actor in actors):
            raise ValueError(
                "Mixed presence of actor_logps; either provide it for all samples or none."
            )
        actor_logps_batch = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
        old_logp_action = torch.empty((batch_size,), dtype=torch.float32, device=device)
        for b, (it, actor) in enumerate(zip(items, actors)):
            assert actor is not None
            token_logps = torch.as_tensor(actor.token_logps, dtype=torch.float32, device=device)
            positions = torch.nonzero(action_mask[b] > 0.0, as_tuple=False).flatten()
            if token_logps.numel() != positions.numel():
                raise ValueError(
                    "Length mismatch between actor_logps and the number of action tokens."
                )
            actor_logps_batch[b, positions] = token_logps
            old_logp_action[b] = token_logps.sum()

        batch["actor_logps"] = actor_logps_batch
        batch["old_logp_action"] = old_logp_action
    return batch


def _truncate_item(item: SAWItem, max_seq_len: int) -> SAWItem:
    length = len(item.input_ids)
    if length <= max_seq_len:
        return item
    if len(item.attention_mask) != length or len(item.action_mask) != length:
        raise ValueError("SAWItem mask lengths must match input_ids length.")

    input_ids = item.input_ids[:max_seq_len]
    attention_mask = item.attention_mask[:max_seq_len]
    action_mask = item.action_mask[:max_seq_len]
    action_tokens = sum(1 for m in action_mask if int(m) == 1)
    prompt_tokens = len(input_ids) - action_tokens

    attachments = item.attachments
    if attachments.actor_logps is not None:
        token_logps = attachments.actor_logps.token_logps[:action_tokens]
        attachments = SampleAttachments(actor_logps=ActorTokenLogps(token_logps=token_logps))

    meta = dict(item.meta)
    meta["seq_len_truncated"] = True
    meta["seq_len_original"] = length
    meta["seq_len_retained"] = len(input_ids)
    meta["seq_len_retained_frac"] = float(len(input_ids)) / float(length) if length > 0 else 1.0
    meta["truncated"] = True
    if meta.get("truncation_reason") is None:
        meta["truncation_reason"] = "max_seq_len"
    if "completion_length" in meta:
        meta["completion_length"] = action_tokens
    if "prompt_length" in meta:
        meta["prompt_length"] = prompt_tokens

    return SAWItem(
        input_ids=input_ids,
        attention_mask=attention_mask,
        action_mask=action_mask,
        weight=item.weight,
        meta=meta,
        attachments=attachments,
    )


def split_items_by_token_budget(
    items: List[SAWItem],
    *,
    micro_token_budget: int,
    max_seq_len: int,
) -> List[List[SAWItem]]:
    if not items:
        return []
    if micro_token_budget <= 0:
        raise ValueError("micro_token_budget must be > 0.")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0.")
    if micro_token_budget < max_seq_len:
        raise ValueError(
            f"micro_token_budget ({micro_token_budget}) must be >= max_seq_len ({max_seq_len})."
        )

    max_item_len = max(len(it.input_ids) for it in items)
    if max_item_len <= max_seq_len:
        processed_items = items
    else:
        processed_items = [_truncate_item(it, max_seq_len) for it in items]
    order = sorted(range(len(processed_items)), key=lambda i: len(processed_items[i].input_ids))
    sorted_items = [processed_items[i] for i in order]

    micro_batches: List[List[SAWItem]] = []
    current: List[SAWItem] = []
    current_max = 0
    current_count = 0

    for it in sorted_items:
        length = len(it.input_ids)
        if length > max_seq_len:
            raise ValueError(
                f"SAWItem length {length} exceeds max_seq_len {max_seq_len} "
                f"(rollout_id={it.meta.get('rollout_id')!r}, step_index={it.meta.get('step_index')!r})."
            )

        next_max = length if current_count == 0 else max(current_max, length)
        next_count = current_count + 1
        if current and (next_max * next_count) > micro_token_budget:
            micro_batches.append(current)
            current = []
            current_max = 0
            current_count = 0
            next_max = length
            next_count = 1

        current.append(it)
        current_max = next_max
        current_count = next_count

    if current:
        micro_batches.append(current)

    return micro_batches


def collate_micro_batches(
    items: List[SAWItem],
    *,
    pad_token_id: int,
    device: torch.device,
    micro_token_budget: int,
    max_seq_len: int,
) -> List[MicroBatch]:
    micro_items = split_items_by_token_budget(
        items,
        micro_token_budget=micro_token_budget,
        max_seq_len=max_seq_len,
    )
    return [
        MicroBatch(
            tensors=collate_saw_items(
                chunk,
                pad_token_id=pad_token_id,
                device=device,
            ),
            num_items=len(chunk),
        )
        for chunk in micro_items
    ]
