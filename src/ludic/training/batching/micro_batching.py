from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

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

    input_ids_list: List[Tensor] = []
    attn_mask_list: List[Tensor] = []
    action_mask_list: List[Tensor] = []
    weights_list: List[Tensor] = []

    for it in items:
        L = len(it.input_ids)

        ids = torch.full((max_len,), pad_token_id, dtype=torch.long)
        am = torch.zeros((max_len,), dtype=torch.long)
        actm = torch.zeros((max_len,), dtype=torch.float32)

        ids[:L] = torch.tensor(it.input_ids, dtype=torch.long)
        am[:L] = torch.tensor(it.attention_mask, dtype=torch.long)
        actm[:L] = torch.tensor(it.action_mask, dtype=torch.float32)

        input_ids_list.append(ids)
        attn_mask_list.append(am)
        action_mask_list.append(actm)
        weights_list.append(torch.tensor(it.weight, dtype=torch.float32))

    # actor_logps is optional; required for ratio objectives (PPO/GRPO).
    old_logp_action: List[Optional[float]] = []
    actor_logps_tokens: List[Optional[List[float]]] = []
    for it in items:
        actor = it.actor_logps
        actor_logps_tokens.append(None if actor is None else list(actor.token_logps))
        old_logp_action.append(None if actor is None else float(sum(float(v) for v in actor.token_logps)))

    batch: Dict[str, Tensor] = {
        "input_ids": torch.stack(input_ids_list, dim=0).to(device),
        "attention_mask": torch.stack(attn_mask_list, dim=0).to(device),
        "action_mask": torch.stack(action_mask_list, dim=0).to(device),
        "weight": torch.stack(weights_list, dim=0).to(device),
    }

    if any(v is not None for v in old_logp_action):
        if any(v is None for v in old_logp_action):
            raise ValueError(
                "Mixed presence of actor_logps; either provide it for all samples or none."
            )
        assert all(v is not None for v in actor_logps_tokens)

        # Optional token-level actor logps aligned to token positions.
        # Shape: [B, T], zeros outside action region / padding.
        actor_logps_batch = torch.zeros((len(items), max_len), dtype=torch.float32, device=device)
        for b, it in enumerate(items):
            token_logps = actor_logps_tokens[b]
            assert token_logps is not None
            action_positions = [i for i, m in enumerate(it.action_mask) if int(m) == 1]
            if len(token_logps) != len(action_positions):
                raise ValueError(
                    "Length mismatch between actor_logps and the number of action tokens."
                )
            for lp, pos in zip(token_logps, action_positions):
                actor_logps_batch[b, pos] = float(lp)

        batch["actor_logps"] = actor_logps_batch
        tensor_vals = [float(v) for v in old_logp_action]  # type: ignore[arg-type]
        batch["old_logp_action"] = torch.tensor(tensor_vals, dtype=torch.float32, device=device)
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
