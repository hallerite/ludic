# Plan: On-Policy Distillation for Ludic

## Overview

On-Policy Distillation (OPD) combines the benefits of on-policy RL (learning from own mistakes) with dense supervision from a teacher model. The core idea:

1. **Sample trajectories from the student** (same as RL)
2. **Compute per-token reverse KL** against a teacher: `KL = log π_student(x) - log π_teacher(x)`
3. **Set per-token advantage to `-reverse_kl`** (reward tokens the teacher likes)
4. **Train with policy gradient** using these dense per-token rewards

### Key Equation

```
reverse_kl_t = log π_θ(x_t | x_{<t}) - log π_teacher(x_t | x_{<t})
advantage_t = -reverse_kl_t = log π_teacher - log π_student
loss = -E[advantage_t * log π_θ(x_t | x_{<t})]
```

This is equivalent to minimizing `E[(log π_student - log π_teacher)²]` in gradient direction.

---

## Ludic Architecture Fit

Ludic's `RLAlgorithm = (CreditAssigner, Loss, optional Preprocess)` pattern is well-suited for OPD:

| Component | Standard RL | OPD |
|-----------|-------------|-----|
| **Sampling** | Student rollouts | Student rollouts (same) |
| **Credit** | Env reward → weight | Per-token teacher KL → weight |
| **Loss** | `ReinforceLoss` or `ClippedSurrogateLoss` | New `ReverseKLLoss` |
| **Preprocess** | Backfill old logprobs | Compute teacher logprobs |

### What Already Exists

- ✅ `step.info["completion_logprobs"]` - student's per-token logprobs from sampling
- ✅ `SAWItem.meta["completion_logprobs"]` - preserved through pipeline
- ✅ `RLAlgorithm.preprocess` hook - can run inference before collation
- ✅ `_ensure_old_token_logprobs()` - pattern for backfilling logprobs
- ✅ `action_mask` - identifies which tokens are actions vs prompts
- ✅ Per-token operations in loss (`compute_logp_action`, `selective_log_softmax`)

### What's Missing

- ❌ Teacher model access during training
- ❌ Teacher logprob storage in metadata
- ❌ Loss that uses per-token KL as the training signal
- ❌ Algorithm preset `make_opd()` or `make_on_policy_distillation()`

---

## Implementation Options

### Option A: Preprocess Hook + New Loss (Recommended)

**Philosophy**: Teacher logprobs are computed in preprocess (like `old_token_logprobs`), loss consumes them.

**Changes**:

1. **New preprocess function** in `algorithm.py`:
   ```python
   def _compute_teacher_logprobs(
       saw_batch: SAWBatch,
       *,
       teacher_model: nn.Module,  # or ChatClient
       pad_token_id: int,
   ) -> SAWBatch:
       """Compute teacher logprobs for each item, store in meta."""
       for item in saw_batch.items:
           # Run teacher forward pass (no grad)
           teacher_logprobs = teacher_forward(teacher_model, item.input_ids)
           item.meta["teacher_token_logprobs"] = teacher_logprobs
       return saw_batch
   ```

2. **Extend collation** in `trainer.py`:
   ```python
   # In _collate_saw_items, after old_logp_action handling:
   if "teacher_token_logprobs" in items[0].meta:
       batch["teacher_token_logprobs"] = padded_tensor(...)  # [B, T]
   ```

3. **New loss** in `loss.py`:
   ```python
   @dataclass
   class ReverseKLLoss(Loss):
       """Per-token reverse KL distillation loss."""

       def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict]:
           # Student log probs from current forward pass
           student_logp = selective_log_softmax(logits, batch["input_ids"])

           # Teacher log probs from preprocess
           teacher_logp = batch["teacher_token_logprobs"]

           # Per-token reverse KL (student's perspective)
           reverse_kl = student_logp - teacher_logp  # minimize this

           # Masked mean over action tokens
           action_mask = batch["action_mask"]
           loss = (reverse_kl * action_mask).sum() / action_mask.sum()

           return loss, {"reverse_kl": loss.item()}
   ```

4. **Algorithm preset** in `algorithm.py`:
   ```python
   def make_opd(
       teacher_model: nn.Module,
       pad_token_id: int,
       credit_assigner: Optional[CreditAssigner] = None,  # default: ConstantCredit
   ) -> RLAlgorithm:
       credit = credit_assigner or ConstantCredit(1.0)
       loss = ReverseKLLoss()
       preprocess = partial(_compute_teacher_logprobs,
                            teacher_model=teacher_model,
                            pad_token_id=pad_token_id)
       return RLAlgorithm("opd", credit, loss, preprocess)
   ```

**Pros**:
- Follows existing patterns (`_ensure_old_token_logprobs`)
- Clean separation: preprocess computes, loss consumes
- Teacher model can be swapped easily
- Credit assigner still usable for hybrid OPD+RL

**Cons**:
- Teacher forward pass in preprocess (CPU→GPU transfer per batch)
- Need to handle teacher model device placement

---

### Option B: Self-Contained Loss with Teacher

**Philosophy**: Loss owns the teacher model and computes everything internally.

**Changes**:

1. **New loss** in `loss.py`:
   ```python
   @dataclass
   class OnPolicyDistillationLoss(Loss):
       teacher_model: nn.Module

       def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict]:
           # Student logprobs from logits
           student_logp = selective_log_softmax(logits, batch["input_ids"])

           # Teacher logprobs (no grad)
           with torch.no_grad():
               teacher_logits = self.teacher_model(
                   input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"]
               ).logits
               teacher_logp = selective_log_softmax(teacher_logits, batch["input_ids"])

           # Reverse KL
           reverse_kl = student_logp - teacher_logp
           action_mask = batch["action_mask"]
           loss = (reverse_kl * action_mask).sum() / action_mask.sum()

           return loss, {"reverse_kl": loss.item()}
   ```

2. **Algorithm preset**:
   ```python
   def make_opd(teacher_model: nn.Module) -> RLAlgorithm:
       return RLAlgorithm(
           name="opd",
           credit_assigner=ConstantCredit(1.0),
           loss=OnPolicyDistillationLoss(teacher_model),
       )
   ```

**Pros**:
- Minimal changes (just one new loss class)
- Self-contained, easy to understand
- No changes to collation or preprocess

**Cons**:
- Teacher forward pass during backward (memory pressure)
- Loss has side effects (model inference)
- Harder to extend (e.g., caching teacher logprobs)

---

### Option C: Teacher Logprobs at Rollout Time

**Philosophy**: Compute teacher logprobs alongside student during rollout generation.

**Changes**:

1. **Extend RolloutEngine** to accept optional `teacher_client`:
   ```python
   class RolloutEngine:
       def __init__(self, ..., teacher_client: Optional[ChatClient] = None):
           self.teacher_client = teacher_client
   ```

2. **After each step**, compute teacher logprobs:
   ```python
   # In interaction protocol or RolloutEngine post-processing
   if self.teacher_client:
       teacher_resp = self.teacher_client.compute_logprobs(
           messages=step.info["chat_prompt_messages"],
           completion=step.action,
       )
       step.info["teacher_completion_logprobs"] = teacher_resp.logprobs
   ```

3. **SAWItem creation** preserves teacher logprobs in `meta`

4. **Simple loss** reads from `batch["teacher_token_logprobs"]`

**Pros**:
- Teacher computation happens during rollout (can overlap with env)
- Clean data flow
- Enables filtering based on teacher score before training

**Cons**:
- Changes core RolloutEngine interface
- Teacher inference latency added to rollout
- Less flexible (teacher must be available at rollout time)

---

### Option D: Hybrid OPD + Environment Reward

**Philosophy**: Combine dense teacher signal with sparse environment rewards.

This extends Option A with weighted combination:

```python
@dataclass
class HybridOPDLoss(Loss):
    """Combines reverse KL with environment reward signal."""
    kl_weight: float = 1.0
    reward_weight: float = 0.1

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict]:
        student_logp = selective_log_softmax(logits, batch["input_ids"])
        teacher_logp = batch["teacher_token_logprobs"]

        # Dense KL term
        reverse_kl = student_logp - teacher_logp
        kl_loss = (reverse_kl * batch["action_mask"]).sum() / batch["action_mask"].sum()

        # Sparse reward term (uses credit-assigned weights)
        reward_loss = -(batch["weight"] * student_logp.sum(-1)).mean()

        loss = self.kl_weight * kl_loss + self.reward_weight * reward_loss

        return loss, {
            "kl_loss": kl_loss.item(),
            "reward_loss": reward_loss.item(),
        }
```

**Use case**: Train math reasoning with teacher supervision + correctness reward.

---

## Recommendation

**Start with Option A** (Preprocess Hook + New Loss) because:

1. **Matches existing patterns** - `_ensure_old_token_logprobs` is the template
2. **Minimal core changes** - no RolloutEngine modifications
3. **Composable** - can combine with any credit assigner for hybrid approaches
4. **Testable** - preprocess and loss can be unit tested separately

### Implementation Order

1. Add `ReverseKLLoss` to `loss.py` (~30 lines)
2. Add `_compute_teacher_logprobs` preprocess to `algorithm.py` (~40 lines)
3. Add collation support for `teacher_token_logprobs` in `trainer.py` (~10 lines)
4. Add `make_opd()` preset to `algorithm.py` (~15 lines)
5. Add example script `examples/opd/train_opd.py`

**Total**: ~100 lines of library code + example.

---

## File-by-File Changes

### `src/ludic/training/loss.py`

```python
# Add after ReinforceLoss

@dataclass
class ReverseKLLoss(Loss):
    """
    On-Policy Distillation loss using per-token reverse KL.

    Minimizes KL(π_student || π_teacher) by training on student samples
    with teacher logprobs as targets.

    Requires batch["teacher_token_logprobs"] from preprocessing.
    """
    length_normalize: bool = False

    def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict[str, Any]]:
        input_ids = batch["input_ids"]
        action_mask = batch["action_mask"]
        teacher_logp = batch["teacher_token_logprobs"]  # [B, T]

        # Current policy logprobs
        student_logp = compute_token_logprobs(logits, input_ids)  # [B, T]

        # Per-token reverse KL (student - teacher), masked to actions
        reverse_kl = (student_logp - teacher_logp) * action_mask

        if self.length_normalize:
            lengths = action_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            reverse_kl = reverse_kl / lengths

        # Mean over batch
        loss = reverse_kl.sum() / action_mask.sum().clamp(min=1)

        stats = {
            "reverse_kl": loss.item(),
            "mean_student_logp": (student_logp * action_mask).sum().item() / action_mask.sum().item(),
            "mean_teacher_logp": (teacher_logp * action_mask).sum().item() / action_mask.sum().item(),
        }

        return loss, stats
```

### `src/ludic/training/algorithm.py`

```python
# Add teacher logprob computation

def _compute_teacher_logprobs(
    saw_batch: SAWBatch,
    *,
    teacher_model: nn.Module,
    pad_token_id: int,
    chunk_size: int = 8,
) -> SAWBatch:
    """
    Compute teacher model logprobs for OPD.

    Similar to _ensure_old_token_logprobs but uses external teacher.
    """
    import torch

    items = saw_batch.items
    device = next(teacher_model.parameters()).device

    # Process in chunks to manage memory
    for chunk_start in range(0, len(items), chunk_size):
        chunk = items[chunk_start : chunk_start + chunk_size]

        # Collate chunk
        max_len = max(len(item.input_ids) for item in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_token_id, device=device)
        attention_mask = torch.zeros((len(chunk), max_len), device=device)

        for i, item in enumerate(chunk):
            seq_len = len(item.input_ids)
            input_ids[i, :seq_len] = torch.tensor(item.input_ids, device=device)
            attention_mask[i, :seq_len] = 1

        # Teacher forward (no grad)
        with torch.no_grad():
            outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, T, V]

        # Extract per-token logprobs
        log_probs = torch.log_softmax(logits, dim=-1)

        # Shift for next-token prediction
        shift_logprobs = log_probs[:, :-1, :]  # [B, T-1, V]
        shift_targets = input_ids[:, 1:]        # [B, T-1]

        # Gather logprobs for actual tokens
        token_logprobs = shift_logprobs.gather(
            dim=-1, index=shift_targets.unsqueeze(-1)
        ).squeeze(-1)  # [B, T-1]

        # Store back in items
        for i, item in enumerate(chunk):
            seq_len = len(item.input_ids) - 1  # -1 for shift
            item.meta["teacher_token_logprobs"] = token_logprobs[i, :seq_len].cpu().tolist()

    return saw_batch


def make_opd_preprocessor(
    teacher_model: nn.Module,
    pad_token_id: int,
    chunk_size: int = 8,
) -> PreprocessFn:
    """Create OPD preprocessor that computes teacher logprobs."""
    from functools import partial
    return partial(
        _compute_teacher_logprobs,
        teacher_model=teacher_model,
        pad_token_id=pad_token_id,
        chunk_size=chunk_size,
    )


def make_opd(
    teacher_model: nn.Module,
    pad_token_id: int,
    *,
    length_normalize: bool = False,
    chunk_size: int = 8,
) -> RLAlgorithm:
    """
    Create On-Policy Distillation algorithm.

    Trains student to match teacher's per-token distribution on student's
    own trajectories. Combines on-policy sampling with dense supervision.

    Args:
        teacher_model: Model to distill from (must be on same device)
        pad_token_id: Tokenizer pad token ID
        length_normalize: Normalize KL by sequence length
        chunk_size: Batch size for teacher inference

    Returns:
        RLAlgorithm configured for on-policy distillation
    """
    return RLAlgorithm(
        name="opd",
        credit_assigner=ConstantCredit(1.0),  # Weight handled by KL
        loss=ReverseKLLoss(length_normalize=length_normalize),
        preprocess=make_opd_preprocessor(teacher_model, pad_token_id, chunk_size),
    )
```

### `src/ludic/training/trainer.py`

```python
# In _collate_saw_items, add after old_logp_action handling (~line 95):

    # Teacher token logprobs for OPD
    if items and "teacher_token_logprobs" in items[0].meta:
        teacher_logprobs_list = []
        for item in items:
            logprobs = item.meta.get("teacher_token_logprobs", [])
            # Pad to max_len - 1 (shifted)
            padded = logprobs + [0.0] * (max_len - 1 - len(logprobs))
            teacher_logprobs_list.append(padded[:max_len - 1])

        batch["teacher_token_logprobs"] = torch.tensor(
            teacher_logprobs_list, dtype=torch.float32, device=device
        )
```

---

## Usage Example

```python
# examples/opd/train_opd.py

from ludic.training.algorithm import make_opd
from ludic.training.trainer import Trainer
from ludic.training.batching import RolloutBatchSource
from transformers import AutoModelForCausalLM

# Load models
student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
teacher.eval()  # Teacher in eval mode, no grad

# Create OPD algorithm
algo = make_opd(
    teacher_model=teacher,
    pad_token_id=tokenizer.pad_token_id,
    length_normalize=True,
)

# Standard training setup
batch_source = RolloutBatchSource(
    engine=rollout_engine,
    requests_fn=make_requests_fn(...),
    credit_assigner=algo.credit_assigner,
)

trainer = Trainer(
    model=student,
    algorithm=algo,
    batch_source=batch_source,
    config=TrainerConfig(...),
)

trainer.train(num_steps=1000)
```

---

## Extensions

### 1. Hybrid OPD + RL (MOPD-style)

```python
def make_mopd(
    teacher_model: nn.Module,
    pad_token_id: int,
    kl_weight: float = 1.0,
    reward_weight: float = 0.1,
) -> RLAlgorithm:
    """Mixed On-Policy Distillation: KL + environment reward."""
    return RLAlgorithm(
        name="mopd",
        credit_assigner=MonteCarloReturn(gamma=1.0),  # Use env rewards
        loss=HybridOPDLoss(kl_weight=kl_weight, reward_weight=reward_weight),
        preprocess=make_opd_preprocessor(teacher_model, pad_token_id),
    )
```

### 2. Async Teacher Inference

For large teachers, compute logprobs asynchronously:

```python
class AsyncTeacherPreprocessor:
    def __init__(self, teacher_client: ChatClient):
        self.teacher_client = teacher_client

    async def __call__(self, saw_batch: SAWBatch) -> SAWBatch:
        # Batch compute_logprobs calls
        tasks = [
            self.teacher_client.compute_logprobs_async(item)
            for item in saw_batch.items
        ]
        results = await asyncio.gather(*tasks)
        for item, logprobs in zip(saw_batch.items, results):
            item.meta["teacher_token_logprobs"] = logprobs
        return saw_batch
```

### 3. Filtered OPD (only train on good trajectories)

```python
def make_filtered_opd(
    teacher_model: nn.Module,
    pad_token_id: int,
    reward_threshold: float = 0.5,
) -> RLAlgorithm:
    """OPD only on trajectories with positive reward."""
    from ludic.training.filters import combine, drop_below_reward

    return RLAlgorithm(
        name="filtered_opd",
        credit_assigner=ConstantCredit(1.0),
        loss=ReverseKLLoss(),
        preprocess=make_opd_preprocessor(teacher_model, pad_token_id),
        sample_filter=drop_below_reward(reward_threshold),
    )
```

---

## Testing Checklist

- [ ] Unit test `ReverseKLLoss` with mock batch
- [ ] Unit test `_compute_teacher_logprobs` with small model
- [ ] Integration test: student logprobs match teacher on identical models
- [ ] Integration test: loss decreases over training steps
- [ ] E2E test: train on GSM8K with OPD, measure accuracy improvement
