# Plan: On-Policy Distillation for Ludic

## Overview

On-Policy Distillation (OPD) combines on-policy sampling with dense teacher supervision.
The student samples trajectories, the teacher scores the sampled tokens, and we apply
policy-gradient updates using per-token reverse-KL advantages.

Core equation:

```
reverse_kl_t = log π_student(x_t | x_{<t}) - log π_teacher(x_t | x_{<t})
advantage_t  = -reverse_kl_t = log π_teacher - log π_student
loss = -E[advantage_t * log π_θ(x_t | x_{<t})]
```

## Design Contract (Current)

OPD follows the same attachment-first design as PPO-style behavior logps.

### Attachments (source of truth)

- `SAWItem.extras` includes `ActorTokenLogps`: behavior-policy per-token logprobs for the sampled action tokens.
- `SAWItem.extras` includes `TeacherTokenLogps`: teacher per-token logprobs for the same sampled action tokens.

Both extras are aligned to the completion (action) tokens only. There is **no**
metadata fallback or backfill in OPD.

### Collation

`_collate_saw_items` produces:

- `batch["actor_logps"]`: `[B, T]` logprobs aligned to `input_ids` with zeros outside the action region.
- `batch["teacher_logps"]`: `[B, T]` logprobs aligned to `input_ids` with zeros outside the action region.

The loss shifts these to `[:, 1:]` to align with next-token targets.

### Loss

`ReverseKLLoss` computes token-level advantages:

```
advantages = (teacher_logps_shifted - actor_logps_shifted) * opd_coef
```

and applies them over the action tokens indicated by `action_mask`.

## Where Teacher Logprobs Come From

Teacher scoring is done **upstream** of the Trainer:

- `TeacherAnnotatedBatchSource` wraps any `BatchSource` and populates
  `TeacherTokenLogps` in `SAWItem.extras` using a `TeacherLogprobScorer`.
- In pipeline RL, the actor process can annotate batches before pushing to Redis
  so the learner never calls the teacher.

The Trainer never calls the teacher.

## Usage Sketch

```python
from ludic.training import RolloutBatchSource, make_opd
from ludic.training.batching import TeacherAnnotatedBatchSource

algo = make_opd(opd_coef=1.0)

base = RolloutBatchSource(...)
annotated = TeacherAnnotatedBatchSource(
    base,
    teacher_scorer=my_teacher_scorer,
)

trainer = Trainer(model=model, algo=algo, batch_source=annotated, ...)
```

**Important**: the actor must return chosen-token logprobs at rollout time
(e.g. `ReturnSpec.for_rl()`), otherwise `actor_logps` will be missing.

## Non-Goals

- No backfilling of behavior logprobs in OPD.
- No metadata-based teacher logprobs.
- No teacher calls inside the Trainer.
