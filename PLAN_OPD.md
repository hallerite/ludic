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

**vLLM 0.11.0 specifics (THIS IS THE IMPORTANT PART)**

We’ll treat the teacher as a **remote vLLM 0.11.0 OpenAI-compatible server** and use a true “prefill-only logprobs” call:

- Endpoint: `POST /v1/completions`
- Key params (teacher-side):
  - `max_tokens: 0`  → *prefill only; no generation*
  - `prompt_logprobs: K` → return per-prompt-token logprobs (K=0 if you only want the chosen token logprob)
  - `return_token_ids: true` → avoid retokenization drift (token IDs returned by server)
  - `echo: true` → ensure returned token arrays include the prompt
  - `stream: false` → prompt_logprobs is not available for streaming anyway

**Critical operational requirement** for prompt logprobs:
- Start the teacher server with prefix caching disabled, e.g. `--no-enable-prefix-caching` (otherwise prompt_logprobs won’t be supported / won’t be reliable for full-prompt logprobs depending on config).

**Teacher response semantics you should expect**
- Prompt token logprobs are returned as *prompt-side* fields (vLLM’s `prompt_logprobs`), and token IDs are included when `return_token_ids=true`.
- First prompt token logprob can be `None` / missing depending on how the server represents “no previous context” (so align shifts carefully).

---

**Changes**:

1. **New preprocess function** in `algorithm.py`:
   ```python
   def _compute_teacher_logprobs_vllm(
       saw_batch: SAWBatch,
       *,
       teacher_base_url: str,
       teacher_model_name: str,
       api_key: str | None,
       prompt_logprobs_k: int = 0,
       timeout_s: float = 60.0,
   ) -> SAWBatch:
       """
       Compute teacher *prompt* token logprobs via vLLM 0.11.0 OpenAI-compatible server.

       Uses /v1/completions with max_tokens=0 (prefill-only) so the teacher
       does exactly: forward pass over provided tokens -> prompt logprobs.

       Stores:
         - item.meta["teacher_prompt_token_ids"]       : list[int]
         - item.meta["teacher_prompt_token_logprobs"]  : list[float|None]
           (aligned to prompt_token_ids; handle first token / shift carefully)
       """
       import requests

       headers = {"Content-Type": "application/json"}
       if api_key:
           headers["Authorization"] = f"Bearer {api_key}"

       for item in saw_batch.items:
           # NOTE: item.input_ids may include both prompt + completion depending on your pipeline.
           # For OPD you typically want teacher logprobs on the *student trajectory tokens* too.
           # That means: send the full prefix+sampled-tokens sequence as "prompt".
           #
           # If your item stores raw text instead of ids, send "prompt": <string>.
           # If you store ids and want to avoid re-tokenization, prefer:
           #   - configure teacher server with --return-tokens-as-token-ids (server flag)
           #   - still send text, and read teacher-returned token_ids
           #
           # If you already have the exact tokens as IDs and need "no retokenization, ever",
           # you’ll want a custom endpoint that accepts token IDs directly; OpenAI compat
           # endpoints are text-first.
           payload = {
               "model": teacher_model_name,
               "prompt": item.prompt_text,  # <- wire this to your actual stored prompt/trajectory text
               "max_tokens": 0,             # <- vLLM 0.11.0: prefill-only
               "temperature": 0.0,
               "echo": True,
               "stream": False,
               "prompt_logprobs": prompt_logprobs_k,
               "logprobs": 0,
               "return_token_ids": True,
           }

           r = requests.post(
               f"{teacher_base_url}/v1/completions",
               headers=headers,
               json=payload,
               timeout=timeout_s,
           )
           r.raise_for_status()
           resp = r.json()

           choice = resp["choices"][0]

           # vLLM may include prompt token IDs when return_token_ids=true.
           # Field names can differ between completions vs chat; keep it simple:
           teacher_prompt_token_ids = choice.get("prompt_token_ids") or choice.get("token_ids")
           teacher_prompt_logprobs = choice.get("prompt_logprobs") or choice.get("logprobs", {}).get("token_logprobs")

           item.meta["teacher_prompt_token_ids"] = teacher_prompt_token_ids
           item.meta["teacher_prompt_token_logprobs"] = teacher_prompt_logprobs

       return saw_batch
````

2. **Extend collation** in `trainer.py`:

   ```python
   # In _collate_saw_items, after old_logp_action handling:

   if items and "teacher_prompt_token_logprobs" in items[0].meta:
       # Pad to max_len (or max_len-1 depending on your shift convention).
       # Pick ONE convention and stick to it across student/teacher:
       #   - Either store "logprob of token t given <t>" aligned to input_ids[t]
       #   - Or store shifted alignment to targets input_ids[t+1]
       teacher_lp_list = []
       teacher_tid_list = []
       for item in items:
           lp = item.meta.get("teacher_prompt_token_logprobs", [])
           tid = item.meta.get("teacher_prompt_token_ids", [])

           teacher_lp_list.append(lp)
           teacher_tid_list.append(tid)

       # You likely already have a pad helper; the point is just "make it [B,T]"
       batch["teacher_token_logprobs"] = padded_float_tensor(teacher_lp_list)  # [B, T]
       batch["teacher_token_ids"] = padded_long_tensor(teacher_tid_list)       # [B, T]
   ```

3. **New loss** in `loss.py`:

   ```python
   @dataclass
   class ReverseKLLoss(Loss):
       """Per-token reverse KL distillation loss."""

       def compute(self, logits: Tensor, batch: Batch) -> Tuple[Tensor, Dict]:
           # Student log probs from current forward pass
           student_logp = selective_log_softmax(logits, batch["input_ids"])

           # Teacher log probs from preprocess (vLLM prefill-only)
           teacher_logp = batch["teacher_token_logprobs"]

           # IMPORTANT: Align masking + shifts consistently.
           # If teacher_logp has None at the first position, mask it out.
           action_mask = batch["action_mask"]
           valid_mask = action_mask & teacher_logp.isfinite()  # or explicit None-handling if stored as float32 with sentinel

           reverse_kl = student_logp - teacher_logp  # minimize this
           loss = (reverse_kl * valid_mask).sum() / valid_mask.sum().clamp(min=1)

           return loss, {"reverse_kl": loss.item()}
   ```

4. **Algorithm preset** in `algorithm.py`:

   ```python
   def make_opd(
       teacher_base_url: str,
       teacher_model_name: str,
       teacher_api_key: str | None,
       *,
       prompt_logprobs_k: int = 0,
       credit_assigner: Optional[CreditAssigner] = None,  # default: ConstantCredit
   ) -> RLAlgorithm:
       credit = credit_assigner or ConstantCredit(1.0)
       loss = ReverseKLLoss()
       preprocess = partial(
           _compute_teacher_logprobs_vllm,
           teacher_base_url=teacher_base_url,
           teacher_model_name=teacher_model_name,
           api_key=teacher_api_key,
           prompt_logprobs_k=prompt_logprobs_k,
       )
       return RLAlgorithm("opd", credit, loss, preprocess)
   ```

**Pros**:

* Follows existing patterns (`_ensure_old_token_logprobs`)
* Clean separation: preprocess computes, loss consumes
* Teacher model can be swapped easily (just point to another vLLM server)
* Credit assigner still usable for hybrid OPD+RL

**Cons**:

* Teacher forward pass in preprocess adds network+compute overhead
* You must manage exact alignment / shifts across student vs teacher logprobs
* OpenAI-compat endpoints are text-first; token-id exactness depends on server settings + returned token IDs

---

### Option B: Self-Contained Loss with Teacher

**Philosophy**: Loss owns the teacher model and computes everything internally.

**Update for vLLM 0.11.0**: you can still do this, but replace `teacher_model(...)` with a **teacher HTTP call** (same payload as Option A). This is almost always worse than Option A because you’re now mixing “RPC + inference” inside the backward pass.

**Changes** (only the teacher part differs):

* In `compute(...)`, do `requests.post(/v1/completions, max_tokens=0, prompt_logprobs=K, ...)` under `torch.no_grad()` (conceptually).
* Everything else stays the same.

---

### Option C: Teacher Logprobs at Rollout Time

**Philosophy**: Compute teacher logprobs alongside student during rollout generation.

**Update for vLLM 0.11.0**:

* Call the teacher’s `/v1/completions` with:

  * `prompt = (prompt_prefix + student_generated_so_far)` (full trajectory prefix)
  * `max_tokens = 0`
  * `prompt_logprobs = 0 or K`
  * `return_token_ids = true`
* Extract the teacher logprob for the *newly appended token* (or segment) and store it per-step.
* This lets you compute `reverse_kl_t` online.

**Pros**:

* Teacher computation can overlap with env / rollout plumbing
* You can early-filter bad rollouts cheaply (teacher KL too high, etc.)

**Cons**:

* Many small teacher calls unless you batch aggressively
* You have to be careful about “prompt_logprobs not available when stream=True”

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

In Ludic, the canonical placement for teacher calls is **upstream of the Trainer**:

1. Annotate each `SAWItem` with `meta["teacher_token_logprobs"]` (one float per action token).
   - Synchronous training: wrap your batch source with `TeacherAnnotatedBatchSource`.
   - Pipeline RL: attach teacher logprobs in the actor before pushing to Redis.
2. Use `make_opd()` which computes OPD weights from existing metadata and **never calls the teacher**.

**Total**: ~120 lines of library code + example.

---

## File-by-File Changes

### `src/ludic/training/loss.py`

Implemented token-level OPD as `OnPolicyDistillationLoss`:

- Advantage per action token: `A_t = logp_teacher_t - logp_old_student_t`
- Loss: `-mean_t( A_t.detach() * logp_student_current_t )` over action tokens (discount=0)
- Expects `batch["teacher_token_logprobs"]` and `batch["old_token_logprobs"]` as `[B, T-1]` tensors aligned to `targets=input_ids[:, 1:]` (zeros on non-action positions).

### `src/ludic/training/algorithm.py`

Implemented `make_opd()` to:

- Never call the teacher (teacher scoring is upstream by design).
- Install a preprocess that ensures `meta["old_token_logprobs"]` exists (copied from rollout-time `completion_logprobs` when available; async/pipeline batches must ship them) and validates that `meta["teacher_token_logprobs"]` exists and has the correct length.
- Use `OnPolicyDistillationLoss` for the actual reverse-KL OPD update.

### Teacher Scoring (Upstream, Required)

Teacher scoring is intentionally kept out of the Trainer/algorithm:

- `TeacherLogprobScorer` / `AsyncTeacherLogprobScorer`: `src/ludic/training/teacher.py`
- `TeacherAnnotatedBatchSource` / `annotate_teacher_logprobs`: `src/ludic/training/batching/teacher_annotated.py`
- Pipeline RL: `run_pipeline_actor(..., teacher_scorer=...)` annotates before pushing to Redis.

### `src/ludic/training/trainer.py`

`_collate_saw_items()` collates per-token logprobs for token-level objectives like OPD:

- `batch["teacher_token_logprobs"]`: `[B, T-1]` aligned to `targets=input_ids[:, 1:]`
- `batch["old_token_logprobs"]`: `[B, T-1]` aligned to `targets=input_ids[:, 1:]`

---

## Usage Example

```python
# examples/opd/train_opd.py

from ludic.training.algorithm import make_opd
from ludic.training.trainer import Trainer
from ludic.training.batching import RolloutBatchSource

# Teacher is remote: vLLM 0.11.0 OpenAI-compatible server
TEACHER_BASE_URL = "http://teacher-host:8000"
TEACHER_MODEL_NAME = "teacher-model"
TEACHER_API_KEY = None  # or "token-..."

# Create OPD algorithm
algo = make_opd(
    teacher_base_url=TEACHER_BASE_URL,
    teacher_model_name=TEACHER_MODEL_NAME,
    teacher_api_key=TEACHER_API_KEY,
    prompt_logprobs_k=0,       # 0 = chosen-token logprob only (cheapest)
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
    teacher_base_url: str,
    teacher_model_name: str,
    teacher_api_key: str | None,
    pad_token_id: int,
    kl_weight: float = 1.0,
    reward_weight: float = 0.1,
) -> RLAlgorithm:
    """Mixed On-Policy Distillation: KL + environment reward."""
    return RLAlgorithm(
        name="mopd",
        credit_assigner=MonteCarloReturn(gamma=1.0),  # Use env rewards
        loss=HybridOPDLoss(kl_weight=kl_weight, reward_weight=reward_weight),
        preprocess=make_opd_preprocessor(teacher_base_url, teacher_model_name, teacher_api_key),
    )
```

### 2. Async Teacher Inference

For large teachers, compute logprobs asynchronously (still using `/v1/completions`, `max_tokens=0`):

```python
class AsyncTeacherPreprocessor:
    def __init__(self, teacher_base_url: str, teacher_model_name: str, api_key: str | None):
        self.teacher_base_url = teacher_base_url
        self.teacher_model_name = teacher_model_name
        self.api_key = api_key

    async def __call__(self, saw_batch: SAWBatch) -> SAWBatch:
        import aiohttp
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with aiohttp.ClientSession(headers=headers) as session:
            async def one(item):
                payload = {
                    "model": self.teacher_model_name,
                    "prompt": item.prompt_text,
                    "max_tokens": 0,
                    "temperature": 0.0,
                    "echo": True,
                    "stream": False,
                    "prompt_logprobs": 0,
                    "logprobs": 0,
                    "return_token_ids": True,
                }
                async with session.post(f"{self.teacher_base_url}/v1/completions", json=payload) as r:
                    r.raise_for_status()
                    resp = await r.json()
                    choice = resp["choices"][0]
                    item.meta["teacher_token_ids"] = choice.get("prompt_token_ids") or choice.get("token_ids") or []
                    lp = choice.get("prompt_logprobs") or choice.get("logprobs", {}).get("token_logprobs")
                    item.meta["teacher_token_logprobs"] = [(float("nan") if x is None else x) for x in lp]
                return item

            await asyncio.gather(*(one(item) for item in saw_batch.items))

        return saw_batch
```

### 3. Filtered OPD (only train on good trajectories)

```python
def make_filtered_opd(
    teacher_base_url: str,
    teacher_model_name: str,
    teacher_api_key: str | None,
    reward_threshold: float = 0.5,
) -> RLAlgorithm:
    """OPD only on trajectories with positive reward."""
    from ludic.training.filters import combine, drop_below_reward

    return RLAlgorithm(
        name="filtered_opd",
        credit_assigner=ConstantCredit(1.0),
        loss=ReverseKLLoss(),
        preprocess=make_opd_preprocessor(teacher_base_url, teacher_model_name, teacher_api_key),
        sample_filter=drop_below_reward(reward_threshold),
    )
```

---

## Testing Checklist

* [ ] Unit test `ReverseKLLoss` with mock batch
* [ ] Unit test `_compute_teacher_logprobs_vllm` against a tiny teacher model served with vLLM 0.11.0
* [ ] Integration test: student logprobs match teacher on identical models
* [ ] Integration test: loss decreases over training steps
* [ ] E2E test: train on GSM8K with OPD, measure accuracy improvement

```

Sources (for the vLLM 0.11.0 wiring details): :contentReference[oaicite:0]{index=0}
::contentReference[oaicite:1]{index=1}
```
