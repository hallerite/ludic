# Signal Composition in Ludic

This document describes how different training signals (rewards, advantages, losses) can be composed in Ludic's RL training pipeline.

## The Training Pipeline

```
Environment ──► Rewards ──► CreditAssigner ──► Advantages ──► Loss
     │                                              │
     ▼                                              ▼
  Scorers                                    CreditModifiers
  (Level 1)                                    (Level 2)
```

There are **three composition levels**:

| Level | Name | Where | Implementation |
|-------|------|-------|----------------|
| **1** | Reward | Before credit assignment | Agent scorers |
| **2** | Advantage | After credit assignment, before loss | CreditModifier |
| **3** | Loss | Separate loss terms | CompositeLoss |

## Level 1: Reward Composition

Add signals to rewards via Agent scorers, before credit assignment.

```python
# Scorers attached to Agent add to per-step rewards
agent = Agent(
    client=client,
    scorers=[intrinsic_reward_scorer],  # adds to step rewards
)
```

**Properties:**
- All signals go through the same credit assignment
- Signals interact (e.g., group normalization affects combined rewards)
- Tightest coupling between signals

**Use when:**
- Intrinsic rewards should be treated identically to environment rewards
- You want signals to interact during advantage estimation

## Level 2: Advantage Modification

Modify advantages after credit assignment, before loss.

```python
# KL penalty added to advantages
kl_penalty = -kl_coeff * (actor_logps - teacher_logps)
advantages = task_advantages + kl_penalty
# Then normal policy gradient with combined advantages
```

**Properties:**
- Each signal can have its own credit assignment strategy
- All signals go through the same importance ratio
- All signals go through the same loss function

**Use when:**
- Different signals need different credit assignment (e.g., sparse task rewards vs dense KL)
- You want all signals to go through importance sampling together

**Implementation in Ludic:**

Use `CreditModifier` to add per-token signals to advantages:

```python
algo = RLAlgorithm(
    credit_assigner=GroupNormalizedReturn(group_size=8),
    credit_modifiers=[KLCreditModifier(coeff=1.0)],
    loss=ClippedSurrogateLoss(...),
)
```

Or use the preset:

```python
algo = make_gspo_opd(group_size=8, kl_coeff=1.0)
```

## Level 3: Loss Composition

Combine independent loss terms additively.

```python
loss = CompositeLoss(terms=[
    LossTerm(name="rl", loss=ClippedSurrogateLoss(...), weight=1.0),
    LossTerm(name="auxiliary", loss=SomeAuxiliaryLoss(...), weight=0.1),
])
```

**Properties:**
- Each loss computed independently
- Different losses can use different data (current vs old policy logprobs)
- Loosest coupling
- Most flexible but signals don't interact

**Use when:**
- Truly independent objectives (e.g., RL + language modeling auxiliary)
- Different losses need fundamentally different handling
- You need maximum flexibility

## Key Differences: Advantage vs Loss Composition

| Aspect | Advantage Modification (Level 2) | Loss Composition (Level 3) |
|--------|----------------------------------|---------------------------|
| **KL source** | Old policy (rollout time) | Current policy (forward pass) |
| **Importance sampling** | Goes through ratio | Doesn't go through ratio |
| **Gradient** | `ratio * (task_adv + kl_penalty)` | `ratio * task_adv + kl_grad` |
| **Interaction** | Signals combined | Signals independent |

### Mathematical Difference

**Advantage Modification:**
```
A_t = task_advantage - kl_coeff * KL_old_t
L = E[ ratio_t * A_t ]
∇L ∝ ratio_t * A_t * ∇log π_t
```

**Loss Composition:**
```
L = E[ ratio_t * task_advantage ] + kl_coeff * E[ KL_current_t ]
∇L ∝ ratio_t * task_advantage * ∇log π_t + kl_coeff * ∇log π_t
```

In synchronous RL with single gradient step, `ratio ≈ 1` and `KL_old ≈ KL_current`, so these are similar. They diverge with:
- Multiple epochs per batch (PPO-style)
- Async RL with stale rollouts
- Large policy updates

## Recommended Patterns

### Pattern 1: Pure RL (task rewards only)
```python
algo = make_gspo(group_size=8)
```

### Pattern 2: GSPO + OPD hybrid (recommended for distillation)
```python
algo = make_gspo_opd(group_size=8, kl_coeff=1.0)
```

### Pattern 3: Independent auxiliary loss
```python
algo = RLAlgorithm(
    credit_assigner=GroupNormalizedReturn(group_size=8),
    loss=CompositeLoss(terms=[
        LossTerm(name="rl", loss=ClippedSurrogateLoss(...), weight=1.0),
        LossTerm(name="lm", loss=LanguageModelingLoss(...), weight=0.1),
    ]),
)
```

## Summary

| Scenario | Level | Implementation |
|----------|-------|----------------|
| Pure RL | - | `make_gspo()` |
| RL + teacher distillation | 2 (Advantage) | `make_gspo_opd()` |
| RL + unrelated auxiliary | 3 (Loss) | `CompositeLoss` |
| Intrinsic rewards | 1 (Reward) | Agent scorers |
