# Ludic - an LLM-RL library for the era of experience

This repo is the result of an ongoing frustration I have had for months with the state of LLM-RL. It is essentially what I wanted during a research project earlier this year, but did not have – a codebase to post-train LLMs with RL built principally with **agentic behavior** in mind, not single-step LLM reasoning. While there are, as of Dec. 2025, many other great codebases for this purpose, I still wanted to share what I have worked on in the last few months, because I believe it to be conceptually quite unique.

This year, I spent a lot of time thinking about LLM-RL library design, and how the architecture would have to look like to prototype new ideas quickly and iterate on research. 

After my latest research project concluded mid-November and without much to do, I decided to rewrite my entire codebase from scratch and make it public.

The codebase follows some design decisions that will be very familiar to those who have already worked with other LLM-RL codebases, but radically departs from other frameworks in other aspects.

Let me convince you.

### Design Decisions

- It's a **library** – not a *framework*. You can rip out parts and swap in your own as you please – everything is loosely coupled.
- There is a clear separation between **Agent** and **Environment**
	- **Environments** are pure state-transition functions which may also emit rewards on state-stransition; they are also multi-agent by design.
	- **Agents** are LLMs with state, placed into a harness that deals with context management, parsing LLM outputs into the format the environment wants it, auxiliary tools & many other things
	- The **Interaction Protocol** explicitly defines the agent–env loop. This allows for reusing different agent harnesses with different envs and vice versa. An `InteractionProtocol` has a method named `run()` that returns a list of **Rollouts** – one for each agent perspective.
- The **Trainer** asks a **batch source** for the next batch by calling **batch_source.get_next_batch()**. The batch source then yields a batch of **State-Action-Weight** pairs. These are the biggest unit of data that the trainer is concerned with – they are essentially the experience that the agent gathered: the *state*, in which the agent took some *action* and the *weight* we want to attach to this action. This weight can be either the ground-truth reward or the rollout return or the group-relative advantage, among many other things. The trainer thus does not know about the concept of an episode or a rollout.
	- The batch source can be a dataset for offline RL (like SFT) or, say, a **Rollout Engine** that is connected to a live **vLLM** server for online RL.
- An **RL algorithm** is then just a strategy that neatly plugs into the trainer. It defines a **Credit Assigner** and a **Loss** – these two concepts are essentially all you need to define (almost) any RL algorithm.
	- The credit assigner tells us how much credit one state-action (or prompt-completion in LLM lingo) pair receives. For GRPO, this is relative to other rollouts.
- There are also primitives for **Intra-Batch** control and **Inter-Batch** control
	- Intra-batch control is how the rollouts in a batch should relate to one another. For instance, GRPO is implemented as an intra-batch control mechanism that takes rollout requests and expands them such that the rollout engine receives each original rollout request `group_size`-many times.
	- Inter-batch control is for things like curriculum learning or other concepts from classical RL that describe a dependence between batches over the course of training.

### Caveat

The codebase is mostly aspirational. It is built for researchers who value clean abstractions, conceptual clarity and hackability. It is neither battle-tested, nor production-grade. It does not aim to replace any other framework. It is best thought of as an intellectual exercise or a form of artistic expression – similar to functional programming.

However, I have found it quite easy to extend this codebase to add new algorithms, agent harnesses or environments using Claude Code or Codex, and I find it a joy to work with. 

## Repository layout

If you want to find your way around quickly, there are two main things in here: the library itself, and a set of runnable scripts that show how to use it end-to-end.

- `src/ludic/`: the actual library. This is where the core abstractions live (agents + context/memory, env interfaces + built-in envs, interaction protocols, inference clients, training/batching, evaluation, and distributed weight pushing).
- `examples/`: runnable “glue” scripts that assemble the pieces into real training/eval runs. These are meant as starting points and reference implementations.
- `environments/`: small runnable environments and configs you can import from, or execute directly when you just want to play with an env in isolation.
- `data/`: small datasets and artifacts used by some examples.
- `tests/`: unit/integration tests (pytest markers include `integration` and `gpu`).
- `scripts/`: standalone utilities (e.g., `calibrate_micro_batch.py` for micro-batch sizing; see `scripts/README.md`).

If you care about truncation semantics (env time limits vs protocol cutoffs vs model finish reasons), read `CONSIDERATIONS.md`.

### Logging

- Training stats use canonical prefixes: `train/`, `eval/`, and `perf/` (e.g., `train/loss`, `eval/accuracy`, `perf/gpu_mem_alloc_mb`).
- `train/step` and `eval/step` are used by loggers to annotate panels and runs.
- For W&B logging, set `WANDB_PROJECT` to your preferred project name; if unset, it defaults to `Ludic`.

### Training Knobs (Macro vs Micro)

- `rollouts_per_update`: number of rollouts per trainer step (must be divisible by `group_size` for GRPO-style grouping).
- `max_seq_len`: maximum token length for any single sample; trainer raises if exceeded.
- `micro_token_budget`: max padded tokens per micro-batch; the collator buckets + splits a macro-batch to fit this budget.

### Examples at a glance

- Tic-Tac-Toe (`examples/tic_tac_toe/`): a small env that’s useful for iterating on the full stack without paying a huge sampling bill.
	- Online RL: `examples/tic_tac_toe/train_tic_tac_toe.py` does LoRA fine-tuning with GRPO-style group-relative credit assignment.
	- SFT: `examples/tic_tac_toe/sft_tic_tac_toe.py` is the “offline” counterpart; it’s useful if you want to bootstrap a policy (format-following, basic competence) before turning on online RL.
	- Data + eval: `examples/tic_tac_toe/generate_synth_data.py` and `examples/tic_tac_toe/eval_tic_tac_toe_vllm.py`.

- GSM8K (`examples/gsm8k/`): a more “standard” QA-shaped workload with training + evaluation scripts (`examples/gsm8k/train_gsm8k.py`, `examples/gsm8k/eval_gsm8k_vllm.py`).

- FSDP2 Math (`examples/fsdp2_training/`): a multi-GPU template showing FSDP2 wrapping, NCCL weight pushes to vLLM, and GRPO-style credit assignment (`examples/fsdp2_training/train_math_fsdp2.py`).

- Pipeline RL (`examples/pipeline_rl/`): an actor/learner split over Redis for async sampling (`examples/pipeline_rl/run_actor.py`, `examples/pipeline_rl/run_trainer.py`). This is still experimental.

- Rejection sampling (`examples/rejection_sampling.py`): generate rollouts, filter them, and write training-ready JSONL for offline training.

## Token-in Inference (Drift-Free)

Ludic applies chat templates locally and sends pre-tokenized prompts to the
vLLM `/v1/completions` endpoint. This keeps training aligned to the exact tokens
sampled by the model, eliminating drift between training and inference.

Key points:
- Agents require a `ChatTemplate` (see `ludic.inference.HFChatTemplate`).
- Provide a shared tokenizer per process to avoid duplicated init costs.
- Tool calling uses a text parser (e.g., `HermesToolParser`) to extract tool calls
  from raw completions.
- If you need explicit stop token IDs, set them via
  `VLLMExtensions.extra_body_overrides` (e.g., `{"stop_token_ids": [...]}`).

### Migration from chat completions API

The library previously used vLLM's `/v1/chat/completions` endpoint. The new
token-in API has breaking changes:

| Old API | New API |
|---------|---------|
| `ChatClient.complete(ChatCompletionRequest)` | `ChatClient.complete_tokens(TokenCompletionRequest)` |
| `ChatCompletionRequest(messages=...)` | `TokenCompletionRequest(prompt_token_ids=...)` |
| `Agent(client, model, ctx, parser)` | `Agent(client, model, ctx, parser, chat_template)` |
| `ToolRequest(tools, tool_choice="auto")` | `ToolRequest(tools)` - tool_choice removed |

The `chat_template` parameter is now **required** on all agents. Create one with:

```python
from transformers import AutoTokenizer
from ludic.inference import HFChatTemplate, HermesToolParser

tokenizer = AutoTokenizer.from_pretrained("your-model")
chat_template = HFChatTemplate(tokenizer)

# For tool-calling agents:
chat_template = HFChatTemplate(tokenizer, tool_parser=HermesToolParser())
```

## Requirements

- Python 3.12+

- PyTorch >= 2.8.0 with CUDA for training examples

- vLLM server exposing the OpenAI API; NCCL available if you want live weight pushes

- Redis for the pipeline RL actor/trainer example

## Installation

Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

For running example code use:

```bash
uv sync --extra examples
```

## TODO

### High Priority
- Add on-policy distillation
	- plus a script for the MOPD variant introduced in [MiMo-V2-Flash](https://github.com/XiaomiMiMo/MiMo-V2-Flash/blob/main/paper.pdf)
- Add a classic Gym-style registry for agent harnesses, environments, and interaction protocols (so they don’t have to be built on the fly).
   - this will also allow us to create general eval and training scripts instead of hand-crafted ones
- Use proper FSDP2 wrapping in the training scripts

### Medium Priority

- Add Single Stream Policy Optimization (as described in [this paper](https://arxiv.org/abs/2509.13232v2)).

- Implement the [findings](https://www.llmdata.com/blog/mismatch-praxis/) from the LLM Data co. regarding **importance sampling**:

### Environments & Agents
- Build agent harness & environment for Pokemon!
   - the agent harness and env are fused together in Claude Plays Pokemon: https://github.com/davidhershey/ClaudePlaysPokemonStarter.
   - we should disentangle them into a re-usable agent harness and the different Pokemon games as environments
- create [Rustorio](https://github.com/albertsgarde/rustorio) env
- add agent with context folding as described in [this paper](https://context-folding.github.io/)



### Low Priority
- Make `scripts/push_to_hub.py` work with `--revision` for uploading different checkpoints.
- Add a progress bar to eval.

### Package changes (very breaking, do only after everything else is done)
- Improve packaging:
	- `eval`, `training`, and `batch_gen` should be distinct modules.
