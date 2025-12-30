import asyncio
import shutil
from pathlib import Path
from typing import List, Optional

import tomli_w
from pydantic import Field

# --- Prime-RL Imports ---
from prime_rl.orchestrator.config import OrchestratorConfig as PrimeOrchestratorConfig
from prime_rl.transport import TrainingBatch, TrainingSample, setup_training_batch_sender
from prime_rl.utils.client import (
    setup_admin_clients,
    check_health,
    update_weights,
    reload_weights,
    init_nccl_broadcast,
)
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_rollout_dir,
    get_step_path,
    get_latest_ckpt_step,
)
from prime_rl.utils.logger import setup_logger

# --- Ludic Imports ---
from ludic.training.types import SAWBatch, BatchSource


# ---------------------------------------------------------------------------
# Configuration bridge
# ---------------------------------------------------------------------------

class LudicConfig(PrimeOrchestratorConfig):
    """
    Extends Prime-RL's OrchestratorConfig to include Ludic-specific settings.
    """
    # Ludic specific registry keys (used by user code to build BatchSource)
    env_kind: str = Field(
        description="Key in the Ludic EnvRegistry", default="my_env"
    )
    protocol_kind: str = Field(
        description="Key in the Ludic ProtocolRegistry", default="single_agent"
    )

    # Ludic specific rollout settings (free-form kwargs for env/protocol)
    protocol_kwargs: dict = Field(default_factory=dict)
    env_kwargs: dict = Field(default_factory=dict)

    # Ludic specific rollout control
    max_steps_per_episode: int = Field(
        description="Max environment steps per Ludic rollout episode.",
        default=1,
    )
    rollout_concurrency: Optional[int] = Field(
        description="Concurrency for Ludic rollout execution (default min(128, batch_size)).",
        default=None,
    )
    rollout_timeout_s: Optional[float] = Field(
        description="Optional per-call timeout (seconds) for Ludic inference during rollouts.",
        default=None,
    )


# ---------------------------------------------------------------------------
# Training batch sender
# ---------------------------------------------------------------------------

class PrimeRLBatchSender:
    """
    Converts Ludic SAWBatches (in-memory) into Prime-RL TrainingBatch payloads.
    """

    def __init__(
        self,
        output_dir: Path,
        temperature: float,
        rollout_transport,
    ):
        self.temperature = temperature
        self.sender = setup_training_batch_sender(output_dir, rollout_transport)

    def send_step(self, saw_batch: SAWBatch, step: int):
        """
        Converts Ludic data -> Prime TrainingSamples -> TrainingBatch -> transport.
        """
        train_examples: List[TrainingSample] = []

        for item in saw_batch.items:
            # Split input_ids based on action_mask (0=prompt, 1=completion).
            prompt_ids: List[int] = []
            completion_ids: List[int] = []
            for token_id, mask in zip(item.input_ids, item.action_mask):
                if mask:
                    completion_ids.append(token_id)
                else:
                    prompt_ids.append(token_id)

            if not completion_ids:
                continue

            # Extract logprobs from attachments (preferred) with meta fallback.
            completion_logprobs = []
            if item.actor_logps is not None:
                completion_logprobs = list(item.actor_logps.token_logps)
            else:
                meta_logprobs = item.meta.get("completion_logprobs")
                if isinstance(meta_logprobs, list):
                    completion_logprobs = list(meta_logprobs)

            # Pad with 0.0 if missing to keep lengths aligned.
            if len(completion_logprobs) < len(completion_ids):
                completion_logprobs += [0.0] * (len(completion_ids) - len(completion_logprobs))

            train_examples.append(
                TrainingSample(
                    prompt_ids=prompt_ids,
                    prompt_mask=[False] * len(prompt_ids),
                    completion_ids=completion_ids,
                    completion_mask=[True] * len(completion_ids),
                    completion_logprobs=completion_logprobs[: len(completion_ids)],
                    # In Ludic, SAWItem.weight is already the scalar advantage
                    advantage=float(item.weight),
                )
            )

        training_batch = TrainingBatch(
            examples=train_examples,
            temperature=self.temperature,
            step=step,
        )
        self.sender.send(training_batch)

    def close(self) -> None:
        self.sender.close()


# ---------------------------------------------------------------------------
# Ludic–Prime orchestrator bridge
# ---------------------------------------------------------------------------

class PrimeOrchestrator:
    """
    Thin bridge:

    - pulls SAWBatches from a user-provided BatchSource
    - handles Prime-RL async weight sync
    - converts & sends TrainingBatch for the Prime-RL packer/trainer
    """

    def __init__(self, config: LudicConfig, batch_source: BatchSource):
        self.config = config
        self.logger = setup_logger(config.log.level)

        # 0. Require a BatchSource
        if batch_source is None:
            raise ValueError("PrimeOrchestrator requires a BatchSource instance.")
        self.batch_source = batch_source

        self._write_orchestrator_config()

        # 1. Setup admin clients for health checks + weight updates
        self.admin_clients = setup_admin_clients(config.client)

        # 2. Setup TrainingBatch sender (Prime transport)
        self.sink = PrimeRLBatchSender(
            output_dir=config.output_dir,
            temperature=config.sampling.temperature,
            rollout_transport=config.rollout_transport,
        )

        # 3. State
        self.step = 0
        self.ckpt_step = 0

    def _write_orchestrator_config(self) -> None:
      config_dir = self.config.output_dir / "configs"
      config_dir.mkdir(parents=True, exist_ok=True)

      data = self.config.model_dump(exclude_none=True, mode="json")
      allowed = set(PrimeOrchestratorConfig.model_fields)
      data = {k: v for k, v in data.items() if k in allowed}

      with open(config_dir / "orch.toml", "wb") as f:
          tomli_w.dump(data, f)

    async def setup(self):
        """Initialize infrastructure connectivity."""
        self.logger.info("Checking Inference Health...")
        await check_health(self.admin_clients)

        # Initialize NCCL broadcast if configured
        if self.config.weight_broadcast.type == "nccl":
            await init_nccl_broadcast(
                self.admin_clients,
                self.config.weight_broadcast.host,
                self.config.weight_broadcast.port,
                self.config.weight_broadcast.timeout,
            )

        # Reset to base model
        self.logger.info("Try resetting to base model...")
        try:
            await reload_weights(self.admin_clients)
        except Exception as e:
            self.logger.warning(f"Skipping reload_weights: {e}")

        # Clean rollout directories at the beginning (filesystem transport only)
        if (
            self.step == 0
            and getattr(self.config.rollout_transport, "type", "filesystem") == "filesystem"
        ):
            shutil.rmtree(get_rollout_dir(self.config.output_dir), ignore_errors=True)

    async def sync_policy(self):
        """
        Check for new weights from the Trainer and update Inference.

        Semantics mirror prime_rl.orchestrator.scheduler.update_policy:
        - Keep async_level = step - ckpt_step <= max_async_level
        - If strict_async_level:
            always use policy at (step - max_async_level) and wait if needed
        - Else:
            use the newest checkpoint, but never violate max_async_level
        """
        broadcast_dir = get_broadcast_dir(self.config.output_dir)

        # Latest checkpoint step that is actually present on disk
        latest_ckpt_step = get_latest_ckpt_step(broadcast_dir) or 0

        # Minimum checkpoint we are allowed to be on (enforce async bound)
        async_away_ckpt_step = max(self.step - self.config.max_async_level, 0)

        if self.config.strict_async_level:
            # Always lag exactly max_async_level behind (or 0 at the beginning)
            target_step = async_away_ckpt_step
        else:
            # Use the latest available checkpoint, but never violate async bound
            target_step = max(async_away_ckpt_step, latest_ckpt_step)

        if target_step <= self.ckpt_step:
            # Already at or ahead of the desired policy, nothing to do
            return

        # If we are forcing an async barrier, log explicitly
        if target_step == async_away_ckpt_step:
            self.logger.info(
                f"Hit async barrier: step={self.step}, "
                f"ckpt_step={self.ckpt_step}, "
                f"max_async_level={self.config.max_async_level}. "
                f"Waiting for checkpoint {target_step}."
            )

        # Wait for STABLE flag to appear for the target checkpoint
        step_dir = get_step_path(broadcast_dir, target_step)
        stable_path = step_dir / "STABLE"
        while not stable_path.exists():
            await asyncio.sleep(0.1)

        self.logger.info(f"Updating inference weights to step {target_step}")

        await update_weights(
            self.admin_clients,
            step_dir,
            lora_name=self.config.lora_name,
        )

        self.ckpt_step = target_step

    async def run_step(self):
        """Generate one batch of data using Ludic via BatchSource."""
        self.logger.info(f"Starting Step {self.step}")

        # 1. Sync policy with trainer checkpoints
        await self.sync_policy()

        # 2. Fetch next batch from the BatchSource
        saw_batch = await self.batch_source.next_batch()

        self.logger.info(
            f"Step {self.step}: Generated {len(saw_batch.items)} items. "
            f"Avg Reward: {saw_batch.meta.get('avg_total_reward', 0.0):.2f}"
        )

        # 3. Send TrainingBatch for Trainer packer

        weights = [item.weight for item in saw_batch.items]
        self.logger.info(
            f"weight min/mean/max: {min(weights):.3f}/"
            f"{sum(weights)/len(weights):.3f}/"
            f"{max(weights):.3f}"
        )
        self.logger.info(f"group_id sample: {saw_batch.items[0].meta.get('group_id')}")
        self.logger.info(f"reward sample: {saw_batch.items[0].meta.get('total_reward')}")
        self.sink.send_step(saw_batch, self.step)

        self.step += 1

    async def loop(self):
        await self.setup()
        try:
            while self.step < (self.config.max_steps or float("inf")):
                await self.run_step()
        finally:
            self.sink.close()
