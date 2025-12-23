import time
import torch
import logging
from typing import List, Optional, Mapping, Dict, Any
from ludic.distributed.interfaces import (
    ControlPlane,
    WeightMetadata,
    TensorCommunicator,
    PolicyPublisher,
)
from ludic.inference.vllm_client import VLLMChatClient
from ludic.distributed.publisher import BroadcastPolicyPublisher
from ludic.distributed.publisher import Rank0OnlyPublisher

# Setup logger
logger = logging.getLogger(__name__)


class VllmControlPlane(ControlPlane):
    def __init__(self, client: VLLMChatClient):
        self.client = client
        self.session = client._session
        self.url = client.server_url

    def announce_update_batch(
        self,
        metadata: List[WeightMetadata],
        version: Optional[int] = None,
    ) -> None:
        """
        Hits the /update_param_batch endpoint to prepare the server.
        """
        payload: Dict[str, Any] = {"metadata": metadata}
        if version is not None:
            payload["version"] = version

        # The server endpoint returns immediately after scheduling the task.
        # We ensure the server is listening via the HTTP response before broadcasting.
        resp = self.session.post(
            f"{self.url}/update_param_batch",
            json=payload,
            timeout=30.0,
        )
        resp.raise_for_status()

    def finalize_update(self, version: str | None = None) -> None:
        """
        Polls the server to ensure background weight application and
        cache resets are complete before allowing training to proceed.
        """
        while self.client.get_num_background_tasks() > 0:
            time.sleep(0.2)


class VllmTensorCommunicator(TensorCommunicator):
    def __init__(self, client: VLLMChatClient):
        if not client._pynccl_comm:
            raise RuntimeError("vLLM Client has no active NCCL communicator")
        self._comm = client._pynccl_comm
        self._rank = client._rank

    @property
    def rank(self) -> int:
        return self._rank

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        self._comm.broadcast(tensor, src=src)

    def barrier(self) -> None:
        self._comm.group.barrier()


def _transform_hf_to_vllm(
    state_dict: Mapping[str, torch.Tensor],
    *,
    debug: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Robustly normalizes HuggingFace/PEFT state dicts into a form that
    vLLM's own weight loader can consume.

    Rules:
    1. Filter out all `lora_` adapter keys.
    2. Strip the global `base_model.model.` prefix (iteratively).
    3. Strip the `.base_layer` artifact.
    4. Ensure `model.` prefix exists for layers if missing.
    5. Fix `model.model.*` → `model.*`.

    Important: we do NOT fuse Q/K/V or Gate/Up here.
    We leave them as HF-style keys (q_proj, k_proj, v_proj, gate_proj, up_proj)
    and let vLLM's AutoWeightsLoader handle stacking/fusion and sharding.
    """

    # --- DEBUG: Print Original Keys ---
    if debug:
        print("\n" + "=" * 80)
        print(
            f"🔍 [DEBUG] RAW STATE DICT KEYS (Before Transform) - Total: {len(state_dict)}"
        )
        print("=" * 80)
        for k in sorted(state_dict.keys()):
            shape_str = str(tuple(state_dict[k].shape))
            print(f"  • {k:<80} | {shape_str}")
        print("=" * 80 + "\n")
    # ----------------------------------

    new_state_dict: Dict[str, torch.Tensor] = {}

    # --- Step 1: Standardization & Cleanup ---
    for k, v in state_dict.items():
        # Filter out LoRA adapter weights, we only want the merged base weights
        if "lora_" in k or "lora." in k:
            continue

        # 1. Remove base_model.model wrapper (iteratively to be safe)
        clean_k = k
        while clean_k.startswith("base_model.model."):
            clean_k = clean_k[len("base_model.model.") :]

        # 2. Remove .base_layer artifact
        clean_k = clean_k.replace(".base_layer", "")

        # 3. Handle double model prefix edge case (model.model.layers -> model.layers)
        if clean_k.startswith("model.model."):
            clean_k = clean_k[len("model.") :]

        # 4. Ensure "layers." is prefixed with "model." (vLLM / HF convention for Qwen/Llama)
        if clean_k.startswith("layers."):
            clean_k = f"model.{clean_k}"

        new_state_dict[clean_k] = v

    # --- DEBUG: Print Transformed Keys (HF-normalized, not fused) ---
    if debug:
        print("\n" + "=" * 80)
        print(
            f"🚀 [DEBUG] NORMALIZED HF KEYS (Ready for vLLM AutoWeightsLoader) - Total: {len(new_state_dict)}"
        )
        print("=" * 80)
        for k in sorted(new_state_dict.keys()):
            shape_str = str(tuple(new_state_dict[k].shape))
            print(f"  • {k:<80} | {shape_str}")
        print("=" * 80 + "\n")
    # -------------------------------------

    return new_state_dict


class VllmPublisherAdapter(PolicyPublisher):
    """
    Wraps a generic broadcaster to perform vLLM-specific tensor cleanup on the fly.
    """
    def __init__(self, inner: BroadcastPolicyPublisher, *, debug: bool = False):
        self.inner = inner
        self._debug = debug

    def publish(
        self,
        state_dict: Mapping[str, torch.Tensor],
        version: Optional[int] = None,
    ) -> None:
        # Transform (Clean/Normalize) the weights before sending.
        # Fusion (QKV, Gate+Up) is left to vLLM's own loader.
        vllm_params = _transform_hf_to_vllm(state_dict, debug=self._debug)
        self.inner.publish(vllm_params, version=version)


def create_vllm_publisher(
    client: VLLMChatClient,
    *,
    debug: bool = False,
    rank0_only: bool = False,
) -> PolicyPublisher:
    """
    Create a PolicyPublisher that pushes weights into a vLLM runtime.

    If rank0_only=True, returns a Rank0OnlyPublisher wrapper that constructs the
    underlying publisher lazily and only publishes on distributed rank 0.
    """

    def _make() -> PolicyPublisher:
        control = VllmControlPlane(client)
        comm = VllmTensorCommunicator(client)
        broadcaster = BroadcastPolicyPublisher(control, comm, src_rank=comm.rank)
        return VllmPublisherAdapter(broadcaster, debug=debug)

    if rank0_only:
        return Rank0OnlyPublisher(_make)
    return _make()
