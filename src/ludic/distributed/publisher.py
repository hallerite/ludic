import torch
from typing import Mapping, List, Optional
from collections.abc import Callable
from ludic.distributed.interfaces import (
    PolicyPublisher, 
    ControlPlane, 
    TensorCommunicator, 
    WeightMetadata
)

class BroadcastPolicyPublisher(PolicyPublisher):
    """
    Publishes weights by:
    1. Sending a metadata manifest via the Control Plane (HTTP).
    2. Streaming the tensors via the Data Plane (NCCL).
    """
    def __init__(
        self, 
        control: ControlPlane, 
        comm: TensorCommunicator,
        src_rank: int
    ):
        self.control = control
        self.comm = comm
        self.src_rank = src_rank

    def publish(self, state_dict: Mapping[str, torch.Tensor], version: Optional[int] = None) -> None:
        # 1. Prepare Metadata
        metadata: List[WeightMetadata] = []
        # We need a stable order for the loop
        sorted_keys = sorted(state_dict.keys())
        
        for name in sorted_keys:
            tensor = state_dict[name]
            metadata.append({
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": tuple(tensor.shape)
            })

        # 2. Control Plane: Announce the Batch
        # This tells the server: "Allocate these tensors and get ready to receive"
        # We pass the version here so the server knows what "time" it is.
        self.control.announce_update_batch(metadata, version=version)

        # 3. Data Plane: Stream Tensors
        # The server is now in a loop waiting for broadcast calls matching the metadata list
        for name in sorted_keys:
            tensor = state_dict[name]
            # Ensure tensor is contiguous and on the right device before sending
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            self.comm.broadcast(tensor, src=self.src_rank)

        # 4. Synchronization
        self.comm.barrier()
        
        # 5. Finalize (e.g., reset prefix cache, bump version)
        self.control.finalize_update()


class Rank0OnlyPublisher(PolicyPublisher):
    """
    PolicyPublisher wrapper that only publishes on rank 0.

    - If torch.distributed is not initialized, it behaves like rank 0.
    - Inner publisher is constructed lazily to avoid rank>0 side effects
      (e.g., NCCL init, network connections).

    Why this exists:
      In multi-process training (DDP/FSDP2), *every rank* constructs a Trainer.
      But "publish weights to a serving runtime" is a singleton side-effect:
      you typically want only rank 0 to talk to the runtime (HTTP/NCCL).

      This wrapper lets you pass a publisher object on all ranks (so training
      code doesn't need per-rank conditionals), while ensuring:
        - only rank 0 actually calls the inner publisher
        - non-rank0 ranks never even construct the inner publisher (important
          when publisher construction requires resources that only rank 0 has,
          e.g. a vLLM weight-update client with an initialized NCCL communicator)
    """

    def __init__(
        self,
        make_inner: Callable[[], PolicyPublisher],
        *,
        enabled: bool = True,
    ) -> None:
        self._make_inner = make_inner
        self._enabled = enabled
        self._inner: PolicyPublisher | None = None

    def _is_rank0(self) -> bool:
        try:
            import torch.distributed as dist
        except Exception:
            return True
        if not (dist.is_available() and dist.is_initialized()):
            return True
        return dist.get_rank() == 0

    def _get_inner(self) -> PolicyPublisher:
        if self._inner is None:
            self._inner = self._make_inner()
        return self._inner

    def publish(self, state_dict: Mapping[str, torch.Tensor], version: Optional[int] = None) -> None:
        if not self._enabled:
            return
        if not self._is_rank0():
            return
        self._get_inner().publish(state_dict, version=version)
