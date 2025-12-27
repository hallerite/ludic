from __future__ import annotations

from .client import ChatClient, VersionedClient
from .sampling import SamplingParams
from .request import TokenCompletionRequest, InferenceSpec, ReturnSpec, ToolRequest
from .extensions import BackendExtensions, VLLMExtensions
from .vllm_client import VLLMChatClient
from .vllm_utils import start_vllm_server, wait_for_vllm_health
from .chat_template import ChatTemplate, HFChatTemplate, TemplateResult
from .tool_parser import ToolParser, HermesToolParser, ToolParseResult

__all__ = [
    # Client protocols
    "ChatClient",
    "VersionedClient",
    # Sampling
    "SamplingParams",
    # Request types
    "ReturnSpec",
    "ToolRequest",
    "InferenceSpec",
    "TokenCompletionRequest",
    # Extensions
    "BackendExtensions",
    "VLLMExtensions",
    # vLLM client
    "VLLMChatClient",
    "start_vllm_server",
    "wait_for_vllm_health",
    # Token-in API
    "ChatTemplate",
    "HFChatTemplate",
    "TemplateResult",
    "ToolParser",
    "HermesToolParser",
    "ToolParseResult",
]
