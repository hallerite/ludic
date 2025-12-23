from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Optional, Tuple

import pytest
import requests
from requests import ConnectionError as RequestsConnectionError

from ludic.inference.vllm_client import VLLMChatClient

from ludic.context.full_dialog import FullDialog
from tests._mocks import MockEnv, MockAgent

# ---------------------------------------------------------------------------
# Server fixture: launch ludic.inference.vllm_server end-to-end
# ---------------------------------------------------------------------------


class VLLMServerHandle:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        cmd: list[str],
        env: dict[str, str],
        dtype: str,
    ) -> None:
        self.host = host
        self.port = port
        self.cmd = cmd
        self.env = env
        self.dtype = dtype
        self.proc: Optional[subprocess.Popen[str]] = None

    @property
    def host_port(self) -> Tuple[str, int]:
        return self.host, self.port

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            return

        proc = subprocess.Popen(
            self.cmd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.proc = proc

        health_url = f"http://{self.host}:{self.port}/health"
        deadline = time.time() + 180.0

        last_err = None
        while time.time() < deadline:
            try:
                r = requests.get(health_url, timeout=2.0)
                if r.status_code == 200:
                    return
            except Exception as e:  # noqa: BLE001
                last_err = e

            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                raise RuntimeError(
                    f"vLLM server exited early with code {proc.returncode}\n"
                    f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                )

            time.sleep(2.0)

        proc.terminate()
        stdout, stderr = proc.communicate(timeout=10)
        raise RuntimeError(
            f"vLLM server failed to become healthy at {health_url}\n"
            f"Last error: {last_err}\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            self.proc = None
            return

        try:
            os.killpg(self.proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=10)
        finally:
            self.proc = None

    def close(self) -> None:
        self.stop()


@pytest.fixture(scope="session")
def vllm_server(request) -> VLLMServerHandle:
    host = os.getenv("VLLM_HOST", "127.0.0.1")
    port = int(os.getenv("VLLM_PORT", "8000"))

    # Check if the test requested specific configuration (e.g. tools enabled)
    # via @pytest.mark.parametrize("vllm_server", [{"enable_tools": True}], indirect=True)
    config = getattr(request, "param", {})
    enable_tools = config.get("enable_tools", False)
    dtype = config.get("dtype", os.getenv("VLLM_DTYPE", "bfloat16"))

    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    cmd = [
        sys.executable,
        "-m",
        "ludic.inference.vllm_server",
        "--model",
        os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        "--dtype",
        dtype,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu_memory_utilization",
        "0.7",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "4096",
        "--enforce-eager",
    ]

    if enable_tools:
        cmd.extend([
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ])

    handle = VLLMServerHandle(host=host, port=port, cmd=cmd, env=env, dtype=dtype)
    try:
        handle.start()
    except Exception as exc:  # noqa: BLE001
        handle.stop()
        pytest.skip(f"vLLM server failed to start: {exc}")
    yield handle
    handle.stop()


@pytest.fixture(scope="session")
def vllm_host_port(vllm_server: VLLMServerHandle) -> Tuple[str, int]:
    return vllm_server.host_port


@pytest.fixture(scope="session")
def vllm_model_name() -> str:
    return os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


@pytest.fixture(scope="session")
def vllm_client(vllm_host_port: Tuple[str, int]) -> VLLMChatClient:
    host, port = vllm_host_port
    try:
        return VLLMChatClient(
            host=host,
            port=port,
            connection_timeout_s=5.0,
            enable_weight_updates=False,
        )
    except RequestsConnectionError:
        pytest.skip(f"vLLM server not reachable at {host}:{port}")

@pytest.fixture
def env_registry():
    """
    Registry mapping env kind -> factory. Used by RolloutEngine tests.
    """
    return {
        "mock": lambda **kwargs: MockEnv(**kwargs),
    }


@pytest.fixture
def ctx_registry():
    """
    Registry mapping ctx kind -> factory. Used by RolloutEngine tests.
    """
    return {
        "full_dialog": lambda **kwargs: FullDialog(**kwargs),
    }


@pytest.fixture
def mock_agent():
    """
    Real Agent wired to MockClient (from tests._mocks).
    """
    return MockAgent()
