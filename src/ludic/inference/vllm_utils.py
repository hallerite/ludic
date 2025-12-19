"""
Small utilities for managing a local vLLM server process in examples or tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import requests

def _default_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    return env


def start_vllm_server(
    model: str,
    host: str,
    port: int,
    *,
    gpu_memory_utilization: float = 0.7,
    max_model_len: int | None = None,
    enforce_eager: bool = True,
    stream_output: bool = True,
    extra_args: list[str] | None = None,
) -> subprocess.Popen[str]:
    """
    Launch a local vLLM server using ludic.inference.vllm_server.

    Set stream_output=False if the caller needs to capture stdout/stderr;
    otherwise logs stream to the parent to avoid pipe backpressure.
    """
    extra_args = extra_args or []
    cmd = [
        sys.executable,
        "-m",
        "ludic.inference.vllm_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
    ]

    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])

    # vLLM can load model-provided HF generation configs that override default
    # sampling params. Default to `vllm` unless the caller explicitly opts out.
    if not any(a == "--generation-config" or a.startswith("--generation-config=") for a in extra_args):
        cmd.extend(["--generation-config", "vllm"])

    cmd.extend(extra_args)

    if enforce_eager:
        cmd.append("--enforce-eager")

    stdout = None if stream_output else subprocess.PIPE
    stderr = None if stream_output else subprocess.PIPE

    return subprocess.Popen(
        cmd,
        env=_default_env(),
        stdout=stdout,
        stderr=stderr,
        text=True,
    )


def collect_proc_output(proc: subprocess.Popen[str], timeout: float | None = None) -> tuple[str, str]:
    """
    Capture process output if it was piped; otherwise just wait for exit.
    """
    if proc.stdout is None and proc.stderr is None:
        proc.wait(timeout=timeout)
        return ("<logs streamed to parent stdout/stderr>", "")

    stdout, stderr = proc.communicate(timeout=timeout)
    return (stdout or "", stderr or "")


def wait_for_vllm_health(host: str, port: int, proc: subprocess.Popen[str], timeout_s: float = 180.0) -> None:
    """
    Poll the vLLM health endpoint until ready or the process exits/fails.
    """
    health_url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e

        if proc.poll() is not None:
            stdout, stderr = collect_proc_output(proc)
            raise RuntimeError(
                f"vLLM server exited early with code {proc.returncode}\n"
                f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        time.sleep(2.0)

    proc.terminate()
    stdout, stderr = collect_proc_output(proc, timeout=10)
    raise RuntimeError(
        f"vLLM server failed to become healthy at {health_url}\n"
        f"Last error: {last_err}\n"
        f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    )
