"""Helpers for managing MAX-served judge models.

Starts/stops Modular MAX inference servers for the open-source judge models
(Qwen3-235B and DeepSeek-V3).
"""

import logging
import subprocess
import time

import requests

from eval.config import JUDGES

logger = logging.getLogger(__name__)

# Map judge keys to their MAX serve ports
JUDGE_PORTS = {
    "qwen3": 8010,
    "deepseek": 8011,
}


def start_judge_server(judge_key: str, gpu_ids: str | None = None) -> subprocess.Popen:
    """Start a MAX serve instance for a judge model.

    Args:
        judge_key: Key into JUDGES config ("qwen3" or "deepseek").
        gpu_ids: CUDA_VISIBLE_DEVICES value (e.g., "0" or "0,1").

    Returns:
        The subprocess.Popen handle.
    """
    judge = JUDGES[judge_key]
    port = JUDGE_PORTS[judge_key]

    cmd = [
        "max", "serve",
        "--model", judge.model_id,
        "--port", str(port),
    ]

    env = None
    if gpu_ids:
        import os
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_ids}

    logger.info("Starting MAX serve for %s on port %d", judge.name, port)
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def wait_for_server(port: int, timeout: int = 300, poll_interval: int = 5) -> bool:
    """Wait for a MAX serve instance to become healthy.

    Args:
        port: The port to check.
        timeout: Maximum seconds to wait.
        poll_interval: Seconds between health checks.

    Returns:
        True if server is ready, False if timeout.
    """
    url = f"http://localhost:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                logger.info("MAX serve on port %d is ready", port)
                return True
        except requests.ConnectionError:
            pass
        time.sleep(poll_interval)
    logger.error("MAX serve on port %d failed to start within %ds", port, timeout)
    return False


def start_all_judges(gpu_mapping: dict[str, str] | None = None) -> dict[str, subprocess.Popen]:
    """Start MAX serve for all open-source judges.

    Args:
        gpu_mapping: Optional mapping of judge_key -> GPU IDs.
            e.g., {"qwen3": "0", "deepseek": "1"}

    Returns:
        Dict of judge_key -> subprocess handle.
    """
    gpu_mapping = gpu_mapping or {}
    processes = {}

    for key in ["qwen3", "deepseek"]:
        proc = start_judge_server(key, gpu_ids=gpu_mapping.get(key))
        processes[key] = proc

    # Wait for all to be ready
    all_ready = True
    for key in processes:
        if not wait_for_server(JUDGE_PORTS[key]):
            all_ready = False
            logger.error("Judge %s failed to start", key)

    if not all_ready:
        logger.warning("Not all judges started successfully")

    return processes


def stop_all_judges(processes: dict[str, subprocess.Popen]) -> None:
    """Stop all MAX serve instances."""
    for key, proc in processes.items():
        logger.info("Stopping MAX serve for %s (pid %d)", key, proc.pid)
        proc.terminate()
        proc.wait(timeout=30)
