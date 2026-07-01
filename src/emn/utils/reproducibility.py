"""
Reproducibility utilities.
Every experiment logs: git hash, config, hardware, CUDA version, seed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _TORCH_AVAILABLE:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_hash() -> str:
    """Return the current git commit hash (short), or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "unknown"


def get_hardware_info() -> Dict[str, Any]:
    """Return dict of hardware/software environment info."""
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": False,
    }
    if _TORCH_AVAILABLE:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        pass
    return info


def build_run_metadata(
    experiment_name: str,
    seed: int,
    config: dict,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    meta = {
        "experiment": experiment_name,
        "seed": seed,
        "git_hash": get_git_hash(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": get_hardware_info(),
        "config": config,
    }
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / "run_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
    return meta


def config_hash(config: dict) -> str:
    s = json.dumps(config, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]
