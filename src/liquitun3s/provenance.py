from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        run = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return run.stdout.strip() or None


def runtime_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_head": git_head(),
        "pid": os.getpid(),
    }
    try:
        import torch

        record["torch"] = torch.__version__
        record["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
            record["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            record["gpu_total_memory"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        pass
    for name in ("transformers", "peft", "trl", "accelerate", "datasets"):
        try:
            module = __import__(name)
            record[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            pass
    return record


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
