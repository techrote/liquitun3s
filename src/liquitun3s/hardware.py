from __future__ import annotations

import json
import platform
import subprocess
from typing import Any


def _nvidia_smi() -> str | None:
    try:
        run = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return run.stdout.strip() or None


def inspect_torch() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "torch_installed": False,
            "cuda_available": False,
            "ready": False,
            "problems": ["PyTorch is not installed."],
        }

    result: dict[str, Any] = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi": _nvidia_smi(),
        "problems": [],
        "warnings": [],
    }
    if not torch.cuda.is_available():
        result["ready"] = False
        result["problems"].append(
            "CUDA is not available to PyTorch. Install a CUDA-enabled PyTorch build before training."
        )
        return result

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    total_gib = props.total_memory / (1024**3)
    capability = torch.cuda.get_device_capability(index)
    result.update(
        {
            "device_index": index,
            "gpu_name": props.name,
            "vram_gib": total_gib,
            "compute_capability": list(capability),
            "cuda_runtime": torch.version.cuda,
            "bf16_supported_by_torch": bool(torch.cuda.is_bf16_supported()),
            "fp16_primary": True,
        }
    )
    if total_gib < 20:
        result["problems"].append(
            f"Only {total_gib:.1f} GiB VRAM detected; Titan profile expects about 24 GiB."
        )
    if capability < (7, 5):
        result["warnings"].append(
            f"Compute capability {capability[0]}.{capability[1]} is below the Titan RTX 7.5 target."
        )
    if torch.cuda.is_bf16_supported():
        result["warnings"].append(
            "Torch reports BF16 support, but the Titan RTX profile remains FP16-first for reproducibility."
        )
    else:
        result["warnings"].append(
            "BF16 is not reported as supported; this is expected for the Titan RTX/Turing profile."
        )
    result["ready"] = not result["problems"]
    return result


def print_preflight() -> int:
    result = inspect_torch()
    print(json.dumps(result, indent=2))
    return 0 if result.get("ready") else 2
