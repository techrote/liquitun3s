from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from .provenance import runtime_record, sha256_file, write_json
from .source import resolve_revision


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: config must be a JSON object")
    return cfg


def seed_everything(seed: int, torch) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_cuda_fp16(torch) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Titan RTX training profiles")
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < 20 * 1024**3:
        raise RuntimeError(
            f"detected only {props.total_memory / 1024**3:.1f} GiB VRAM; "
            "the default profile expects a 24 GiB-class GPU"
        )


def discover_linear_targets(
    model,
    torch,
    *,
    scope: str,
) -> tuple[list[str], dict[str, int]]:
    if scope not in {"language", "language_projector", "all_linear"}:
        raise ValueError("target_scope must be language, language_projector, or all_linear")

    targets: list[str] = []
    groups = {"language": 0, "projector": 0, "vision": 0, "skipped_head": 0}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lower = name.lower()
        if lower.endswith("lm_head") or ".lm_head" in lower:
            groups["skipped_head"] += 1
            continue
        is_projector = any(token in lower for token in ("projector", "connector", "multi_modal"))
        is_vision = any(token in lower for token in ("vision", "visual", "image_encoder"))
        if is_projector:
            group = "projector"
        elif is_vision:
            group = "vision"
        else:
            group = "language"
        groups[group] += 1

        include = scope == "all_linear"
        if scope == "language":
            include = group == "language"
        elif scope == "language_projector":
            include = group in {"language", "projector"}
        if include:
            targets.append(name)

    if not targets:
        raise RuntimeError(f"no linear LoRA targets discovered for scope={scope}")
    return targets, groups


def resize_image(image, max_side: int):
    if max_side <= 0:
        return image
    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image
    scale = max_side / largest
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        resample=3,
    )


class TrainingMeter:
    def __init__(self, torch) -> None:
        self.torch = torch
        self.started = time.monotonic()
        self.steps: list[dict[str, Any]] = []
        torch.cuda.reset_peak_memory_stats()

    def add(self, *, step: int, loss: float, tokens: int, seconds: float) -> None:
        self.steps.append(
            {
                "step": step,
                "loss": loss,
                "tokens": tokens,
                "seconds": seconds,
                "tokens_per_second": tokens / seconds if seconds else None,
                "allocated_bytes": self.torch.cuda.memory_allocated(),
                "reserved_bytes": self.torch.cuda.memory_reserved(),
            }
        )

    def summary(self) -> dict[str, Any]:
        total_tokens = sum(row["tokens"] for row in self.steps)
        total_seconds = sum(row["seconds"] for row in self.steps)
        return {
            "steps": len(self.steps),
            "tokens": total_tokens,
            "training_seconds": total_seconds,
            "tokens_per_second": total_tokens / total_seconds if total_seconds else None,
            "first_loss": self.steps[0]["loss"] if self.steps else None,
            "last_loss": self.steps[-1]["loss"] if self.steps else None,
            "mean_loss": (
                sum(row["loss"] for row in self.steps) / len(self.steps) if self.steps else None
            ),
            "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(),
            "rows": self.steps,
        }


def base_run_record(
    *,
    config_path: Path,
    dataset_path: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "liquitun3s-local-training-v1",
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "source": source,
        "runtime": runtime_record(),
    }


def resolve_training_source(alias: str, requested: str | None) -> dict[str, Any]:
    return resolve_revision(alias, requested)


def save_run(output_dir: Path, record: dict[str, Any]) -> None:
    write_json(output_dir / "run.json", record)
