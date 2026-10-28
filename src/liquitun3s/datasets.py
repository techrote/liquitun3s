from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cybersand import validate_intent

TASKS = {"scene_json", "hazard_json", "intent_json", "visual_regression", "freeform"}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def validate_vl_manifest(
    path: Path,
    *,
    allow_missing_images: bool = False,
) -> dict[str, Any]:
    rows = load_manifest(path)
    base = path.parent
    ids: set[str] = set()
    tasks: dict[str, int] = {}
    for index, row in enumerate(rows, 1):
        where = f"{path}:{index}"
        required = {"id", "image", "task", "prompt", "response"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"{where}: missing keys {sorted(missing)}")
        sample_id = row["id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{where}: id must be a non-empty string")
        if sample_id in ids:
            raise ValueError(f"{where}: duplicate id {sample_id!r}")
        ids.add(sample_id)
        task = row["task"]
        if task not in TASKS:
            raise ValueError(f"{where}: unknown task {task!r}")
        tasks[task] = tasks.get(task, 0) + 1
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError(f"{where}: prompt must be a non-empty string")
        if not isinstance(row["response"], str) or not row["response"].strip():
            raise ValueError(f"{where}: response must be a non-empty string")
        image = Path(row["image"])
        image_path = image if image.is_absolute() else base / image
        if not allow_missing_images and not image_path.is_file():
            raise ValueError(f"{where}: missing image {image_path}")
        if task == "intent_json":
            try:
                response = json.loads(row["response"])
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: intent response is not JSON") from exc
            try:
                validate_intent(response)
            except ValueError as exc:
                raise ValueError(f"{where}: {exc}") from exc
    return {"examples": len(rows), "tasks": tasks, "ids": len(ids)}


def vl_messages(row: dict[str, Any], image: Any) -> list[dict[str, Any]]:
    context = row.get("context")
    prompt = row["prompt"]
    if context is not None:
        prompt = f"{prompt}\n\nCopied Cybersand observation/context:\n{json.dumps(context, sort_keys=True)}"
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a bounded Cybersand visual specialist. "
                        "Use only visible evidence and copied context. "
                        "Do not invent hidden world state."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": row["response"]}],
        },
    ]
