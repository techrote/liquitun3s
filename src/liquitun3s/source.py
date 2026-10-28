from __future__ import annotations

from typing import Any

from .registry import model_entry


def resolve_revision(alias: str, requested: str | None = None) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required") from exc

    entry = model_entry(alias)
    model_id = entry["id"]
    expected = entry.get("expected_revision")
    revision = requested or expected or "main"
    info = HfApi().model_info(model_id, revision=revision)
    resolved = info.sha
    if not resolved:
        raise RuntimeError(f"Hugging Face returned no revision for {model_id}@{revision}")
    if expected and requested is None and resolved != expected:
        raise RuntimeError(
            f"{alias} source drift: registry expects {expected}, Hugging Face resolved {resolved}"
        )
    return {
        "alias": alias,
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": resolved,
        "registry_expected_revision": expected,
    }
