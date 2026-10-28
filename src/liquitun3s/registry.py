from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def load_registry() -> dict[str, Any]:
    path = files("liquitun3s").joinpath("model_registry.json")
    return json.loads(path.read_text(encoding="utf-8"))


def aliases() -> list[str]:
    return sorted(load_registry()["models"])


def model_entry(alias: str) -> dict[str, Any]:
    models = load_registry()["models"]
    if alias not in models:
        raise KeyError(f"unknown model alias {alias!r}; choose one of: {', '.join(sorted(models))}")
    return {"alias": alias, **models[alias]}
