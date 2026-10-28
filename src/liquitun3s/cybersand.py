from __future__ import annotations

from typing import Any

INTENT_MODES = {"hold", "move_to", "pursue_visible"}
SMART_LINK_REQUEST_KEYS = {
    "version",
    "request_id",
    "run_id",
    "actor_id",
    "epoch",
    "observation_id",
    "observed_tick",
    "expires_tick",
    "observation",
}


def validate_intent(intent: Any) -> dict[str, Any]:
    if not isinstance(intent, dict):
        raise ValueError("intent must be an object")
    if set(intent) != {"mode", "goal", "speed", "engage"}:
        raise ValueError("intent keys must be exactly mode, goal, speed, engage")
    if intent["mode"] not in INTENT_MODES:
        raise ValueError(f"invalid intent mode {intent['mode']!r}")
    goal = intent["goal"]
    if not isinstance(goal, list) or len(goal) != 2:
        raise ValueError("intent goal must be [x, y]")
    if not all(isinstance(value, (int, float)) and 0 <= float(value) <= 1024 for value in goal):
        raise ValueError("intent goal coordinates must be finite values in 0..1024")
    speed = intent["speed"]
    if not isinstance(speed, (int, float)) or not 0 <= float(speed) <= 1:
        raise ValueError("intent speed must be in 0..1")
    if not isinstance(intent["engage"], bool):
        raise ValueError("intent engage must be boolean")
    return {
        "mode": intent["mode"],
        "goal": [float(goal[0]), float(goal[1])],
        "speed": float(speed),
        "engage": intent["engage"],
    }


def wrap_smart_link_response(
    request: dict[str, Any],
    intent: dict[str, Any],
    *,
    valid_for_ticks: int = 30,
) -> dict[str, Any]:
    missing = SMART_LINK_REQUEST_KEYS - set(request)
    if missing:
        raise ValueError(f"smart-link request missing keys: {sorted(missing)}")
    if request["version"] != 1:
        raise ValueError("only smart-link request version 1 is supported")
    if not isinstance(valid_for_ticks, int) or not 1 <= valid_for_ticks <= 120:
        raise ValueError("valid_for_ticks must be an integer in 1..120")
    checked = validate_intent(intent)
    return {
        "version": 1,
        "request_id": request["request_id"],
        "run_id": request["run_id"],
        "actor_id": request["actor_id"],
        "epoch": request["epoch"],
        "observation_id": request["observation_id"],
        "observed_tick": request["observed_tick"],
        "valid_for_ticks": valid_for_ticks,
        "intent": checked,
    }
