import json

import pytest

from liquitun3s.datasets import validate_vl_manifest


def _write(path, row):
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_intent_manifest_accepts_high_level_intent(tmp_path):
    image = tmp_path / "frame.png"
    image.touch()
    manifest = tmp_path / "train.jsonl"
    _write(
        manifest,
        {
            "id": "intent-1",
            "image": "frame.png",
            "task": "intent_json",
            "prompt": "Choose intent.",
            "response": json.dumps(
                {
                    "mode": "move_to",
                    "goal": [10.0, 20.0],
                    "speed": 0.5,
                    "engage": False,
                }
            ),
        },
    )
    result = validate_vl_manifest(manifest)
    assert result["examples"] == 1
    assert result["tasks"] == {"intent_json": 1}


def test_intent_manifest_rejects_model_generated_identity_fields(tmp_path):
    image = tmp_path / "frame.png"
    image.touch()
    manifest = tmp_path / "train.jsonl"
    _write(
        manifest,
        {
            "id": "intent-1",
            "image": "frame.png",
            "task": "intent_json",
            "prompt": "Choose intent.",
            "response": json.dumps(
                {
                    "request_id": "model-should-not-own-this",
                    "mode": "hold",
                    "goal": [0.0, 0.0],
                    "speed": 0.0,
                    "engage": False,
                }
            ),
        },
    )
    with pytest.raises(ValueError, match="intent keys"):
        validate_vl_manifest(manifest)


def test_example_manifest_can_be_schema_checked_without_images():
    result = validate_vl_manifest(
        __import__("pathlib").Path("examples/cybersand_vl_manifest.example.jsonl"),
        allow_missing_images=True,
    )
    assert result["examples"] == 4
