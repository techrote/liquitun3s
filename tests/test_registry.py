from liquitun3s.registry import aliases, model_entry


def test_primary_vision_lane_is_pinned():
    entry = model_entry("vl")
    assert entry["id"] == "LiquidAI/LFM3-VL-2B"
    assert entry["local_training"] == "fp16-lora"
    assert len(entry["expected_revision"]) == 40


def test_registry_is_intentionally_bounded():
    assert aliases() == ["text", "text-base", "vl"]
    assert model_entry("text-base")["status"] == "registered-not-default"
