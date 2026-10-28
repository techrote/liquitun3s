import json
from pathlib import Path


def test_configs_parse_and_use_fp16_lora_profiles():
    vl = json.loads(Path("config/titan_vl_lora.json").read_text())
    text = json.loads(Path("config/titan_text_lora.json").read_text())
    assert vl["model"] == "vl"
    assert vl["lora"]["target_scope"] == "language_projector"
    assert vl["lora"]["r"] == 16
    assert text["model"] == "text"
    assert text["lora"]["r"] == 16
