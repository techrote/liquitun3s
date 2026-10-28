from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .provenance import runtime_record, write_json
from .source import resolve_revision
from .training import require_cuda_fp16, resize_image


def run_vl(
    *,
    image_path: Path,
    prompt: str,
    adapter: Path | None = None,
    revision: str | None = None,
    max_image_side: int = 768,
    max_new_tokens: int = 256,
    thinking: bool = False,
    record_path: Path | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError('install local dependencies with: pip install -e ".[train]"') from exc

    require_cuda_fp16(torch)
    source = resolve_revision("vl", revision)
    model_id = source["model_id"]
    resolved = source["resolved_revision"]
    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=resolved,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision=resolved,
        trust_remote_code=True,
        dtype=torch.float16,
    ).to("cuda")

    if adapter is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to load an adapter") from exc
        model = PeftModel.from_pretrained(model, adapter).to("cuda")
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    with Image.open(image_path) as source_image:
        image = resize_image(source_image.convert("RGB"), max_image_side)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=thinking,
        )
    inputs = {
        key: value.to("cuda") if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    started = time.monotonic()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.1,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1,
            max_new_tokens=max_new_tokens,
        )
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    prompt_tokens = inputs["input_ids"].shape[1]
    generated = output_ids[:, prompt_tokens:]
    output = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    record = {
        "schema": "liquitun3s-vl-eval-v1",
        "source": source,
        "adapter": str(adapter.resolve()) if adapter else None,
        "image": str(image_path.resolve()),
        "max_image_side": max_image_side,
        "thinking": thinking,
        "prompt_tokens": int(prompt_tokens),
        "generated_tokens": int(generated.numel()),
        "seconds": seconds,
        "tokens_per_second": float(generated.numel() / seconds) if seconds else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "output": output,
        "runtime": runtime_record(),
    }
    try:
        record["parsed_json"] = json.loads(output)
        record["json_valid"] = True
    except json.JSONDecodeError:
        record["parsed_json"] = None
        record["json_valid"] = False

    if record_path:
        write_json(record_path, record)
    return record
