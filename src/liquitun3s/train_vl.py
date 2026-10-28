from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .datasets import load_manifest, validate_vl_manifest, vl_messages
from .training import (
    TrainingMeter,
    base_run_record,
    discover_linear_targets,
    load_config,
    require_cuda_fp16,
    resize_image,
    resolve_training_source,
    save_run,
    seed_everything,
)


def _assistant_labels(batch, torch):
    mask = batch.pop("assistant_masks", None)
    if mask is None:
        mask = batch.pop("assistant_tokens_mask", None)
    if mask is None:
        raise RuntimeError(
            "the LFM3-VL chat template did not return an assistant token mask; "
            "refusing to train on prompt/image tokens"
        )
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask)
    labels = batch["input_ids"].clone()
    mask = mask.to(labels.device).bool()
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != labels.shape:
        raise RuntimeError(
            f"assistant mask shape {tuple(mask.shape)} does not match input ids {tuple(labels.shape)}"
        )
    labels[~mask] = -100
    if "attention_mask" in batch:
        labels[batch["attention_mask"] == 0] = -100
    if not torch.any(labels != -100):
        raise RuntimeError("assistant target was entirely masked")
    return labels


def train_vl(config_path: Path) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError('install local training dependencies with: pip install -e ".[train]"') from exc

    cfg = load_config(config_path)
    if cfg.get("model") != "vl":
        raise ValueError("VL trainer requires config model='vl'")
    require_cuda_fp16(torch)
    seed = int(cfg.get("seed", 20261028))
    seed_everything(seed, torch)

    dataset_path = (config_path.parent / cfg["dataset"]).resolve()
    dataset_summary = validate_vl_manifest(dataset_path)
    rows = load_manifest(dataset_path)
    source = resolve_training_source("vl", cfg.get("revision"))
    revision = source["resolved_revision"]
    model_id = source["model_id"]

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    load_started = time.monotonic()
    base = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        dtype=torch.float16,
    ).to("cuda")
    base.config.use_cache = False
    if bool(cfg.get("gradient_checkpointing", True)):
        base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()

    lora = cfg["lora"]
    targets, target_groups = discover_linear_targets(
        base,
        torch,
        scope=str(lora.get("target_scope", "language_projector")),
    )
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    load_seconds = time.monotonic() - load_started

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.get("learning_rate", 1e-4)),
        betas=(0.9, 0.95),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    grad_accum = int(cfg.get("gradient_accumulation_steps", 8))
    if grad_accum <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    epochs = int(cfg.get("epochs", 1))
    max_steps = cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    max_sequence = int(cfg.get("max_sequence_tokens", 2048))
    max_image_side = int(cfg.get("max_image_side", 768))
    enable_thinking = bool(cfg.get("enable_thinking", False))
    clip = float(cfg.get("gradient_clip", 1.0))
    output_dir = (config_path.parent / cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meter = TrainingMeter(torch)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    optimizer_step = 0
    accumulated_tokens = 0
    accumulated_started: float | None = None
    last_raw_loss: float | None = None
    stop = False
    model.train()

    for _epoch in range(epochs):
        for row in rows:
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = dataset_path.parent / image_path
            with Image.open(image_path) as source_image:
                image = resize_image(source_image.convert("RGB"), max_image_side)
                messages = vl_messages(row, image)
                batch = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    return_assistant_tokens_mask=True,
                    enable_thinking=enable_thinking,
                )
            if batch["input_ids"].shape[-1] > max_sequence:
                raise RuntimeError(
                    f"{row['id']}: {batch['input_ids'].shape[-1]} tokens exceeds "
                    f"max_sequence_tokens={max_sequence}; resize or edit the source deliberately"
                )
            labels = _assistant_labels(batch, torch)
            batch["labels"] = labels
            batch = {
                key: value.to("cuda") if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            token_count = int(batch["attention_mask"].sum()) if "attention_mask" in batch else int(
                batch["input_ids"].numel()
            )

            if accumulated_started is None:
                accumulated_started = time.monotonic()
            accumulated_tokens += token_count
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = model(**batch)
                raw_loss = result.loss
                loss = raw_loss / grad_accum
            scaler.scale(loss).backward()
            last_raw_loss = float(raw_loss.detach().cpu())
            micro_step += 1

            if micro_step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                torch.cuda.synchronize()
                meter.add(
                    step=optimizer_step,
                    loss=last_raw_loss,
                    tokens=accumulated_tokens,
                    seconds=time.monotonic() - accumulated_started,
                )
                accumulated_tokens = 0
                accumulated_started = None
                if max_steps is not None and optimizer_step >= max_steps:
                    stop = True
                    break
        if stop:
            break

    if micro_step % grad_accum and not stop:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        torch.cuda.synchronize()
        if accumulated_started is not None and last_raw_loss is not None:
            meter.add(
                step=optimizer_step,
                loss=last_raw_loss,
                tokens=accumulated_tokens,
                seconds=time.monotonic() - accumulated_started,
            )

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    processor.save_pretrained(adapter_dir)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    record = base_run_record(
        config_path=config_path,
        dataset_path=dataset_path,
        source=source,
    )
    record.update(
        {
            "lane": "vl-fp16-lora",
            "dataset_summary": dataset_summary,
            "seed": seed,
            "load_seconds": load_seconds,
            "precision": "fp16",
            "image": {"max_side": max_image_side},
            "max_sequence_tokens": max_sequence,
            "enable_thinking": enable_thinking,
            "lora": {
                "r": peft_config.r,
                "alpha": peft_config.lora_alpha,
                "dropout": peft_config.lora_dropout,
                "target_scope": lora.get("target_scope", "language_projector"),
                "target_count": len(targets),
                "target_groups_seen": target_groups,
                "targets": targets,
                "trainable_parameters": trainable_params,
                "total_parameters": total_params,
                "trainable_fraction": trainable_params / total_params,
            },
            "training": meter.summary(),
            "micro_steps": micro_step,
            "optimizer_steps": optimizer_step,
            "adapter": str(adapter_dir),
        }
    )
    save_run(output_dir, record)
    print(json.dumps(record, indent=2))
    return record
