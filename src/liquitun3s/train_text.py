from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .training import (
    TrainingMeter,
    base_run_record,
    discover_linear_targets,
    load_config,
    require_cuda_fp16,
    resolve_training_source,
    save_run,
    seed_everything,
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"{path}:{number}: messages must end with assistant")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def train_text(config_path: Path) -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('install local training dependencies with: pip install -e ".[train]"') from exc

    cfg = load_config(config_path)
    if cfg.get("model") != "text":
        raise ValueError("text trainer requires config model='text'")
    require_cuda_fp16(torch)
    seed = int(cfg.get("seed", 20261028))
    seed_everything(seed, torch)

    dataset_path = (config_path.parent / cfg["dataset"]).resolve()
    rows = _load_rows(dataset_path)
    source = resolve_training_source("text", cfg.get("revision"))
    revision = source["resolved_revision"]
    model_id = source["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_started = time.monotonic()
    base = AutoModelForCausalLM.from_pretrained(
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
    targets, groups = discover_linear_targets(base, torch, scope="language")
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

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg.get("learning_rate", 2e-4)),
        betas=(0.9, 0.95),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    grad_accum = int(cfg.get("gradient_accumulation_steps", 8))
    epochs = int(cfg.get("epochs", 1))
    max_sequence = int(cfg.get("max_sequence_tokens", 2048))
    enable_thinking = bool(cfg.get("enable_thinking", False))
    output_dir = (config_path.parent / cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

    meter = TrainingMeter(torch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    opt_step = 0
    accumulated_tokens = 0
    accumulated_started: float | None = None
    last_raw_loss: float | None = None
    for _epoch in range(epochs):
        for row in rows:
            batch = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                return_assistant_tokens_mask=True,
                enable_thinking=enable_thinking,
            )
            mask = batch.pop("assistant_masks", None)
            if mask is None:
                raise RuntimeError("chat template returned no assistant token mask")
            labels = batch["input_ids"].clone()
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask)
            mask = mask.bool()
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            labels[~mask] = -100
            if not torch.any(labels != -100):
                raise RuntimeError("assistant target was entirely masked")
            if labels.shape[-1] > max_sequence:
                raise RuntimeError(
                    f"example has {labels.shape[-1]} tokens, exceeds max_sequence_tokens={max_sequence}"
                )
            batch["labels"] = labels
            batch = {key: value.to("cuda") for key, value in batch.items() if torch.is_tensor(value)}
            tokens = int(batch["input_ids"].numel())
            if accumulated_started is None:
                accumulated_started = time.monotonic()
            accumulated_tokens += tokens
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(**batch)
                raw_loss = output.loss
                loss = raw_loss / grad_accum
            scaler.scale(loss).backward()
            last_raw_loss = float(raw_loss.detach().cpu())
            micro += 1
            if micro % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg.get("gradient_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                torch.cuda.synchronize()
                meter.add(
                    step=opt_step,
                    loss=last_raw_loss,
                    tokens=accumulated_tokens,
                    seconds=time.monotonic() - accumulated_started,
                )
                accumulated_tokens = 0
                accumulated_started = None

    if micro % grad_accum:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, float(cfg.get("gradient_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        opt_step += 1
        torch.cuda.synchronize()
        if accumulated_started is not None and last_raw_loss is not None:
            meter.add(
                step=opt_step,
                loss=last_raw_loss,
                tokens=accumulated_tokens,
                seconds=time.monotonic() - accumulated_started,
            )

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    record = base_run_record(config_path=config_path, dataset_path=dataset_path, source=source)
    record.update(
        {
            "lane": "text-fp16-lora",
            "seed": seed,
            "load_seconds": load_seconds,
            "precision": "fp16",
            "max_sequence_tokens": max_sequence,
            "lora": {
                "r": peft_config.r,
                "alpha": peft_config.lora_alpha,
                "target_count": len(targets),
                "target_groups_seen": groups,
                "trainable_parameters": trainable_params,
                "total_parameters": total_params,
                "trainable_fraction": trainable_params / total_params,
            },
            "training": meter.summary(),
            "micro_steps": micro,
            "optimizer_steps": opt_step,
            "adapter": str(adapter_dir),
        }
    )
    save_run(output_dir, record)
    print(json.dumps(record, indent=2))
    return record
