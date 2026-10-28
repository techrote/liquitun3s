# Titan RTX local training profile

The primary target is an NVIDIA Titan RTX with 24 GB VRAM.

## Precision policy

The Titan RTX is a Turing GPU. liquitun3s therefore uses **FP16 mixed precision** for its default training profiles and does not assume native BF16 tensor-core training.

The Windows preflight reports:

- CUDA visibility to PyTorch;
- GPU name;
- total VRAM;
- compute capability;
- CUDA runtime reported by PyTorch;
- PyTorch's BF16 capability report;
- driver/memory summary from `nvidia-smi` when available.

A failed CUDA preflight is a hard stop. The project does not silently fall back to CPU training.

## Initial VLM profile

`config/titan_vl_lora.json` starts conservatively:

- `LiquidAI/LFM3-VL-2B`;
- exact upstream source pin;
- FP16 base weights;
- LoRA rank 16 / alpha 32;
- language + multimodal-projector linear modules;
- vision encoder excluded from LoRA in the first pass;
- gradient checkpointing enabled;
- one example at a time with gradient accumulation 8;
- images reduced to at most 768 px on their longest side;
- 2,048-token hard sequence bound;
- thinking disabled for compact structured Cybersand targets.

This is a bootstrap profile, not an asserted optimum.

## Why not start with QLoRA?

LFM3-VL-2B is only 1.92B parameters. On a 24 GB card, ordinary FP16 LoRA should be tested first because it is simpler and preserves an unquantized frozen base during adaptation.

`.[qlora]` exposes bitsandbytes as an optional dependency, but no QLoRA result is admitted until a local run demonstrates that it is needed and source/runtime compatible on the Windows/Turing stack.

## Escalation order

If the first VL run has substantial headroom:

1. raise `max_image_side` from 768 toward native screenshot resolution;
2. raise `max_sequence_tokens`;
3. consider `target_scope=all_linear` to LoRA-adapt the vision encoder as well;
4. consider rank 32, matching the scale Liquid reports for one of its LFM3-VL robotics adapters;
5. only then consider larger effective batches or additional modalities.

If the first run is near VRAM capacity:

1. reduce image side;
2. reduce sequence bound;
3. keep batch size 1 and increase gradient accumulation rather than batch size;
4. use language-only targets;
5. evaluate optional 4-bit QLoRA as a separate lane.

## First local qualification

Run in this order:

```powershell
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1

liquitun3s preflight
liquitun3s source vl
liquitun3s vl path\to\cybersand.png --prompt "Describe visible materials and actors as JSON."
liquitun3s validate-vl data\cybersand\train.jsonl
liquitun3s train-vl config\titan_vl_lora.json
```

Keep the first training dataset small enough to inspect manually. The first goal is to establish correct token masking, target-module counts, VRAM use, throughput, checkpoint load and held-out behavior—not to maximize training time.
