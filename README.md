# liquitun3s

Local **NVIDIA Titan RTX 24 GB** training and evaluation for Liquid AI's LFM3 family, focused on Cybersand.

This repo intentionally replaces the deleted GitHub-runner experiment with a **local-GPU-first** design:

- GitHub stores code, configs, schemas and evidence.
- The Titan RTX downloads weights and performs all inference/training.
- GitHub Actions is static/unit CI only and never downloads LFM weights.
- Every real run records exact Hugging Face source identity, config/dataset hashes and GPU/runtime metadata.

The primary lane is **LFM3-VL-2B**, because its image/video understanding, spatial grounding and embodied reasoning map well to Cybersand screenshot analysis and long-latency actor intent.

Start with [RAG.md](RAG.md).

## What is supported

| Alias | Model | Purpose | Bootstrap training path |
|---|---|---|---|
| `vl` | `LiquidAI/LFM3-VL-2B` | screenshots/images -> structured visual understanding / high-level intent | FP16 LoRA, primary |
| `text` | `LiquidAI/LFM3-1.7B` | structured output, tools, domain language | FP16 LoRA |
| `text-base` | `LiquidAI/LFM3-1.7B-Base` | heavy adaptation / research | registered only |

Audio, Omni and d1 are deliberately outside v0.1.

## Why the vision model is the lead

Cybersand already has a bounded actor observation/action architecture, but its compact numeric observation does not contain terrain imagery or neighboring-world perception.

LFM3-VL gives us a complementary local perception layer for:

- visible material/actor/hazard extraction;
- screenshot-based visual regression;
- material/environment specialization;
- high-level `hold` / `move_to` / `pursue_visible` intent;
- later image-history/video experiments.

The model does **not** own simulation state. See [docs/CYBERSAND.md](docs/CYBERSAND.md).

## Windows / Titan RTX setup

Python 3.11 is the default tested environment.

From PowerShell:

```powershell
git clone https://github.com/techrote/liquitun3s.git
cd liquitun3s

.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

The setup script installs PyTorch first, then the training dependencies, then runs:

```powershell
liquitun3s preflight
liquitun3s source vl
```

If the default PyTorch install does not expose CUDA on your machine, set `PYTORCH_INDEX_URL` to the current CUDA wheel index recommended by PyTorch and rerun the setup script.

The Titan profile is **FP16**, not BF16.

## First vision inference

```powershell
liquitun3s vl C:\captures\cybersand.png --prompt "Return compact JSON describing visible materials, actors and hazards." --record runs\smoke-vl.json
```


The default model source is pinned in the registry. The command refuses source drift unless you explicitly choose another revision.

## Prepare a Cybersand VLM dataset

Copy the example manifest:

```powershell
mkdir data\cybersand -ErrorAction SilentlyContinue
copy examples\cybersand_vl_manifest.example.jsonl data\cybersand\train.jsonl
```

Replace each placeholder image path and target with real reviewed captures, then validate:

```powershell
liquitun3s validate-vl data\cybersand\train.jsonl
```

Manifest rows look like:

```json
{
  "id": "intent-0042",
  "image": "images/frame-0042.png",
  "task": "intent_json",
  "prompt": "Choose a high-level intent for the visible actor.",
  "response": "{\"mode\":\"move_to\",\"goal\":[420.0,280.0],\"speed\":0.6,\"engage\":false}",
  "context": {
    "actor_id": 0,
    "observation_tick": 1200
  }
}
```

For `intent_json`, the validator deliberately permits only:

- `mode`
- `goal`
- `speed`
- `engage`

The model is never trained to invent request IDs, epochs, ticks or command sequence authority.

## First VL LoRA run

The bootstrap Titan profile is:

- FP16;
- rank 16 / alpha 32;
- batch size effectively 1 with gradient accumulation;
- gradient checkpointing;
- max image side 768;
- max sequence length 2,048;
- language + multimodal projector LoRA;
- vision encoder frozen from LoRA initially;
- assistant-token-only loss.

Run:

```powershell
liquitun3s train-vl config\titan_vl_lora.json
```

Output goes under `runs/vl-cybersand/` and includes:

```
adapter/
run.json
```

`run.json` records the exact model revision, data/config hashes, GPU/runtime packages, LoRA target modules, trainable parameter count, peak CUDA memory and measured training performance.

## Evaluate the adapter

```powershell
liquitun3s vl C:\captures\heldout.png --adapter runs\vl-cybersand\adapter --prompt "Return only the high-level intent JSON for the visible actor." --record runs\heldout-intent.json
```

Always compare the base model and adapter on the same held-out captures.

## Text LoRA

Create `data/text/train.jsonl` using [examples/text_sft.example.jsonl](examples/text_sft.example.jsonl), then:

```powershell
liquitun3s train-text config\titan_text_lora.json
```

The text lane is secondary to VL for this project.

## Development / CI

Lightweight CI only:

```powershell
python -m pip install -e ".[dev]" --no-deps
python -m compileall -q src tests
ruff check src tests
pytest -q
```

CI does not install PyTorch, download model weights or train anything.

## Current evidence boundary

The repo is now set up for local Titan work, but no Titan training result is claimed until you actually run it.

Upstream Liquid AI states that LFM3 models are intended for single-GPU fine-tuning, and specifically exposes TRL SFT for LFM3-VL-2B. The local profile translates the upstream BF16-oriented examples to the Titan RTX's FP16/Turing environment.

See:

- [docs/TITAN_RTX.md](docs/TITAN_RTX.md)
- [docs/CYBERSAND.md](docs/CYBERSAND.md)
- [docs/LFM3_NOTES.md](docs/LFM3_NOTES.md)
- [docs/MODEL_STORAGE.md](docs/MODEL_STORAGE.md)
