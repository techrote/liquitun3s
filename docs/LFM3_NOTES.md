# LFM3 source notes

Last checked: 2026-10-28.

## Upstream family

Liquid AI released LFM3 on 2026-10-27 as a family of roughly 2B-class text, vision, audio, omni and decision models intended for specialization and local/edge deployment.

liquitun3s currently admits only:

| Alias | Model | Role | Local training status |
|---|---|---|---|
| `vl` | `LiquidAI/LFM3-VL-2B` | image/video -> text | primary FP16 LoRA bootstrap |
| `text` | `LiquidAI/LFM3-1.7B` | text -> text/tools/JSON | secondary FP16 LoRA bootstrap |
| `text-base` | `LiquidAI/LFM3-1.7B-Base` | base checkpoint | registered, not default |

Audio, Omni and d1 models are intentionally not admitted by the bootstrap.

## Vision model facts

The current LFM3-VL-2B model card describes:

- 1.92B parameters total;
- 1.73B LFM3 language model;
- 164M vision encoder;
- 25M projector;
- images and video as input, text as output;
- 128K context;
- native-resolution images, with 2x2 patch blocks represented as one language token;
- video sampled at 2 FPS;
- Transformers `AutoModelForImageTextToText` + `AutoProcessor`;
- `transformers>=5.21.0`;
- TRL-based SFT and GRPO fine-tuning examples.

Pinned bootstrap revision:

`f83d2b0a2296aaa277f394990a9b8b949ddf9691`

The source resolver verifies that pin before the default VL training run.

## Fine-tuning precedent

Liquid's release material reports an LFM3-VL robot orchestrator fine-tuned with a rank-32 LoRA on about 40,000 samples combining manipulation trajectories, tool use and pointing data. The reported run took about six GPU-hours on one AMD Instinct MI325X.

That is evidence that the architecture is intended to accept LoRA specialization. It is **not** a Titan RTX runtime estimate.

Liquid's general TRL documentation recommends LoRA for VLM fine-tuning, batch size 1, gradient accumulation and gradient checkpointing. The upstream examples are BF16-oriented; the Titan RTX profile translates that to FP16.

## Text model

LFM3-1.7B is 1.73B parameters with a 256K context window and is positioned for local agents, structured output, tool use and domain-specific tuning.

The text registry currently resolves and records the upstream SHA at run time rather than claiming a bootstrap pin from the VL source record.

## Deliberate non-claims

The bootstrap does not claim:

- Titan RTX training throughput yet;
- a safe native-resolution image limit yet;
- video fine-tuning qualification;
- vision-encoder LoRA qualification;
- QLoRA qualification;
- full-parameter fine-tuning;
- GRPO/DPO/CPT qualification;
- audio/omni/d1 support;
- GGUF export of local adapters;
- production Cybersand inference integration.

Those require measured local evidence or separate implementation work.
