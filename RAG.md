# liquitun3s retrieval map

Read this file first when modifying training support or interpreting results.

## Authority order

1. `src/liquitun3s/model_registry.json` — admitted model lanes and pinned source expectations.
2. `config/*.json` — concrete local Titan training profiles.
3. Generated `runs/**/run.json` — exact model revision, dataset/config hashes, runtime packages, GPU identity and measured memory/runtime.
4. Training dataset manifests — the examples actually supplied to a run.
5. `docs/CYBERSAND.md` — Cybersand integration boundary and task semantics.
6. `docs/TITAN_RTX.md` — hardware/precision policy.
7. `docs/LFM3_NOTES.md` — upstream model facts and deliberate non-claims.
8. `docs/MODEL_STORAGE.md` — weight/checkpoint storage policy.

## Rules

- Training is local. GitHub Actions must never download LFM weights or execute model training.
- Resolve and record the exact Hugging Face revision used by every run.
- The primary VLM source is pinned; source drift fails closed unless a deliberate config/review updates the pin.
- Titan RTX runs are FP16-first. Do not silently switch a profile to BF16.
- Do not silently truncate image/text examples. Reduce image size or edit the source deliberately.
- VLM SFT must supervise assistant tokens only. Refuse training if the native chat template cannot identify the assistant span.
- A visual model sees pixels plus explicitly copied context; it does not know hidden cells or authoritative World state.
- For Cybersand actor control, the model proposes a high-level intent only. Runtime request IDs, actor IDs, epoch, observation identity/tick, lease and action admission remain engine/wrapper authority.
- Raw run records outrank prose summaries.
- Do not claim video fine-tuning, audio/omni tuning, d1 support, QLoRA, merged export or deployment qualification until those paths have their own local evidence.
- Never commit upstream weights or ordinary training checkpoints to Git.
