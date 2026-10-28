# Model, dataset and checkpoint storage

Git is the control plane, not the weight store.

## Keep in Git

- source/config files;
- model registry and exact source expectations;
- small public/example manifests;
- dataset schema and provenance metadata;
- run summaries that do not contain private data;
- evaluation definitions;
- documentation.

## Keep out of ordinary Git history

- Hugging Face model snapshots;
- safetensors weight files;
- GGUFs;
- optimizer checkpoints;
- local adapters unless deliberately published elsewhere;
- large screenshot/video datasets;
- private Cybersand captures;
- caches.

The root `.gitignore` excludes common model/checkpoint formats and local run/cache directories.

Every completed training run should retain at least:

- exact Hugging Face model ID + resolved revision;
- repository Git head;
- config SHA-256;
- dataset manifest SHA-256;
- package versions;
- GPU/driver/CUDA identity;
- LoRA target list and parameter counts;
- peak allocated/reserved VRAM;
- training wall time and losses;
- adapter directory identity;
- evaluation results.

If an adapter becomes important to Cybersand, publish/archive it deliberately with its `run.json` rather than relying on a workstation cache.
