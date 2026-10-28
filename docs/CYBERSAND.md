# Cybersand integration

This project is intended to produce local LFM3 adapters for Cybersand without giving the model authority that belongs to the simulation.

Reference Cybersand source when this bootstrap was authored:

- repository: `techrote/cybersand`
- main: `e50a7db32a5f0b94240902f305bd176ac94b05cc`
- relevant contracts:
  - `docs/systems/actor-control-prototypes.md`
  - `docs/systems/actor-rigs.md`
  - `godot/examples/thunderbowl/smart_link.gd`

## Why LFM3-VL-2B is interesting here

The model accepts images and video and is aimed at visual understanding, grounding, GUI understanding and embodied/spatial reasoning. That creates several Cybersand-specific research lanes that do not exist with the current compact numeric actor observation alone.

### 1. Visual scene reader

Input:

- one rendered frame;
- optional copied actor/session metadata.

Output:

- visible materials;
- visible actors;
- visible hazards;
- anomalies;
- uncertainty/unknown for information that is not visible.

This can support debugging, playtest analysis, visual acceptance and later NPC perception experiments.

### 2. Long-latency actor intent

Cybersand already separates long-latency planning from immediate motor control. The VLM should therefore predict only this bounded high-level intent:

```json
{
  "mode": "hold | move_to | pursue_visible",
  "goal": [420.0, 280.0],
  "speed": 0.6,
  "engage": false
}
```

The model must **not** generate or own:

- request ID;
- run ID;
- actor ID;
- epoch;
- observation ID;
- observed tick;
- validity/lease identity;
- accepted action sequence.

The runtime wrapper copies those values from the current request and rejects stale output. The existing Cybersand controller translates an admitted intent into bounded movement/actions.

This preserves the current policy: slow inference must not stall simulation ticks or acquire fresh authority merely because it arrived late.

### 3. Visual regression specialist

Screenshot + fixture question -> structured observation.

Examples:

- did water visibly cross an actor body?
- did a portal/objective render enabled before its gate?
- is a generated structure visibly enclosed?
- did a material effect disappear or clip?
- are intended materials visually distinguishable?

The VLM can triage or score visual evidence. It does not replace deterministic correctness fixtures.

### 4. Material / environment specialist

Train on screenshots plus labels derived from known fixture/world metadata so the adapter learns Cybersand-specific material appearance and spatial vocabulary.

Potential outputs:

- material regions;
- interaction descriptions;
- visible combustion/corrosion/phase-change state;
- local traversability;
- candidate target locations for a bounded planner.

## Dataset contract

The VLM JSONL manifest has:

- `id`: unique sample ID;
- `image`: path relative to the manifest;
- `task`: one of `scene_json`, `hazard_json`, `intent_json`, `visual_regression`, `freeform`;
- `prompt`: user instruction;
- `response`: assistant target string;
- optional `context`: copied non-authoritative metadata.

For `intent_json`, the validator requires exactly the four intent fields above. This deliberately rejects training data that teaches the model to manufacture request/epoch/tick identity.

## Collection guidance

Prefer source-linked captures:

- exact Cybersand commit;
- map/fixture ID;
- scenario seed;
- screenshot generation method;
- actor/world observation tick when available;
- expected answer and reviewer;
- whether the label comes from authoritative fixture state or manual visual judgement.

Do not use gameplay recordings as though they were exact replay or complete training state; the Cybersand contracts explicitly distinguish copied actor/render context from authoritative controller state.
