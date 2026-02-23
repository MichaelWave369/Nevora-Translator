# Nevora English-to-Code Translator

Translate natural-language ideas into starter code for Python, Blueprint, C++, C#, JavaScript, and GDScript.

## Next phase (v16) implemented

### 1) Full optimization pass
- Added fast plan reuse with an internal plan cache (`_plan_cache`) to avoid repeated planning work for identical prompt/mode combinations.
- Added per-item latency tracking (`elapsed_ms`) and report aggregation (`avg_elapsed_ms`) for batch profiling.

### 2) AI swarm + speed features retained
- Parallel batch execution with `--swarm-workers`.
- Optional 12x12x12x12 lattice RAG acceleration with `--enable-rag-cache`.
- VM-like sandbox execution with `--sandbox-command`.

### 3) Unreal/Unity asset manager integration
- New engine-aware generation path that can use a **user asset library JSON**:
  - `translate_with_asset_library(...)`
  - `export_engine_asset_manifest(...)`
- Supports `engine=unreal` and `engine=unity`.
- Asset matching uses prompt/token overlap and tag weighting to select best assets from the user’s library.

### 4) CLI support for engine asset flows
- Added:
  - `--engine {unreal,unity}`
  - `--asset-library <path>`
  - `--asset-budget <n>`
  - `--export-engine-manifest <path>`
- When engine/library args are provided, CLI prints selected assets and can emit engine manifest JSON.


### 5) Built-in AI assistant guidance
- Added assistant guidance API to help users operate the system effectively:
  - `assistant_guide(...)` returns prompt diagnostics, intent preview, optimization tips, quality tips, and a suggested CLI command.
  - `warm_plan_cache(...)` preloads generation plans to reduce first-hit latency for known prompt sets.
- New CLI options:
  - `--assistant-guide`
  - `--assistant-report <batch_report.json>`
  - `--warm-cache-file <prompts.txt>`
  - `--assistant-report-advice`


### 6) Assistant-driven optimization advisor
- Added report-aware optimization advisor:
  - `analyze_batch_report(...)` summarizes speed/quality health and returns optimization recommendations.
  - `suggest_swarm_workers(...)` auto-suggests concurrency based on batch size and available CPU.
- Added CLI support:
  - `--assistant-report-advice` (requires `--assistant-report`)
  - `--swarm-workers 0` now means auto-select workers.
- Batch reports now include `p95_elapsed_ms` for tail-latency visibility.

## Asset library format

```json
{
  "unreal": [
    {"id": "SM_Enemy", "name": "Enemy Mesh", "tags": ["enemy", "mesh"], "path": "/Game/Meshes/SM_Enemy"}
  ],
  "unity": [
    {"id": "PlayerPrefab", "name": "Player Prefab", "tags": ["player", "jump"], "path": "Assets/Prefabs/Player.prefab"}
  ]
}
```

## Unreal asset-aware example

```bash
python -m translator.cli \
  --target python \
  --engine unreal \
  --asset-library ./my_assets.json \
  --asset-budget 5 \
  --prompt "Spawn enemy and play hit sound" \
  --export-engine-manifest artifacts/unreal_manifest.json
```

## Unity asset-aware example

```bash
python -m translator.cli \
  --target csharp \
  --engine unity \
  --asset-library ./my_assets.json \
  --prompt "Player jump controller" \
  --export-engine-manifest artifacts/unity_manifest.json
```

## Evaluation

```bash
python eval/run_eval.py
```
