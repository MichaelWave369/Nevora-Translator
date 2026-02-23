# Nevora Translator (English/Multilingual-to-Code)

Nevora converts natural-language prompts into starter code for multiple targets (`python`, `blueprint`, `cpp`, `csharp`, `javascript`, `gdscript`) with optional AI planners, batch workflows, and engine asset integration.

## Next phase (v19) implemented

### 1) MVP free-LLM planner path (Hugging Face)
- Added `HuggingFaceSemanticPlanner` (`translator/planners/huggingface_planner.py`) using HF text2text generation.
- Planner providers now support:
  - `auto`
  - `heuristic`
  - `huggingface`
  - `openai`

### 2) Core code + examples populated
- Core translation logic remains in `translator/core.py` and CLI in `translator/cli.py`.
- Added concrete example input/output pairs in `examples/pairs/`.

### 3) Dependencies expanded
- Base dependencies now include `joblib`, `faiss-cpu`, and `streamlit`.
- Optional LLM dependencies include `openai`, `transformers`, `torch`, `accelerate`.

### 4) Evaluation upgraded
- `eval/run_eval.py` now reports:
  - structure checks,
  - intent coverage,
  - determinism,
  - syntax validity,
  - BLEU-like similarity,
  - buildability.

### 5) Enhancements
- Streamlit web UI MVP added in `app.py`.
- Godot target support remains available via `gdscript` output and scaffold support.

### 6) Community + OSS foundations
- Added `CONTRIBUTING.md`.
- Added `LICENSE` (MIT).


### 8) Non-technical user UX improvements
- Assistant guide is now hidden by default in the web UI to keep first-time usage simple.
- Generated code templates are more complete and beginner-friendly with inline comments explaining each section.
- Added a simple **Copy Code** button in the Streamlit UI for easy copy/paste into editors.

## Installation

```bash
pip install -r requirements.txt
# optional AI planners
pip install -r requirements-llm.txt
```

## CLI quick start

```bash
python -m translator.cli \
  --planner-provider huggingface \
  --target python \
  --prompt "When player presses space, jump and play sound"
```

## Streamlit web UI

```bash
streamlit run app.py
```

## Prompt engineering tips (edge-case handling)
To avoid vague outputs:
- Include **entity + action + condition**: e.g., “When player health reaches zero, play death animation and respawn after 3 seconds.”
- Name environment constraints: “For Unity C#, avoid external packages.”
- Ask for expected outputs explicitly: “Also log result and update UI text.”
- For multilingual prompts, include key technical terms in English if needed.

## Asset manager integration (Unreal/Unity)
Use engine-aware generation against your own asset library:

```bash
python -m translator.cli \
  --target csharp \
  --engine unity \
  --asset-library ./my_assets.json \
  --prompt "Player jump controller" \
  --export-engine-manifest artifacts/unity_manifest.json
```

## Batch + assistant optimization features
- Auto swarm tuning with `--swarm-workers 0`
- Swarm benchmark with `--benchmark-swarm --benchmark-workers 1,2,4`
- Assistant guidance with `--assistant-guide`
- Report advice with `--assistant-report-advice`
- Runbook output with `--assistant-runbook-file`

## Evaluation

```bash
python eval/run_eval.py
```
