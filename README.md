# Nevora Translator (English/Multilingual-to-Code)

Nevora converts natural-language prompts into starter code for multiple targets (`python`, `blueprint`, `cpp`, `csharp`, `javascript`, `gdscript`) with optional AI planners, batch workflows, and engine asset integration.

## Next phase (v22) implemented

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


### 9) Claude-powered real code generation in Streamlit
- The Generate flow now supports direct Anthropic Claude generation using **`claude-haiku-4-5`**.
- It reads API credentials from environment variable: `ANTHROPIC_API_KEY`.
- System message enforces functional, beginner-friendly, commented code generation from plain-English requests.
- Added a fallback engine (`nevora-template-fallback`) so deployments can still run without API credentials.


### 10) Multi-provider generation options (Claude default)
- Streamlit now supports provider choices:
  - **Claude** (default, `claude-haiku-4-5`)
  - OpenAI
  - Grok (xAI)
  - Gemini
  - Ollama (local)
  - Nevora template fallback
- Environment variables:
  - `ANTHROPIC_API_KEY` (Claude)
  - `OPENAI_API_KEY` (OpenAI)
  - `XAI_API_KEY` (Grok)
  - `GEMINI_API_KEY` or `GOOGLE_API_KEY` (Gemini)
  - `OLLAMA_BASE_URL` (optional local endpoint)


### 11) Visual category menu for non-technical users
- Added clickable visual category cards before the prompt box:
  - 🎮 Game Mechanics
  - 🌍 World Building
  - 🤖 AI & NPCs
  - 🎨 Visual Effects
  - 🔧 Automation
  - 🌐 Web & Apps
- Selecting a category shows a second row of example prompt buttons.
- Clicking an example auto-fills the prompt box.
- Manual typing still works through the same prompt text area.

## Installation

```bash
pip install -r requirements.txt
# optional expanded AI planner stack
pip install -r requirements-llm.txt
export ANTHROPIC_API_KEY="your_key_here"
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
# choose generation engine: Claude default, or OpenAI/Grok/Gemini/Ollama
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
