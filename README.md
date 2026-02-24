# Nevora Translator (English/Multilingual-to-Code)

Nevora converts natural-language prompts into starter code for multiple targets (`python`, `blueprint`, `cpp`, `csharp`, `javascript`, `gdscript`) with optional AI planners, batch workflows, and engine asset integration.

## Next phase (v24) implemented

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
- Base dependencies now include `joblib`, `faiss-cpu`, `streamlit`, and runtime provider SDKs.
- Optional LLM/tooling extras in `requirements-llm.txt` include `openai`, `transformers`, `torch`, `accelerate`, and `mypy` for gradual type checking.

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


### 12) World Builder mode (guided 4-stage flow)
- Added a dedicated **World Builder** mode in Streamlit.
- Guides users through 4 staged inputs with clickable examples:
  1. Environment
  2. Characters
  3. Rules
  4. Events
- Clicking an example auto-fills each stage, while manual editing remains available.
- **Build My World** sends all four stages together to Claude and returns connected code sections:
  - Environment
  - Characters
  - Rules
  - Events
  - Main (runnable pygame starter)
- Includes one-click copy for the full multi-file starter project output.


### 13) Multi-template World Builder project types
- World Builder now supports selectable project types:
  - 🎮 Game World
  - 🏪 Small Business App
  - 📱 Personal Tool
  - 🤖 Automation Bot
  - 📊 Dashboard
- Each project type has its own 4-stage guided flow with clickable examples.
- For example:
  - Small Business App stages:
    1. What's your business?
    2. What do you need to manage?
    3. Who uses it?
    4. What should it do automatically?
  - Personal Tool stages:
    1. What problem are you solving?
    2. What information do you want to track?
    3. How often do you use it?
    4. What should it show or tell you?
- Build button now adapts (`Build My World` or `Build My App`) and sends all 4 stages to Claude as one structured request.


### 14) One-click project download + beginner code explanations
- After generation, users can click **Download My Project** to get a ready-to-open `.zip` package.
- The package includes:
  - generated code file(s),
  - `README.txt` written in plain English,
  - `requirements.txt`,
  - `run.bat` (Windows) and `run.sh` (Mac/Linux).
- Added **Explain My Code** below outputs so Claude can explain the generated code like a teacher for first-time coders.

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


## Optional static type checking (mypy)

```bash
pip install -r requirements-llm.txt
mypy translator
```
