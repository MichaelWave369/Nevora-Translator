# Nevora English-to-Code Translator

A small, extensible translator that converts plain-English feature descriptions into starter code for:

- **Python**
- **Unreal Engine Blueprint (pseudo-graph format)**

It is designed as an MVP for idea-to-code workflows (games, scripts, and automation concepts).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m translator.cli --target python --prompt "Create a player that can jump when space is pressed"
```

### Unreal Blueprint output

```bash
python -m translator.cli --target blueprint --prompt "When health is zero, play death animation and disable input"
```

## What this does

1. Detects intent from English using lightweight keyword extraction.
2. Generates deterministic code templates based on detected actions/conditions/entities.
3. Uses target-specific renderers:
   - Python renderer outputs runnable Python starter code.
   - Blueprint renderer outputs node-flow style pseudocode that can be mapped to UE Blueprints.

## Future improvements

- LLM-backed semantic planner for deeper interpretation.
- More targets (C++, C#, JavaScript, GDScript).
- Direct `.uasset` graph export integration for Unreal.
- Dataset-driven fine-tuning and benchmarking.
