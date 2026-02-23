# Nevora English-to-Code Translator

English idea → starter code for Python, Blueprint, C++, C#, JavaScript, and GDScript.

## Next phase (v8) implemented

### 1) Batch artifact pipelines
- Batch mode now supports per-item artifact generation.
- Added `--batch-artifact-dir` to store generated code files per batch item.
- Added `--batch-include-explain` to store explain-plan payloads per item.

### 2) Batch observability improvements
- Batch reports now include `generated_at` timestamps.
- Explain output now tracks both configured provider and runtime resolved provider.

### 3) Existing capabilities retained
- Scaffold verification + scaffold build checks.
- Explain plan console/file export.
- Strict safety mode and planner provider selection.

## Quick start

```bash
pip install -r requirements.txt
python -m translator.cli --target python --prompt "Create player jump on space" --mode gameplay --verify
```

## Batch mode with artifacts

Create `batch.jsonl`:

```json
{"prompt":"Create a player that can jump","target":"python"}
{"prompt":"Spawn enemy when timer reaches zero","target":"cpp"}
```

Run:

```bash
python -m translator.cli \
  --target python \
  --batch-input batch.jsonl \
  --batch-report artifacts/batch_report.json \
  --batch-artifact-dir artifacts/batch_items \
  --batch-include-explain
```

## Evaluation

```bash
python eval/run_eval.py
```
