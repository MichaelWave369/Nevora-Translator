# Nevora English-to-Code Translator

Translate ideas into code for Python, Blueprint, C++, C#, JavaScript, and GDScript.

## Next phase (v11) implemented

### 1) Multilingual prompt input
- Added `source_language` support so prompts can be written in:
  - `english`
  - `spanish`
  - `french`
  - `german`
  - `portuguese`
- Prompts are normalized into English before planning so existing target renderers work consistently.

### 2) Multilingual explainability and reporting
- Explain-plan now includes:
  - `source_language`
  - `normalized_prompt`
- Batch reports now include `source_language_counts` for observability across mixed-language runs.

### 3) Existing v10 batch verification/gates retained
- `--batch-verify-output`, `--batch-verify-build`
- `--batch-min-success-rate`, `--batch-min-verify-output-rate`, `--batch-min-verify-build-rate`

## Quick start

```bash
pip install -r requirements.txt
python -m translator.cli --target python --prompt "Create player jump on space" --mode gameplay --verify
```

## Spanish input example

```bash
python -m translator.cli \
  --target python \
  --source-language spanish \
  --prompt "Cuando jugador saltar"
```

## Batch mode with language mix + quality gates

```bash
python -m translator.cli \
  --target python \
  --source-language english \
  --batch-input batch.jsonl \
  --batch-report artifacts/batch_report.json \
  --batch-artifact-dir artifacts/batch_items \
  --batch-include-explain \
  --batch-verify-output \
  --batch-verify-build \
  --batch-min-success-rate 0.90 \
  --batch-min-verify-output-rate 0.90 \
  --batch-min-verify-build-rate 0.90
```

## Evaluation

```bash
python eval/run_eval.py
```
