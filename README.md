# Nevora English-to-Code Translator

English idea → starter code for Python, Blueprint, C++, C#, JavaScript, and GDScript.

## Next phase (v10) implemented

### 1) Batch verification passes
- Added per-item batch verification options:
  - generated output verification (`verify_output`)
  - scaffold build verification (`verify_scaffold_build`)
- CLI flags:
  - `--batch-verify-output`
  - `--batch-verify-build`

### 2) Verification quality gates
- Added CI gates for verification rates:
  - `--batch-min-verify-output-rate`
  - `--batch-min-verify-build-rate`
- Existing success-rate gate is still supported:
  - `--batch-min-success-rate`

### 3) Richer batch report metrics
- Batch report now includes:
  - `verify_output_ok`, `verify_build_ok`
  - `verify_output_rate`, `verify_build_rate`
  - `success_rate`, `target_counts`, `resolved_provider_counts`, `generated_at`

## Quick start

```bash
pip install -r requirements.txt
python -m translator.cli --target python --prompt "Create player jump on space" --mode gameplay --verify
```

## Batch mode with full quality gates

```bash
python -m translator.cli \
  --target python \
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
