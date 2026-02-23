# Nevora English-to-Code Translator

Translate natural-language ideas into starter code for Python, Blueprint, C++, C#, JavaScript, and GDScript.

## Next phase (v12) implemented

### 1) Multilingual text prompts
- Source prompt languages supported:
  - `english`
  - `spanish`
  - `french`
  - `german`
  - `portuguese`
- Non-English prompts are token-normalized into English before planning/rendering so all existing targets continue to work.

### 2) Audio input support (multilingual)
- New single-run CLI flag: `--audio-input`.
- You can pass:
  - a transcript-like file (`.txt`, `.md`, `.prompt`) for deterministic local workflows, or
  - an audio file when optional speech dependencies are installed.
- Language control: `--source-language` is used for transcription language handling.

### 3) Audio output support (multilingual)
- New single-run CLI flags:
  - `--audio-output`
  - `--audio-output-language`
- Translator attempts TTS output using optional runtime dependency (`pyttsx3`).
- If TTS is unavailable, it safely falls back to writing a transcript sidecar file (`<audio_output_path>.txt`).

### 4) Existing batch quality gates retained
- `--batch-verify-output`, `--batch-verify-build`
- `--batch-min-success-rate`, `--batch-min-verify-output-rate`, `--batch-min-verify-build-rate`
- `--batch-fail-fast`

## Quick start

```bash
pip install -r requirements.txt
python -m translator.cli --target python --prompt "Create player jump on space" --mode gameplay --verify
```

## Multilingual text example (Spanish)

```bash
python -m translator.cli \
  --target python \
  --source-language spanish \
  --prompt "Cuando jugador saltar"
```

## Audio input example (transcript file)

```bash
python -m translator.cli \
  --target python \
  --source-language spanish \
  --audio-input /tmp/nevora_audio_prompt.txt \
  --explain-plan
```

## Audio output example

```bash
python -m translator.cli \
  --target python \
  --prompt "Create player jump on space" \
  --audio-output artifacts/speech.wav \
  --audio-output-language english
```

## Optional audio dependencies

```bash
pip install SpeechRecognition pyttsx3
```

> Note: audio support is best-effort in constrained environments. If speech dependencies or system TTS engines are unavailable, use transcript input and/or transcript output fallback.

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
