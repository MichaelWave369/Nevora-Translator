# Nevora Translator

Nevora Translator turns natural-language prompts into starter code for multiple targets (`python`, `blueprint`, `cpp`, `csharp`, `javascript`, `gdscript`) with both:
- a CLI workflow, and
- a Streamlit web app.

It supports multi-provider generation (Claude/OpenAI/Grok/Gemini/Ollama) **and** an offline-safe template fallback path that works without API keys.

## Release status

Current version: **0.1.0rc1** (first public release candidate).

## Quick install

### Core install (fallback mode, no cloud API required)
```bash
pip install -e .
```

### UI install
```bash
pip install -e .[ui]
```

### Provider SDK install (Claude/OpenAI/Grok/Gemini)
```bash
pip install -e .[providers]
```

### Optional planner stack (Hugging Face planners)
```bash
pip install -e .[planners]
```

### Dev/test install
```bash
pip install -e .[dev]
```

## CLI quick start

```bash
nevora-translator \
  --target python \
  --prompt "When player presses space, jump and play sound"
```

or

```bash
python -m translator.cli \
  --target python \
  --prompt "When player presses space, jump and play sound"
```

## Streamlit quick start

```bash
pip install -e .[ui]
streamlit run app.py
```

## Provider environment variables

Set only what you use:

- `ANTHROPIC_API_KEY` for Claude
- `OPENAI_API_KEY` for OpenAI
- `XAI_API_KEY` for Grok
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` for Gemini
- `OLLAMA_BASE_URL` for local Ollama (optional, defaults to `http://localhost:11434`)
- `GITHUB_TOKEN` for Streamlit GitHub export

Copy `.env.example` to `.env` and fill values as needed.

## Fallback mode (no API keys)

Fallback stays first-class:

- CLI translation works without provider credentials.
- World Builder can run via provider `nevora-template-fallback`.
- Missing provider keys return friendly runtime errors so users can switch providers instead of crashing.

## Ollama mode

Ollama is optional. To use local generation:

1. Run Ollama locally.
2. Set `OLLAMA_BASE_URL` if non-default.
3. Choose `ollama (local)` in Streamlit.

If unavailable, Nevora surfaces a clear error and you can switch to fallback mode.

## Testing

```bash
pip install -e .[dev]
pytest
```

## Release checklist

- [ ] `pip install -e .` succeeds.
- [ ] `pytest` passes.
- [ ] CLI smoke: `nevora-translator --target python --prompt "Create jump"`.
- [ ] Streamlit launch smoke: `streamlit run app.py`.
- [ ] Fallback generation works with no API keys.
- [ ] Provider failures are graceful with helpful messages.
- [ ] Version in `pyproject.toml` and `translator/_version.py` matches.
- [ ] `CHANGELOG.md` updated.
- [ ] Tag pushed (`vX.Y.Z` or pre-release tag) and release workflow artifacts attached.

## Known limitations

- Cloud provider quality and latency vary by model and account limits.
- Hugging Face planner dependencies are heavy; install only when needed.
- World Builder multi-file output format depends on provider returning valid JSON.

## License

MIT (see `LICENSE`).
