# Contributing to Nevora Translator

Thanks for your interest in contributing.

## Development setup
1. Create a virtual environment.
2. Install base deps: `pip install -r requirements.txt`.
3. Optional LLM deps: `pip install -r requirements-llm.txt`.
4. Run tests: `python -m pytest -q`.

## Contribution flow
1. Open an issue describing the problem/feature.
2. Create a branch and implement a focused change.
3. Add/adjust tests.
4. Update docs/examples when behavior changes.
5. Submit a PR with motivation, description, and testing notes.

## Code guidelines
- Keep deterministic fallbacks for optional AI integrations.
- Prefer typed, small functions over large monoliths.
- Add CLI flags only when they serve repeatable workflows.

## Community
If you share the project publicly, please be respectful and constructive.
Suggested communities: ML engineering forums, game-dev communities, and open-source Python channels.
