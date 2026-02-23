from __future__ import annotations

import json
from typing import Any

from translator.models import ParsedIntent


class HuggingFaceSemanticPlanner:
    """Free-model planner path using Hugging Face inference locally.

    Uses a small text2text model when available and falls back by raising
    RuntimeError so callers can degrade to other planners.
    """

    def __init__(self, model: str = "google/flan-t5-base", max_new_tokens: int = 200) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens

    def _validate_payload(self, payload: Any) -> ParsedIntent:
        if not isinstance(payload, dict):
            raise ValueError("Planner payload must be a JSON object")

        def as_list(name: str, fallback: str) -> list[str]:
            value = payload.get(name)
            if not isinstance(value, list):
                return [fallback]
            normalized = [str(v) for v in value if str(v).strip()]
            return normalized or [fallback]

        return ParsedIntent(
            entities=as_list("entities", "system"),
            actions=as_list("actions", "process"),
            conditions=as_list("conditions", "always"),
            outputs=as_list("outputs", "state"),
        )

    def plan(self, prompt: str, mode: str = "gameplay") -> ParsedIntent:
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError("transformers is required for HuggingFaceSemanticPlanner") from exc

        system_prompt = (
            "Extract intent as strict JSON with keys entities/actions/conditions/outputs as arrays of strings. "
            f"Mode: {mode}. Prompt: {prompt}"
        )
        generator = pipeline("text2text-generation", model=self.model)
        raw = generator(system_prompt, max_new_tokens=self.max_new_tokens)[0]["generated_text"]

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Hugging Face planner returned non-JSON output: {raw[:200]}") from exc

        return self._validate_payload(payload)
