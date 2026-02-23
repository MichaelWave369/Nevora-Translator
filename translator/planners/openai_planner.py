from __future__ import annotations

import json
import os
from typing import Any

from translator.models import ParsedIntent


class OpenAISemanticPlanner:
    def __init__(self, model: str = "gpt-4.1-mini", retries: int = 2) -> None:
        self.model = model
        self.retries = retries

    def _validate_payload(self, payload: Any) -> ParsedIntent:
        if not isinstance(payload, dict):
            raise ValueError("Planner payload must be a JSON object")

        def as_list(name: str, fallback: str) -> list[str]:
            value = payload.get(name)
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                return [fallback]
            return value or [fallback]

        return ParsedIntent(
            entities=as_list("entities", "system"),
            actions=as_list("actions", "process"),
            conditions=as_list("conditions", "always"),
            outputs=as_list("outputs", "state"),
        )

    def plan(self, prompt: str, mode: str = "gameplay") -> ParsedIntent:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        instruction = (
            "Return strict JSON only with keys entities/actions/conditions/outputs as string arrays. "
            f"Mode: {mode}."
        )
        errors: list[str] = []
        for _ in range(self.retries + 1):
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = getattr(response, "output_text", "")
            try:
                payload = json.loads(raw)
                return self._validate_payload(payload)
            except Exception as exc:
                errors.append(str(exc))
        raise RuntimeError(f"OpenAI planner failed after retries: {'; '.join(errors)}")
