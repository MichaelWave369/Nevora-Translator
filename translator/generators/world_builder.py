from __future__ import annotations

import json
import os
from typing import Any


WORLD_BUILDER_SYSTEM_PROMPT = (
    "You are a project builder and code generator. "
    "Given staged design inputs, generate a complete connected Python starter project. "
    "Return STRICT JSON only with keys: section_one, section_two, section_three, section_four, main. "
    "Each value must be complete runnable-oriented code text. "
    "Use clear comments and keep the sections coherent with each other."
)


LEGACY_WORLD_KEYS = {
    "section_one": "environment",
    "section_two": "characters",
    "section_three": "rules",
    "section_four": "events",
}


def build_structured_project_prompt(project_type: str, stages: list[tuple[str, str]]) -> str:
    if len(stages) != 4:
        raise ValueError("Exactly 4 stages are required")

    body_parts = [f"Project Type: {project_type}"]
    for idx, (title, value) in enumerate(stages, start=1):
        body_parts.append(f"{idx}) {title}:\n{value}")

    body_parts.append(
        "Output strict JSON with keys: section_one, section_two, section_three, section_four, main. "
        "The `main` section must connect and run the project logic described by all stages."
    )
    return "\n\n".join(body_parts)


def build_world_builder_prompt(
    environment: str,
    characters: str,
    rules: str,
    events: str,
) -> str:
    return build_structured_project_prompt(
        "Game World",
        [
            ("Environment", environment),
            ("Characters", characters),
            ("Rules", rules),
            ("Events", events),
        ],
    )


def parse_world_builder_response(raw: str) -> dict[str, str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("World Builder response was not valid JSON") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("World Builder response must be a JSON object")

    required = ["section_one", "section_two", "section_three", "section_four", "main"]
    if all(k in payload for k in required):
        result: dict[str, str] = {}
        for key in required:
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(f"World Builder key '{key}' must be a non-empty string")
            result[key] = value
        return result

    # Backward compatibility with previous world-specific key format.
    legacy_required = ["environment", "characters", "rules", "events", "main"]
    if all(k in payload for k in legacy_required):
        return {
            "section_one": str(payload["environment"]),
            "section_two": str(payload["characters"]),
            "section_three": str(payload["rules"]),
            "section_four": str(payload["events"]),
            "main": str(payload["main"]),
        }

    missing = [k for k in required if k not in payload]
    raise RuntimeError(f"World Builder response missing keys: {', '.join(missing)}")


def _call_claude(prompt: str, model: str, max_tokens: int) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    try:
        from anthropic import Anthropic  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("anthropic package is required for World Builder generation") from exc

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        system=WORLD_BUILDER_SYSTEM_PROMPT,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    chunks: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)

    raw = "\n".join(chunks).strip()
    if not raw:
        raise RuntimeError("Claude returned an empty response for World Builder")
    return raw


def generate_structured_project_with_claude(
    project_type: str,
    stages: list[tuple[str, str]],
    model: str = "claude-haiku-4-5",
    max_tokens: int = 4000,
) -> dict[str, str]:
    prompt = build_structured_project_prompt(project_type, stages)
    raw = _call_claude(prompt, model=model, max_tokens=max_tokens)
    return parse_world_builder_response(raw)


def generate_world_with_claude(
    environment: str,
    characters: str,
    rules: str,
    events: str,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 4000,
) -> dict[str, str]:
    return generate_structured_project_with_claude(
        project_type="Game World",
        stages=[
            ("Environment", environment),
            ("Characters", characters),
            ("Rules", rules),
            ("Events", events),
        ],
        model=model,
        max_tokens=max_tokens,
    )
