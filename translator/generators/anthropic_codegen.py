from __future__ import annotations

import os
from typing import Optional


SYSTEM_PROMPT = (
    "You are a code generation assistant. "
    "The user will describe a game mechanic or app feature in plain English. "
    "Write clean, functional, beginner-friendly code with comments explaining each line. "
    "Make the code actually do what they described, not just a skeleton."
)


def generate_code_with_claude(
    prompt: str,
    target: str,
    mode: str = "gameplay",
    source_language: str = "english",
    model: str = "claude-haiku-4-5",
    max_tokens: int = 3000,
) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    try:
        from anthropic import Anthropic  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency/runtime dependent
        raise RuntimeError("anthropic package is required for Claude code generation") from exc

    client = Anthropic(api_key=api_key)
    user_prompt = (
        f"Target language/output: {target}.\n"
        f"Mode/context: {mode}.\n"
        f"User language: {source_language}.\n"
        "Return code only (no markdown fences).\n\n"
        f"User request:\n{prompt}"
    )

    response = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_prompt}],
    )

    chunks: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)

    output = "\n".join(chunks).strip()
    if not output:
        raise RuntimeError("Claude returned an empty response")
    return output
