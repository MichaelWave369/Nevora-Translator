from __future__ import annotations

import os


SYSTEM_PROMPT = (
    "You are a code generation assistant. "
    "The user will describe a game mechanic or app feature in plain English. "
    "Write clean, functional, beginner-friendly code with comments explaining each line. "
    "Make the code actually do what they described, not just a skeleton."
)


EXPLAIN_SYSTEM_PROMPT = (
    "You are a friendly programming teacher. "
    "Explain generated code in very plain English for someone who has never coded before. "
    "Avoid jargon. Use short sections and bullet points."
)


def _anthropic_client() -> "object":
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    try:
        from anthropic import Anthropic  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency/runtime dependent
        raise RuntimeError("anthropic package is required for Claude generation") from exc

    return Anthropic(api_key=api_key)


def _extract_text(response: object) -> str:
    chunks: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    output = "\n".join(chunks).strip()
    if not output:
        raise RuntimeError("Claude returned an empty response")
    return output


def generate_code_with_claude(
    prompt: str,
    target: str,
    mode: str = "gameplay",
    source_language: str = "english",
    model: str = "claude-haiku-4-5",
    max_tokens: int = 3000,
) -> str:
    client = _anthropic_client()
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
    return _extract_text(response)


def explain_code_with_claude(
    code: str,
    user_goal: str,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 1800,
) -> str:
    client = _anthropic_client()
    user_prompt = (
        f"User goal:\n{user_goal}\n\n"
        f"Generated code:\n{code}\n\n"
        "Please explain what each major section does, how data flows, and what to change safely first."
    )

    response = client.messages.create(
        model=model,
        system=EXPLAIN_SYSTEM_PROMPT,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_prompt}],
    )
    output = _extract_text(response)
    if not output:
        raise RuntimeError("Claude returned an empty explanation")
    return output
