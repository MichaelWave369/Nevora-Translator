from __future__ import annotations

import os
from typing import Optional


from .anthropic_codegen import SYSTEM_PROMPT


def _build_user_prompt(prompt: str, target: str, mode: str, source_language: str) -> str:
    return (
        f"Target language/output: {target}.\n"
        f"Mode/context: {mode}.\n"
        f"User language: {source_language}.\n"
        "Return code only (no markdown fences).\n\n"
        f"User request:\n{prompt}"
    )


def generate_code_with_openai(
    prompt: str,
    target: str,
    mode: str = "gameplay",
    source_language: str = "english",
    model: str = "gpt-4o-mini",
    max_tokens: int = 3000,
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for OpenAI code generation") from exc

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(prompt, target, mode, source_language)},
        ],
        max_output_tokens=max_tokens,
    )
    output = getattr(response, "output_text", "").strip()
    if not output:
        raise RuntimeError("OpenAI returned an empty response")
    return output


def generate_code_with_grok(
    prompt: str,
    target: str,
    mode: str = "gameplay",
    source_language: str = "english",
    model: str = "grok-2-latest",
    max_tokens: int = 3000,
) -> str:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY is not set")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for Grok code generation") from exc

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(prompt, target, mode, source_language)},
        ],
        max_output_tokens=max_tokens,
    )
    output = getattr(response, "output_text", "").strip()
    if not output:
        raise RuntimeError("Grok returned an empty response")
    return output


def generate_code_with_gemini(
    prompt: str,
    target: str,
    mode: str = "gameplay",
    source_language: str = "english",
    model: str = "gemini-1.5-flash",
) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")

    try:
        import google.generativeai as genai  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("google-generativeai package is required for Gemini code generation") from exc

    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model_name=model, system_instruction=SYSTEM_PROMPT)
    result = gm.generate_content(_build_user_prompt(prompt, target, mode, source_language))
    output = getattr(result, "text", "").strip()
    if not output:
        raise RuntimeError("Gemini returned an empty response")
    return output


def generate_code_with_ollama(
    prompt: str,
    target: str,
    mode: str = "gameplay",
    source_language: str = "english",
    model: str = "llama3.2",
    base_url: Optional[str] = None,
) -> str:
    api_base = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    try:
        import requests  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("requests package is required for Ollama generation") from exc
    payload = {
        "model": model,
        "prompt": f"{SYSTEM_PROMPT}\n\n{_build_user_prompt(prompt, target, mode, source_language)}",
        "stream": False,
    }
    try:
        response = requests.post(f"{api_base}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        output = response.json().get("response", "").strip()
    except Exception as exc:
        raise RuntimeError(f"Ollama generation failed: {exc}") from exc

    if not output:
        raise RuntimeError("Ollama returned an empty response")
    return output


def generate_code(
    provider: str,
    prompt: str,
    target: str,
    mode: str = "gameplay",
    source_language: str = "english",
    model: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
) -> str:
    normalized = provider.lower().strip()
    if normalized == "claude":
        from .anthropic_codegen import generate_code_with_claude

        return generate_code_with_claude(prompt, target, mode=mode, source_language=source_language, model=model or "claude-haiku-4-5")
    if normalized == "openai":
        return generate_code_with_openai(prompt, target, mode=mode, source_language=source_language, model=model or "gpt-4o-mini")
    if normalized == "grok":
        return generate_code_with_grok(prompt, target, mode=mode, source_language=source_language, model=model or "grok-2-latest")
    if normalized == "gemini":
        return generate_code_with_gemini(prompt, target, mode=mode, source_language=source_language, model=model or "gemini-1.5-flash")
    if normalized == "ollama":
        return generate_code_with_ollama(prompt, target, mode=mode, source_language=source_language, model=model or "llama3.2", base_url=ollama_base_url)
    raise ValueError(f"Unsupported generation provider '{provider}'")
