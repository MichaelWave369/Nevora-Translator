from __future__ import annotations

import streamlit as st

from translator.core import EnglishToCodeTranslator
from translator.generators.multi_codegen import generate_code


st.set_page_config(page_title="Nevora Translator", layout="wide")
st.title("Nevora Translator Web UI")

with st.sidebar:
    generation_engine = st.selectbox(
        "Generation engine",
        [
            "claude-haiku-4-5 (default)",
            "openai",
            "grok",
            "gemini",
            "ollama (local)",
            "nevora-template-fallback",
        ],
        index=0,
        help="Claude is default. You can switch to OpenAI, Grok, Gemini, or local Ollama.",
    )
    source_language = st.selectbox("Source language", ["english", "spanish", "french", "german", "portuguese"], index=0)
    mode = st.selectbox("Mode", ["gameplay", "automation", "video-processing", "web-backend"], index=0)
    target = st.selectbox("Target", ["python", "blueprint", "cpp", "csharp", "javascript", "gdscript"], index=0)
    model_name = st.text_input("Model override (optional)", "")
    ollama_base_url = st.text_input("Ollama base URL", "http://localhost:11434")
    use_rag_cache = st.checkbox("Use lattice RAG cache (fallback engine only)", value=False)
    show_guide = st.checkbox("Show assistant guide", value=False)

prompt = st.text_area("Prompt", "When player presses space, jump and play sound", height=140)

if st.button("Generate"):
    translator = EnglishToCodeTranslator(planner_provider="auto")

    provider_map = {
        "claude-haiku-4-5 (default)": "claude",
        "openai": "openai",
        "grok": "grok",
        "gemini": "gemini",
        "ollama (local)": "ollama",
    }

    if generation_engine in provider_map:
        try:
            output = generate_code(
                provider=provider_map[generation_engine],
                prompt=prompt,
                target=target,
                mode=mode,
                source_language=source_language,
                model=model_name or None,
                ollama_base_url=ollama_base_url,
            )
        except Exception as exc:
            st.error(
                f"{generation_engine} generation failed: {exc}. "
                "Set required API keys and dependencies, or use fallback engine."
            )
            st.stop()
    else:
        output = translator.translate(
            prompt=prompt,
            target=target,
            mode=mode,
            source_language=source_language,
            use_rag_cache=use_rag_cache,
        )

    st.subheader("Generated Output")
    st.code(output)

    st.markdown("### Copy code")
    st.components.v1.html(
        f"""
        <textarea id=\"nevora_code\" style=\"width:100%;height:180px;\">{output}</textarea>
        <button onclick=\"navigator.clipboard.writeText(document.getElementById('nevora_code').value)\" style=\"margin-top:8px;\">Copy Code</button>
        """,
        height=240,
    )

    if show_guide:
        guide = translator.assistant_guide(
            prompt=prompt,
            target=target,
            mode=mode,
            source_language=source_language,
        )
        st.subheader("Assistant Guide")
        st.json(guide)

st.caption(
    "Tip: run with `streamlit run app.py`. Keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY, GEMINI_API_KEY; local Ollama uses OLLAMA_BASE_URL."
)
