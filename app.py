from __future__ import annotations

import streamlit as st

from translator.core import EnglishToCodeTranslator
from translator.generators.anthropic_codegen import generate_code_with_claude


st.set_page_config(page_title="Nevora Translator", layout="wide")
st.title("Nevora Translator Web UI")

with st.sidebar:
    generation_engine = st.selectbox(
        "Generation engine",
        ["claude-haiku-4-5", "nevora-template-fallback"],
        index=0,
        help="Use Claude for real functional code generation. Fallback is the internal template pipeline.",
    )
    source_language = st.selectbox("Source language", ["english", "spanish", "french", "german", "portuguese"], index=0)
    mode = st.selectbox("Mode", ["gameplay", "automation", "video-processing", "web-backend"], index=0)
    target = st.selectbox("Target", ["python", "blueprint", "cpp", "csharp", "javascript", "gdscript"], index=0)
    use_rag_cache = st.checkbox("Use lattice RAG cache (fallback engine only)", value=False)
    show_guide = st.checkbox("Show assistant guide", value=False)

prompt = st.text_area("Prompt", "When player presses space, jump and play sound", height=140)

if st.button("Generate"):
    translator = EnglishToCodeTranslator(planner_provider="auto")

    if generation_engine == "claude-haiku-4-5":
        try:
            output = generate_code_with_claude(
                prompt=prompt,
                target=target,
                mode=mode,
                source_language=source_language,
                model="claude-haiku-4-5",
            )
        except Exception as exc:
            st.error(
                f"Claude generation failed: {exc}. "
                "Set ANTHROPIC_API_KEY and install anthropic dependencies, or use fallback engine."
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

st.caption("Tip: run with `streamlit run app.py`. For Claude, set ANTHROPIC_API_KEY.")
