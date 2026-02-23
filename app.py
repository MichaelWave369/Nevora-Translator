from __future__ import annotations

import json
import streamlit as st

from translator.core import EnglishToCodeTranslator


st.set_page_config(page_title="Nevora Translator", layout="wide")
st.title("Nevora Translator Web UI (MVP)")

with st.sidebar:
    planner_provider = st.selectbox("Planner", ["auto", "heuristic", "huggingface", "openai"], index=0)
    source_language = st.selectbox("Source language", ["english", "spanish", "french", "german", "portuguese"], index=0)
    mode = st.selectbox("Mode", ["gameplay", "automation", "video-processing", "web-backend"], index=0)
    target = st.selectbox("Target", ["python", "blueprint", "cpp", "csharp", "javascript", "gdscript"], index=0)
    use_rag_cache = st.checkbox("Use lattice RAG cache", value=False)
    show_guide = st.checkbox("Show assistant guide", value=False)  # hidden by default for non-technical users

prompt = st.text_area("Prompt", "When player presses space, jump and play sound", height=140)

if st.button("Generate"):
    translator = EnglishToCodeTranslator(planner_provider=planner_provider)
    output = translator.translate(
        prompt=prompt,
        target=target,
        mode=mode,
        source_language=source_language,
        use_rag_cache=use_rag_cache,
    )

    st.subheader("Generated Output")
    st.code(output)

    # Simple copy-code UX for non-technical users.
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

st.caption("Tip: run with `streamlit run app.py`.")
