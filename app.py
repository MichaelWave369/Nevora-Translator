from __future__ import annotations

import streamlit as st

from translator.core import EnglishToCodeTranslator
from translator.generators.multi_codegen import generate_code


CATEGORY_EXAMPLES = {
    "game_mechanics": {
        "label": "🎮 Game Mechanics",
        "examples": [
            "Player jumps when spacebar pressed",
            "Enemy chases player",
            "Collect item to gain points",
            "Health bar decreases on damage",
        ],
    },
    "world_building": {
        "label": "🌍 World Building",
        "examples": [
            "Generate day-night cycle with smooth lighting transition",
            "Spawn weather zones with rain and fog",
            "Open portal when player enters ancient shrine",
            "Load biome-specific ambient sound by region",
        ],
    },
    "ai_npcs": {
        "label": "🤖 AI & NPCs",
        "examples": [
            "NPC patrols between waypoints and reacts to noise",
            "Friendly companion follows player and heals at low health",
            "Shopkeeper dialogue tree with item purchase",
            "Boss switches attack phase at 50 percent health",
        ],
    },
    "visual_effects": {
        "label": "🎨 Visual Effects",
        "examples": [
            "Add screen shake and flash on explosion",
            "Trail effect behind fast-moving projectile",
            "Glow outline when interactable object is nearby",
            "Particle burst when enemy is defeated",
        ],
    },
    "automation": {
        "label": "🔧 Automation",
        "examples": [
            "Rename files in folder by date prefix",
            "Validate CSV rows and export error report",
            "Schedule daily backup and notification",
            "Sync assets from source folder to build folder",
        ],
    },
    "web_apps": {
        "label": "🌐 Web & Apps",
        "examples": [
            "User login with JWT and role-based access",
            "REST endpoint to create and list tasks",
            "Rate limit API requests per user",
            "Upload image and generate thumbnail preview",
        ],
    },
}


st.set_page_config(page_title="Nevora Translator", layout="wide")
st.title("Nevora Translator Web UI")

st.markdown(
    """
    <style>
    .nevora-card-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .nevora-help {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-top: -0.25rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "selected_category" not in st.session_state:
    st.session_state.selected_category = "game_mechanics"
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = "When player presses space, jump and play sound"

st.markdown('<div class="nevora-card-title">Choose a category</div>', unsafe_allow_html=True)
st.markdown('<div class="nevora-help">Pick a category card, then click an example to auto-fill the prompt.</div>', unsafe_allow_html=True)

category_keys = list(CATEGORY_EXAMPLES.keys())
for row_start in range(0, len(category_keys), 3):
    row_keys = category_keys[row_start : row_start + 3]
    cols = st.columns(3)
    for idx, key in enumerate(row_keys):
        label = CATEGORY_EXAMPLES[key]["label"]
        selected = st.session_state.selected_category == key
        button_label = f"✅ {label}" if selected else label
        if cols[idx].button(button_label, use_container_width=True, key=f"category_{key}"):
            st.session_state.selected_category = key

active_category = st.session_state.selected_category
st.markdown("#### Example prompts")
example_cols = st.columns(4)
for i, example in enumerate(CATEGORY_EXAMPLES[active_category]["examples"]):
    if example_cols[i % 4].button(example, use_container_width=True, key=f"example_{active_category}_{i}"):
        st.session_state.prompt_input = example

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

prompt = st.text_area("Prompt", key="prompt_input", height=160)

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
        <textarea id="nevora_code" style="width:100%;height:180px;">{output}</textarea>
        <button onclick="navigator.clipboard.writeText(document.getElementById('nevora_code').value)" style="margin-top:8px;">Copy Code</button>
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
