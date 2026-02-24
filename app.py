from __future__ import annotations

import html

import streamlit as st

from translator.core import EnglishToCodeTranslator
from translator.generators.anthropic_codegen import explain_code_with_claude
from translator.generators.multi_codegen import generate_code
from translator.generators.project_packager import package_single_file_project, package_world_builder_project
from translator.generators.world_builder import generate_structured_project


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

PROJECT_TYPE_CONFIG = {
    "game_world": {
        "label": "🎮 Game World",
        "build_button": "Build My World",
        "stage_titles": [
            "Environment — what does the world look like?",
            "Characters — who or what is in the world?",
            "Rules — how does the world work?",
            "Events — what happens?",
        ],
        "examples": [
            [
                "Mystic forest with glowing trees and floating ruins",
                "Space station orbiting a dying star",
                "Ancient dungeon with lava rivers",
                "Cyberpunk city with neon rain",
            ],
            [
                "Player explorer, shadow enemies, merchant NPC",
                "Pilot hero, drone swarms, station engineer NPC",
                "Knight player, skeleton enemies, trapped villagers",
                "Hacker player, corporate bots, resistance contact NPC",
            ],
            [
                "Low gravity, health system, score for relics, day/night cycle",
                "Energy shields, oxygen timer, score by surviving waves",
                "Heavy gravity, stamina + health bars, checkpoint respawn",
                "Fast movement, stealth meter, dynamic wanted level",
            ],
            [
                "Enemies ambush at night, relic items spawn every minute",
                "Meteor storms trigger system failures and enemy waves",
                "Trap rooms activate, healing item drops after boss phase",
                "Weather glitches alter visibility and spawn elite enemies",
            ],
        ],
        "defaults": [
            "Mystic forest with glowing trees",
            "Player ranger, wolf enemies, guide NPC",
            "Health bar, score by collecting runes, day/night cycle",
            "Wolves attack at night, rune items respawn every 30 seconds",
        ],
    },
    "small_business_app": {
        "label": "🏪 Small Business App",
        "build_button": "Build My App",
        "stage_titles": [
            "What's your business?",
            "What do you need to manage?",
            "Who uses it?",
            "What should it do automatically?",
        ],
        "examples": [
            ["Neighborhood restaurant", "Hair salon", "Online boutique shop", "Freelance design studio"],
            ["Bookings and customer queue", "Inventory and suppliers", "Customers and loyalty", "Invoices and payments"],
            ["Just me", "My staff", "My customers", "Staff and customers"],
            ["Send reminders", "Track stock", "Generate reports", "Flag overdue invoices"],
        ],
        "defaults": [
            "Neighborhood restaurant",
            "Bookings, customers, and invoices",
            "My staff",
            "Send reminders and generate reports",
        ],
    },
    "personal_tool": {
        "label": "📱 Personal Tool",
        "build_button": "Build My App",
        "stage_titles": [
            "What problem are you solving?",
            "What information do you want to track?",
            "How often do you use it?",
            "What should it show or tell you?",
        ],
        "examples": [
            ["I forget daily habits", "I lose track of spending", "I miss assignment deadlines", "I need better workout consistency"],
            ["Habit streaks and notes", "Income and expenses", "Tasks and due dates", "Workout sets and progress"],
            ["Every day", "A few times per week", "Only on weekdays", "Once per week"],
            ["Daily summary", "Alerts when I miss a goal", "Weekly trend chart", "Simple progress score"],
        ],
        "defaults": [
            "I forget daily habits",
            "Habit streaks and notes",
            "Every day",
            "Daily summary and alerts when streak drops",
        ],
    },
    "automation_bot": {
        "label": "🤖 Automation Bot",
        "build_button": "Build My App",
        "stage_titles": [
            "What workflow should be automated?",
            "What inputs does it need?",
            "When should it run?",
            "What output or alert should it produce?",
        ],
        "examples": [
            ["Process incoming support emails", "Organize downloaded invoices", "Backup project folders", "Sync CRM leads to spreadsheet"],
            ["Email subject/body", "Folder path and file names", "Source and backup paths", "CSV exports"],
            ["Every 10 minutes", "At end of day", "Every hour", "On file change"],
            ["Send Slack alert", "Generate summary report", "Write logs", "Create exception list"],
        ],
        "defaults": [
            "Organize downloaded invoices",
            "Folder path and file names",
            "At end of day",
            "Generate summary report and send alert",
        ],
    },
    "dashboard": {
        "label": "📊 Dashboard",
        "build_button": "Build My App",
        "stage_titles": [
            "What is this dashboard for?",
            "What metrics should it show?",
            "Who will view it?",
            "What insights/alerts should it provide?",
        ],
        "examples": [
            ["Sales performance", "Marketing campaign health", "Support team operations", "Personal productivity"],
            ["Revenue, orders, conversion", "CTR, CPC, spend", "Tickets opened/closed", "Tasks done and focus time"],
            ["Founder only", "Team leads", "Operations staff", "Clients"],
            ["Highlight anomalies", "Weekly trends", "Low-performance alerts", "Top opportunities"],
        ],
        "defaults": [
            "Sales performance",
            "Revenue, orders, conversion",
            "Team leads",
            "Highlight anomalies and weekly trends",
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
if "world_project_type" not in st.session_state:
    st.session_state.world_project_type = "game_world"
for project_key, config in PROJECT_TYPE_CONFIG.items():
    for idx, default_value in enumerate(config["defaults"], start=1):
        state_key = f"{project_key}_stage_{idx}"
        if state_key not in st.session_state:
            st.session_state[state_key] = default_value

with st.sidebar:
    ui_mode = st.radio("Mode", ["Quick Generate", "World Builder"], index=0)
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
    mode = st.selectbox("App/Game mode", ["gameplay", "automation", "video-processing", "web-backend"], index=0)
    target = st.selectbox("Target", ["python", "blueprint", "cpp", "csharp", "javascript", "gdscript"], index=0)
    model_name = st.text_input("Model override (optional)", "")
    ollama_base_url = st.text_input("Ollama base URL", "http://localhost:11434")
    use_rag_cache = st.checkbox("Use lattice RAG cache (fallback engine only)", value=False)
    show_guide = st.checkbox("Show assistant guide", value=False)

translator = EnglishToCodeTranslator(planner_provider="auto")

if ui_mode == "Quick Generate":
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

    prompt = st.text_area("Prompt", key="prompt_input", height=160)

    if st.button("Generate"):
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

        zip_bytes = package_single_file_project(
            code=output,
            prompt=prompt,
            target=target,
            include_pygame=False,
        )
        st.download_button(
            "Download My Project",
            data=zip_bytes,
            file_name="nevora_project.zip",
            mime="application/zip",
            use_container_width=True,
        )

        st.markdown("### Copy code")
        st.components.v1.html(
            f"""
            <textarea id=\"nevora_code\" style=\"width:100%;height:180px;\">{html.escape(output)}</textarea>
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

        st.subheader("Explain My Code")
        try:
            explanation = explain_code_with_claude(
                code=output,
                user_goal=prompt,
                model=model_name or "claude-haiku-4-5",
            )
            st.markdown(explanation)
        except Exception as exc:
            st.info(f"Code explanation unavailable right now: {exc}")

else:
    st.markdown("## 🌍 World Builder")
    st.markdown("Choose a project type, complete 4 stages, then click build.")

    project_cols = st.columns(5)
    project_keys = list(PROJECT_TYPE_CONFIG.keys())
    for idx, project_key in enumerate(project_keys):
        config = PROJECT_TYPE_CONFIG[project_key]
        selected = st.session_state.world_project_type == project_key
        label = f"✅ {config['label']}" if selected else config["label"]
        if project_cols[idx].button(label, use_container_width=True, key=f"ptype_{project_key}"):
            st.session_state.world_project_type = project_key

    selected_project_key = st.session_state.world_project_type
    selected_project = PROJECT_TYPE_CONFIG[selected_project_key]

    def _stage_ui(stage_index: int, title: str, state_key: str, examples: list[str]) -> None:
        st.markdown(f"### {stage_index}) {title}")
        cols = st.columns(4)
        for i, ex in enumerate(examples):
            if cols[i].button(ex, use_container_width=True, key=f"wb_{selected_project_key}_{stage_index}_{i}"):
                st.session_state[state_key] = ex
        st.text_area("", key=state_key, height=90)

    stage_values: list[tuple[str, str]] = []
    for idx, stage_title in enumerate(selected_project["stage_titles"], start=1):
        state_key = f"{selected_project_key}_stage_{idx}"
        _stage_ui(idx, stage_title, state_key, selected_project["examples"][idx - 1])
        stage_values.append((stage_title, st.session_state[state_key]))

    if st.button(selected_project["build_button"], type="primary"):
        provider_map = {
            "claude-haiku-4-5 (default)": "claude",
            "openai": "openai",
            "grok": "grok",
            "gemini": "gemini",
            "ollama (local)": "ollama",
            "nevora-template-fallback": "nevora-template-fallback",
        }

        try:
            sections = generate_structured_project(
                project_type=selected_project["label"],
                stages=stage_values,
                provider=provider_map[generation_engine],
                model=model_name or ("claude-haiku-4-5" if provider_map[generation_engine] == "claude" else None),
                ollama_base_url=ollama_base_url,
            )
        except Exception as exc:
            st.error(f"World Builder generation failed: {exc}. Ensure ANTHROPIC_API_KEY is set.")
            st.stop()

        labels = [selected_project["stage_titles"][0], selected_project["stage_titles"][1], selected_project["stage_titles"][2], selected_project["stage_titles"][3]]
        st.success("Project generated. Review each section below.")
        st.subheader(labels[0])
        st.code(sections["section_one"], language="python")
        st.subheader(labels[1])
        st.code(sections["section_two"], language="python")
        st.subheader(labels[2])
        st.code(sections["section_three"], language="python")
        st.subheader(labels[3])
        st.code(sections["section_four"], language="python")
        st.subheader("Main (runnable starter)")
        st.code(sections["main"], language="python")

        combined = (
            "# section_one.py\n" + sections["section_one"]
            + "\n\n# section_two.py\n" + sections["section_two"]
            + "\n\n# section_three.py\n" + sections["section_three"]
            + "\n\n# section_four.py\n" + sections["section_four"]
            + "\n\n# main.py\n" + sections["main"]
        )
        st.markdown("### Copy full starter project")
        st.components.v1.html(
            f"""
            <textarea id=\"nevora_world_code\" style=\"width:100%;height:260px;\">{html.escape(combined)}</textarea>
            <button onclick=\"navigator.clipboard.writeText(document.getElementById('nevora_world_code').value)\" style=\"margin-top:8px;\">Copy Full Project</button>
            """,
            height=320,
        )

        world_files = {
            "section_one.py": sections["section_one"],
            "section_two.py": sections["section_two"],
            "section_three.py": sections["section_three"],
            "section_four.py": sections["section_four"],
            "main.py": sections["main"],
        }
        world_summary = " | ".join(f"{title}: {value}" for title, value in stage_values)
        world_zip = package_world_builder_project(
            files=world_files,
            project_title=f"Nevora {selected_project['label']}",
            app_summary=world_summary,
        )
        st.download_button(
            "Download My Project",
            data=world_zip,
            file_name="nevora_world_project.zip",
            mime="application/zip",
            use_container_width=True,
        )

        st.subheader("Explain My Code")
        try:
            explanation = explain_code_with_claude(
                code=combined,
                user_goal=world_summary,
                model=model_name or "claude-haiku-4-5",
            )
            st.markdown(explanation)
        except Exception as exc:
            st.info(f"Code explanation unavailable right now: {exc}")

st.caption(
    "Tip: run with `streamlit run app.py`. Keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY, GEMINI_API_KEY; local Ollama uses OLLAMA_BASE_URL."
)
