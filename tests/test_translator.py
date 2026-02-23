import json
from pathlib import Path

import pytest

from translator.core import EnglishToCodeTranslator
from translator.models import ParsedIntent
from translator.planners.heuristic import HeuristicPlanner


class BrokenPlanner:
    def plan(self, prompt: str, mode: str = "gameplay") -> ParsedIntent:
        return ParsedIntent(entities=[1, 2], actions=["jump"], conditions=["when"], outputs=[])


def test_pipeline_translation_and_modes() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    out = translator.translate(
        prompt="When request arrives validate and respond",
        target="python",
        mode="web-backend",
    )
    assert "GeneratedFeature" in out
    assert "Mode: web-backend" in out
    assert "IR:" in out


def test_explain_plan_returns_ir_and_steps() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    explanation = translator.explain_plan(
        prompt="Spawn enemy when timer reaches zero",
        target="cpp",
        mode="gameplay",
    )
    assert explanation["target"] == "cpp"
    assert explanation["planner_provider"] == "auto"
    assert explanation["resolved_provider"] in {"custom", "heuristic", "heuristic-fallback", "openai"}
    assert "intent" in explanation
    assert "ir" in explanation
    assert len(explanation["steps"]) >= 3


def test_canonical_schema_normalizes_values() -> None:
    translator = EnglishToCodeTranslator(planner=BrokenPlanner())
    intent = translator.plan_intent("jump", mode="gameplay")
    assert intent.entities == ["1", "2"]
    assert intent.outputs == ["state"]


def test_new_target_registry_renders() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    prompt = "Spawn enemy when timer reaches zero"

    assert "GeneratedFeature" in translator.translate(prompt=prompt, target="cpp")
    assert "public class GeneratedFeature" in translator.translate(prompt=prompt, target="csharp")
    assert "class GeneratedFeature" in translator.translate(prompt=prompt, target="javascript")
    assert "extends Node" in translator.translate(prompt=prompt, target="gdscript")


def test_unreal_uasset_payload_export_v2_contains_ir(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    target = tmp_path / "bp_feature.json"

    result = translator.export_unreal_uasset_payload(
        prompt="When player collides with enemy play hit animation",
        output_path=str(target),
        blueprint_name="BP_TestFeature",
    )

    assert result.endswith("bp_feature.json")
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["schema"] == "nevora.unreal.blueprint.graph.v2"
    assert data["blueprint_name"] == "BP_TestFeature"
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)
    assert "ir" in data


def test_scaffold_project_and_verify_python(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    root = translator.scaffold_project(
        prompt="Create a player that jumps",
        target="python",
        output_dir=str(tmp_path / "pyproj"),
    )
    assert (tmp_path / "pyproj" / "src" / "generated_feature.py").exists()
    assert root.endswith("pyproj")
    ok, _ = translator.verify_scaffold(root, "python")
    assert ok


def test_verify_scaffold_build_python(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    root = translator.scaffold_project(
        prompt="Create a player that jumps",
        target="python",
        output_dir=str(tmp_path / "pyproj"),
    )
    ok, message = translator.verify_scaffold_build(root, "python")
    assert ok, message


def test_refine_with_context_changes_prompt_usage() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    out = translator.translate(
        prompt="add saving",
        target="python",
        mode="automation",
        context="Previous output had no persistence",
        refine=True,
    )
    assert "Previous output context" in out


def test_verify_python_passes() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    output = translator.translate("Create jump", "python")
    ok, _ = translator.verify_output(output, "python")
    assert ok


def test_deterministic_translation_for_eval_signal() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    a = translator.translate("Create a player jump on space and play sound", "python")
    b = translator.translate("Create a player jump on space and play sound", "python")
    assert a == b


def test_planner_provider_heuristic_is_selectable() -> None:
    translator = EnglishToCodeTranslator(planner_provider="heuristic")
    out = translator.translate("Create a player that can jump", "python")
    assert "GeneratedFeature" in out
    assert translator.last_resolved_provider == "heuristic"


def test_strict_safety_blocks_unsafe_prompt() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    with pytest.raises(ValueError):
        translator.translate("Please run rm -rf / immediately", "python", strict_safety=True)


def test_batch_translate_and_report(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    batch = [
        {"prompt": "Create a player that can jump", "target": "python"},
        {"prompt": "Spawn enemy when timer reaches zero", "target": "cpp"},
    ]
    results = translator.translate_batch(batch, default_target="python")
    assert len(results) == 2
    assert all(item["ok"] for item in results)

    report_path = translator.write_batch_report(results, str(tmp_path / "report.json"))
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report_path.endswith("report.json")
    assert payload["total"] == 2
    assert payload["ok"] == 2


def test_batch_translate_with_artifacts_and_explain(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    batch = [
        {"prompt": "Create a player that can jump", "target": "python"},
        {"prompt": "Spawn enemy when timer reaches zero", "target": "cpp"},
    ]
    artifact_dir = tmp_path / "artifacts"
    results = translator.translate_batch(
        batch,
        default_target="python",
        artifact_dir=str(artifact_dir),
        include_explain=True,
    )
    assert len(results) == 2
    assert all(item["ok"] for item in results)
    assert all("artifact_output_file" in item for item in results)
    assert all("artifact_plan_file" in item for item in results)
    assert all(Path(item["artifact_output_file"]).exists() for item in results)
    assert all(Path(item["artifact_plan_file"]).exists() for item in results)


def test_batch_report_contains_generated_at(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    results = [{"index": 0, "ok": True, "target": "python", "mode": "gameplay", "output": "x"}]
    report = translator.write_batch_report(results, str(tmp_path / "batch.json"))
    payload = json.loads(Path(report).read_text(encoding="utf-8"))
    assert "generated_at" in payload


def test_batch_fail_fast_stops_after_first_error() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    batch = [
        {"prompt": "Please run rm -rf / immediately", "target": "python"},
        {"prompt": "Create a player that can jump", "target": "python"},
    ]
    results = translator.translate_batch(
        batch,
        default_target="python",
        strict_safety=True,
        fail_fast=True,
    )
    assert len(results) == 1
    assert results[0]["ok"] is False


def test_batch_report_contains_rates_and_counts(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    results = [
        {"index": 0, "ok": True, "target": "python", "mode": "gameplay", "resolved_provider": "heuristic"},
        {"index": 1, "ok": False, "target": "cpp", "mode": "gameplay", "resolved_provider": "heuristic-fallback"},
    ]
    report = translator.write_batch_report(results, str(tmp_path / "batch.json"))
    payload = json.loads(Path(report).read_text(encoding="utf-8"))
    assert payload["success_rate"] == 0.5
    assert payload["target_counts"]["python"] == 1
    assert payload["target_counts"]["cpp"] == 1
    assert payload["resolved_provider_counts"]["heuristic"] == 1


def test_batch_translate_with_verify_flags(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    batch = [{"prompt": "Create a player that can jump", "target": "python"}]
    results = translator.translate_batch(
        batch,
        default_target="python",
        verify_generated=True,
        verify_build=True,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    assert len(results) == 1
    assert results[0]["verify_output_ok"] is True
    assert "verify_build_ok" in results[0]


def test_batch_report_contains_verify_metrics(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    results = [
        {
            "index": 0,
            "ok": True,
            "target": "python",
            "mode": "gameplay",
            "resolved_provider": "heuristic",
            "verify_output_ok": True,
            "verify_build_ok": True,
        },
        {
            "index": 1,
            "ok": False,
            "target": "cpp",
            "mode": "gameplay",
            "resolved_provider": "heuristic-fallback",
            "verify_output_ok": False,
            "verify_build_ok": False,
        },
    ]
    report = translator.write_batch_report(results, str(tmp_path / "batch_verify.json"))
    payload = json.loads(Path(report).read_text(encoding="utf-8"))
    assert payload["verify_output_ok"] == 1
    assert payload["verify_build_ok"] == 1
    assert payload["verify_output_rate"] == 0.5
    assert payload["verify_build_rate"] == 0.5


def test_multilingual_spanish_prompt_translation() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    out = translator.translate(
        prompt="Cuando jugador saltar",
        target="python",
        source_language="spanish",
    )
    assert "player" in out.lower()
    assert "jump" in out.lower()


def test_explain_plan_includes_normalized_prompt() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    explanation = translator.explain_plan(
        prompt="Quand joueur sauter",
        target="python",
        source_language="french",
    )
    assert explanation["source_language"] == "french"
    assert "player" in explanation["normalized_prompt"].lower()


def test_batch_report_contains_source_language_counts(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    batch = [
        {"prompt": "Cuando jugador saltar", "target": "python", "source_language": "spanish"},
        {"prompt": "Spawn enemy when timer reaches zero", "target": "cpp", "source_language": "english"},
    ]
    results = translator.translate_batch(batch, default_target="python")
    report = translator.write_batch_report(results, str(tmp_path / "lang_batch.json"))
    payload = json.loads(Path(report).read_text(encoding="utf-8"))
    assert payload["source_language_counts"]["spanish"] == 1
    assert payload["source_language_counts"]["english"] == 1


def test_audio_input_txt_transcript() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    tmp = Path("/tmp/nevora_audio_prompt.txt")
    tmp.write_text("Cuando jugador saltar", encoding="utf-8")
    prompt = translator.transcribe_audio_input(str(tmp), source_language="spanish")
    out = translator.translate(prompt, target="python", source_language="spanish")
    assert "player" in out.lower()


def test_audio_output_fallback_txt(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    output_audio = tmp_path / "speech.wav"
    written = translator.synthesize_audio_output("hello world", str(output_audio), output_language="english")
    assert Path(written).exists()


def test_rag_lattice_store_and_retrieve() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    translator.translate("Create jump", "python", use_rag_cache=True)
    neighbors = translator.rag_retrieve("Create jump", "python")
    assert neighbors
    assert neighbors[-1]["target"] == "python"


def test_translate_batch_swarm_workers() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    batch = [
        {"prompt": "Create a player that can jump", "target": "python"},
        {"prompt": "Spawn enemy when timer reaches zero", "target": "cpp"},
        {"prompt": "When request arrives validate and respond", "target": "javascript"},
    ]
    results = translator.translate_batch(batch, default_target="python", swarm_workers=3)
    assert len(results) == 3
    assert all("lattice_bucket" in item for item in results)


def test_vm_sandbox_execution() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    ok, message = translator.run_in_vm_sandbox(["python3", "-c", "print('ok')"])
    assert ok
    assert "ok" in message


def test_translate_with_unreal_asset_library(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    lib = {
        "unreal": [
            {"id": "SM_Enemy", "name": "Enemy Mesh", "tags": ["enemy", "mesh"], "path": "/Game/Meshes/SM_Enemy"},
            {"id": "SFX_Jump", "name": "Jump Sound", "tags": ["jump", "sound"], "path": "/Game/Audio/SFX_Jump"},
        ],
        "unity": [],
    }
    lib_path = tmp_path / "library.json"
    lib_path.write_text(json.dumps(lib), encoding="utf-8")

    result = translator.translate_with_asset_library(
        prompt="Spawn enemy and play jump sound",
        target="python",
        engine="unreal",
        asset_library_path=str(lib_path),
    )
    assert result["engine"] == "unreal"
    assert len(result["selected_assets"]) >= 1
    assert "GeneratedFeature" in result["output"]


def test_export_unity_asset_manifest(tmp_path) -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    lib = {
        "unreal": [],
        "unity": [
            {"id": "PlayerPrefab", "name": "Player Prefab", "tags": ["player", "jump"], "path": "Assets/Prefabs/Player.prefab"}
        ],
    }
    lib_path = tmp_path / "library.json"
    out_path = tmp_path / "unity_manifest.json"
    lib_path.write_text(json.dumps(lib), encoding="utf-8")

    written = translator.export_engine_asset_manifest(
        prompt="Player jump",
        target="csharp",
        engine="unity",
        asset_library_path=str(lib_path),
        output_path=str(out_path),
    )
    payload = json.loads(Path(written).read_text(encoding="utf-8"))
    assert payload["schema"] == "nevora.unity.asset.manifest.v1"
    assert payload["assets"]


def test_plan_cache_populated() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    translator.build_generation_plan("Create jump", mode="gameplay")
    assert ("Create jump", "gameplay") in translator._plan_cache


def test_assistant_guide_contains_suggestions() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    guide = translator.assistant_guide(
        prompt="Create player jump",
        target="python",
        mode="gameplay",
    )
    assert guide["target"] == "python"
    assert guide["suggestions"]
    assert "suggested_command" in guide


def test_warm_plan_cache_counts_entries() -> None:
    translator = EnglishToCodeTranslator(planner=HeuristicPlanner())
    result = translator.warm_plan_cache(["Create jump", "Create jump", "Spawn enemy"], mode="gameplay")
    assert result["cache_size"] >= 2
    assert result["warmed"] >= 2
