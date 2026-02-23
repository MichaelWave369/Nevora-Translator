import json
from pathlib import Path

import pytest

from translator.core import EnglishToCodeTranslator
from translator.models import ParsedIntent
from translator.planners.heuristic import HeuristicPlanner


class BrokenPlanner:
    def plan(self, prompt: str, mode: str = "gameplay") -> ParsedIntent:  # intentionally bad shape
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
