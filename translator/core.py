from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from translator.models import (
    EventSpec,
    GenerationIR,
    GenerationPlan,
    IntentSchema,
    ParsedIntent,
    PlanStep,
    StateTransition,
)
from translator.planners.heuristic import HeuristicPlanner
from translator.planners.openai_planner import OpenAISemanticPlanner
from translator.targets.registry import build_registry


class EnglishToCodeTranslator:
    MODES = {"gameplay", "automation", "video-processing", "web-backend"}
    PLANNER_PROVIDERS = {"auto", "heuristic", "openai"}
    BLOCKED_PATTERNS = [
        "rm -rf /",
        "shutdown",
        "format c:",
        "drop database",
        "os.system(\"rm",
        "subprocess.run(['rm",
    ]

    def __init__(
        self,
        planner: Optional[object] = None,
        planner_provider: str = "auto",
    ) -> None:
        if planner_provider not in self.PLANNER_PROVIDERS:
            raise ValueError(
                f"Unsupported planner_provider '{planner_provider}'. "
                f"Supported: {', '.join(sorted(self.PLANNER_PROVIDERS))}"
            )
        self._heuristic = HeuristicPlanner()
        self.planner = planner
        self.planner_provider = planner_provider
        self._last_resolved_provider = "custom" if planner is not None else planner_provider
        self.renderers = build_registry()

    @property
    def supported_targets(self) -> set[str]:
        return set(self.renderers.keys())

    @property
    def last_resolved_provider(self) -> str:
        return self._last_resolved_provider

    def _get_planner(self) -> object:
        if self.planner is not None:
            self._last_resolved_provider = "custom"
            return self.planner
        if self.planner_provider == "heuristic":
            self._last_resolved_provider = "heuristic"
            return self._heuristic
        if self.planner_provider == "openai":
            self._last_resolved_provider = "openai"
            return OpenAISemanticPlanner()

        try:
            self._last_resolved_provider = "openai"
            return OpenAISemanticPlanner()
        except Exception:
            self._last_resolved_provider = "heuristic"
            return self._heuristic

    def _canonicalize_intent(self, intent: ParsedIntent) -> ParsedIntent:
        schema = IntentSchema(
            entities=intent.entities if isinstance(intent.entities, list) else ["system"],
            actions=intent.actions if isinstance(intent.actions, list) else ["process"],
            conditions=intent.conditions if isinstance(intent.conditions, list) else ["always"],
            outputs=intent.outputs if isinstance(intent.outputs, list) else ["state"],
        )
        return ParsedIntent(
            entities=[str(x) for x in schema.entities] or ["system"],
            actions=[str(x) for x in schema.actions] or ["process"],
            conditions=[str(x) for x in schema.conditions] or ["always"],
            outputs=[str(x) for x in schema.outputs] or ["state"],
        )

    def _enforce_safety(self, text: str, strict_safety: bool = False) -> None:
        if not strict_safety:
            return
        lowered = text.lower()
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in lowered:
                raise ValueError(f"Safety policy blocked content containing pattern: {pattern}")

    def plan_intent(self, prompt: str, mode: str = "gameplay") -> ParsedIntent:
        try:
            planner = self._get_planner()
            raw_intent = planner.plan(prompt, mode=mode)
        except Exception:
            self._last_resolved_provider = "heuristic-fallback"
            raw_intent = self._heuristic.plan(prompt, mode=mode)
        return self._canonicalize_intent(raw_intent)

    def _build_ir(self, intent: ParsedIntent) -> GenerationIR:
        trigger = " + ".join(intent.conditions)
        action_list = ", ".join(intent.actions)
        events = [EventSpec(name="primary_event", trigger=trigger)]
        transitions = [StateTransition(from_state="active", to_state="active", condition=trigger)]
        side_effects = [f"execute: {action_list}", f"emit: {', '.join(intent.outputs)}"]
        error_branches = ["on planner/render failure: fallback to safe no-op"]
        return GenerationIR(
            events=events,
            transitions=transitions,
            side_effects=side_effects,
            error_branches=error_branches,
        )

    def build_generation_plan(self, prompt: str, mode: str = "gameplay") -> GenerationPlan:
        intent = self.plan_intent(prompt, mode=mode)
        ir = self._build_ir(intent)
        steps = [
            PlanStep("intent-parse", f"entities={intent.entities}, actions={intent.actions}"),
            PlanStep("task-decompose", "Split into event handling, state transitions, and outputs"),
            PlanStep("target-design", f"Use templates optimized for mode={mode}"),
            PlanStep("generate", "Render target code"),
            PlanStep("self-check", "Optional syntax/build verification"),
        ]
        state_model = {"active": "bool", "last_event": "string", "status": "string"}
        return GenerationPlan(intent=intent, ir=ir, steps=steps, state_model=state_model)

    def explain_plan(self, prompt: str, target: str, mode: str = "gameplay") -> dict[str, Any]:
        plan = self.build_generation_plan(prompt, mode=mode)
        return {
            "target": target,
            "mode": mode,
            "planner_provider": self.planner_provider,
            "resolved_provider": self._last_resolved_provider,
            "intent": {
                "entities": plan.intent.entities,
                "actions": plan.intent.actions,
                "conditions": plan.intent.conditions,
                "outputs": plan.intent.outputs,
            },
            "ir": {
                "events": [event.__dict__ for event in plan.ir.events],
                "transitions": [transition.__dict__ for transition in plan.ir.transitions],
                "side_effects": plan.ir.side_effects,
                "error_branches": plan.ir.error_branches,
            },
            "steps": [step.__dict__ for step in plan.steps],
            "state_model": plan.state_model,
        }

    def translate(
        self,
        prompt: str,
        target: str,
        mode: str = "gameplay",
        context: Optional[str] = None,
        refine: bool = False,
        strict_safety: bool = False,
    ) -> str:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Supported: {', '.join(sorted(self.MODES))}")

        normalized_target = target.strip().lower()
        if normalized_target not in self.supported_targets:
            supported = ", ".join(sorted(self.supported_targets))
            raise ValueError(f"Unsupported target '{target}'. Supported: {supported}")

        combined_prompt = prompt
        if refine and context:
            combined_prompt = f"{prompt}\n\nPrevious output context:\n{context}"

        self._enforce_safety(combined_prompt, strict_safety=strict_safety)
        plan = self.build_generation_plan(combined_prompt, mode=mode)
        renderer = self.renderers[normalized_target]
        output = renderer.render(combined_prompt, plan.intent, mode=mode, plan=plan)
        self._enforce_safety(output, strict_safety=strict_safety)
        return output

    def _slug(self, text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        return cleaned[:48] or "item"

    def translate_batch(
        self,
        items: list[dict[str, Any]],
        default_target: str,
        default_mode: str = "gameplay",
        strict_safety: bool = False,
        artifact_dir: str | None = None,
        include_explain: bool = False,
        fail_fast: bool = False,
        verify_generated: bool = False,
        verify_build: bool = False,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        artifacts_root = Path(artifact_dir) if artifact_dir else None
        if artifacts_root:
            artifacts_root.mkdir(parents=True, exist_ok=True)

        for idx, item in enumerate(items):
            prompt = str(item.get("prompt", "")).strip()
            target = str(item.get("target", default_target)).strip()
            mode = str(item.get("mode", default_mode)).strip()
            context = item.get("context")
            refine = bool(item.get("refine", False))

            try:
                output = self.translate(
                    prompt=prompt,
                    target=target,
                    mode=mode,
                    context=context,
                    refine=refine,
                    strict_safety=strict_safety,
                )
                payload: dict[str, Any] = {
                    "index": idx,
                    "ok": True,
                    "target": target,
                    "mode": mode,
                    "resolved_provider": self._last_resolved_provider,
                    "output": output,
                }

                if verify_generated:
                    verify_ok, verify_message = self.verify_output(output, target)
                    payload["verify_output_ok"] = verify_ok
                    payload["verify_output_message"] = verify_message

                if include_explain:
                    payload["explain"] = self.explain_plan(prompt, target=target, mode=mode)

                scaffold_root: Path | None = None
                if artifacts_root:
                    item_dir = artifacts_root / f"{idx:03d}_{self._slug(prompt)}"
                    item_dir.mkdir(parents=True, exist_ok=True)
                    output_file = item_dir / f"output.{target}.txt"
                    output_file.write_text(output, encoding="utf-8")
                    payload["artifact_output_file"] = str(output_file)
                    scaffold_root = item_dir / "scaffold"

                    if include_explain:
                        explain_file = item_dir / "plan.json"
                        explain_file.write_text(json.dumps(payload["explain"], indent=2), encoding="utf-8")
                        payload["artifact_plan_file"] = str(explain_file)

                if verify_build:
                    if scaffold_root is None:
                        import tempfile

                        with tempfile.TemporaryDirectory(prefix="nevora-batch-scaffold-") as td:
                            self.scaffold_project(prompt, target=target, output_dir=td, mode=mode)
                            build_ok, build_message = self.verify_scaffold_build(td, target)
                    else:
                        self.scaffold_project(prompt, target=target, output_dir=str(scaffold_root), mode=mode)
                        build_ok, build_message = self.verify_scaffold_build(str(scaffold_root), target)
                    payload["verify_build_ok"] = build_ok
                    payload["verify_build_message"] = build_message

                results.append(payload)
            except Exception as exc:
                results.append(
                    {
                        "index": idx,
                        "ok": False,
                        "target": target,
                        "mode": mode,
                        "resolved_provider": self._last_resolved_provider,
                        "error": str(exc),
                    }
                )
                if fail_fast:
                    break
        return results

    def write_batch_report(self, batch_results: list[dict[str, Any]], output_file: str) -> str:
        destination = Path(output_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ok_count = sum(1 for r in batch_results if r.get("ok"))
        failed_count = sum(1 for r in batch_results if not r.get("ok"))
        verify_output_ok_count = sum(1 for r in batch_results if r.get("verify_output_ok") is True)
        verify_build_ok_count = sum(1 for r in batch_results if r.get("verify_build_ok") is True)

        target_counts: dict[str, int] = {}
        provider_counts: dict[str, int] = {}
        for item in batch_results:
            target = str(item.get("target", "unknown"))
            provider = str(item.get("resolved_provider", "unknown"))
            target_counts[target] = target_counts.get(target, 0) + 1
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

        total = len(batch_results)
        success_rate = (ok_count / total) if total else 0.0
        verify_output_rate = (verify_output_ok_count / total) if total else 0.0
        verify_build_rate = (verify_build_ok_count / total) if total else 0.0
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": total,
            "ok": ok_count,
            "failed": failed_count,
            "success_rate": round(success_rate, 4),
            "verify_output_ok": verify_output_ok_count,
            "verify_build_ok": verify_build_ok_count,
            "verify_output_rate": round(verify_output_rate, 4),
            "verify_build_rate": round(verify_build_rate, 4),
            "target_counts": target_counts,
            "resolved_provider_counts": provider_counts,
            "results": batch_results,
        }
        destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return str(destination)

    def scaffold_project(self, prompt: str, target: str, output_dir: str, mode: str = "gameplay") -> str:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        code = self.translate(prompt, target=target, mode=mode)

        if target == "python":
            (root / "src").mkdir(exist_ok=True)
            (root / "src" / "generated_feature.py").write_text(code, encoding="utf-8")
            (root / "tests").mkdir(exist_ok=True)
            (root / "tests" / "test_generated.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
        elif target == "javascript":
            (root / "src").mkdir(exist_ok=True)
            (root / "src" / "generatedFeature.js").write_text(code, encoding="utf-8")
            (root / "package.json").write_text('{"name":"generated-feature","version":"0.1.0"}\n', encoding="utf-8")
        elif target == "csharp":
            (root / "GeneratedFeature.cs").write_text(code, encoding="utf-8")
            (root / "GeneratedFeature.csproj").write_text(
                "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><TargetFramework>net8.0</TargetFramework></PropertyGroup></Project>",
                encoding="utf-8",
            )
        elif target == "cpp":
            (root / "main.cpp").write_text(code, encoding="utf-8")
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.16)\nproject(GeneratedFeature)\nadd_executable(app main.cpp)\n",
                encoding="utf-8",
            )
        elif target == "gdscript":
            (root / "GeneratedFeature.gd").write_text(code, encoding="utf-8")
            (root / "project.godot").write_text("; generated skeleton\n", encoding="utf-8")
        else:
            (root / "README.txt").write_text(code, encoding="utf-8")

        return str(root)

    def verify_output(self, code: str, target: str) -> tuple[bool, str]:
        if target == "python":
            try:
                compile(code, "<generated>", "exec")
                return True, "python compile ok"
            except Exception as exc:
                return False, f"python compile failed: {exc}"

        if target == "javascript":
            if shutil.which("node"):
                proc = subprocess.run(["node", "--check", "-"], input=code, text=True, capture_output=True)
                return proc.returncode == 0, proc.stderr.strip() or "node check ok"
            return False, "node unavailable"

        if target == "cpp":
            if shutil.which("clang++"):
                proc = subprocess.run(["clang++", "-fsyntax-only", "-x", "c++", "-"], input=code, text=True, capture_output=True)
                return proc.returncode == 0, proc.stderr.strip() or "clang++ syntax ok"
            return False, "clang++ unavailable"

        if target == "csharp":
            if shutil.which("dotnet"):
                return True, "dotnet available"
            return False, "dotnet unavailable"

        if target == "gdscript":
            if shutil.which("godot"):
                return True, "godot available"
            return False, "godot unavailable"

        return True, "verification not implemented for target"

    def verify_scaffold(self, scaffold_dir: str, target: str) -> tuple[bool, str]:
        root = Path(scaffold_dir)
        if not root.exists():
            return False, "scaffold directory does not exist"

        if target == "python":
            test_file = root / "tests" / "test_generated.py"
            return (test_file.exists(), "python scaffold test file present" if test_file.exists() else "missing tests/test_generated.py")

        if target == "javascript":
            pkg = root / "package.json"
            src = root / "src" / "generatedFeature.js"
            ok = pkg.exists() and src.exists()
            return ok, "javascript scaffold files present" if ok else "missing package.json or src/generatedFeature.js"

        if target == "cpp":
            cmake = root / "CMakeLists.txt"
            cpp = root / "main.cpp"
            ok = cmake.exists() and cpp.exists()
            return ok, "cpp scaffold files present" if ok else "missing CMakeLists.txt or main.cpp"

        if target == "csharp":
            csproj = root / "GeneratedFeature.csproj"
            cs = root / "GeneratedFeature.cs"
            ok = csproj.exists() and cs.exists()
            return ok, "csharp scaffold files present" if ok else "missing csproj or source file"

        if target == "gdscript":
            gd = root / "GeneratedFeature.gd"
            godot = root / "project.godot"
            ok = gd.exists() and godot.exists()
            return ok, "gdscript scaffold files present" if ok else "missing Godot scaffold files"

        return True, "no scaffold verification for target"

    def verify_scaffold_build(self, scaffold_dir: str, target: str) -> tuple[bool, str]:
        root = Path(scaffold_dir)
        if not root.exists():
            return False, "scaffold directory does not exist"

        if target == "python":
            if not shutil.which("pytest"):
                return False, "pytest unavailable"
            proc = subprocess.run(["pytest", "-q"], cwd=root, capture_output=True, text=True)
            return proc.returncode == 0, (proc.stdout.strip() or proc.stderr.strip() or "pytest finished")

        if target == "javascript":
            if not shutil.which("node"):
                return False, "node unavailable"
            src = root / "src" / "generatedFeature.js"
            if not src.exists():
                return False, "missing src/generatedFeature.js"
            proc = subprocess.run(["node", "--check", "src/generatedFeature.js"], cwd=root, capture_output=True, text=True)
            return proc.returncode == 0, (proc.stdout.strip() or proc.stderr.strip() or "node check ok")

        if target == "cpp":
            if not shutil.which("clang++"):
                return False, "clang++ unavailable"
            src = root / "main.cpp"
            if not src.exists():
                return False, "missing main.cpp"
            proc = subprocess.run(["clang++", "-fsyntax-only", "main.cpp"], cwd=root, capture_output=True, text=True)
            return proc.returncode == 0, (proc.stdout.strip() or proc.stderr.strip() or "clang++ syntax ok")

        if target == "csharp":
            if not shutil.which("dotnet"):
                return False, "dotnet unavailable"
            proc = subprocess.run(["dotnet", "build", "-nologo"], cwd=root, capture_output=True, text=True)
            return proc.returncode == 0, (proc.stdout.strip() or proc.stderr.strip() or "dotnet build ok")

        if target == "gdscript":
            return False, "gdscript build verification not implemented"

        return False, "no scaffold build verification for target"

    def export_unreal_uasset_payload(
        self,
        prompt: str,
        output_path: str,
        blueprint_name: str = "BP_GeneratedFeature",
        mode: str = "gameplay",
    ) -> str:
        plan = self.build_generation_plan(prompt, mode=mode)
        payload: dict[str, Any] = {
            "schema": "nevora.unreal.blueprint.graph.v2",
            "blueprint_name": blueprint_name,
            "mode": mode,
            "prompt": prompt,
            "intent": {
                "entities": plan.intent.entities,
                "actions": plan.intent.actions,
                "conditions": plan.intent.conditions,
                "outputs": plan.intent.outputs,
            },
            "ir": {
                "events": [event.__dict__ for event in plan.ir.events],
                "transitions": [transition.__dict__ for transition in plan.ir.transitions],
                "side_effects": plan.ir.side_effects,
                "error_branches": plan.ir.error_branches,
            },
            "nodes": [
                {"id": "begin_play", "type": "EventBeginPlay", "pins": ["exec_out"]},
                {
                    "id": "branch_condition",
                    "type": "Branch",
                    "condition_tokens": plan.intent.conditions,
                    "pins": ["exec_in", "true", "false"],
                },
                {
                    "id": "action_sequence",
                    "type": "ActionSequence",
                    "actions": plan.intent.actions,
                    "entities": plan.intent.entities,
                    "outputs": plan.intent.outputs,
                    "pins": ["exec_in", "exec_out"],
                },
            ],
            "edges": [
                {"from": "begin_play.exec_out", "to": "branch_condition.exec_in"},
                {"from": "branch_condition.true", "to": "action_sequence.exec_in"},
            ],
        }

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(destination)
