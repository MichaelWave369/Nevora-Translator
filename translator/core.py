from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from datetime import datetime, timezone
from time import perf_counter
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
from translator.planners.huggingface_planner import HuggingFaceSemanticPlanner
from translator.targets.registry import build_registry


class EnglishToCodeTranslator:
    MODES = {"gameplay", "automation", "video-processing", "web-backend"}
    PLANNER_PROVIDERS = {"auto", "heuristic", "openai", "huggingface"}
    SOURCE_LANGUAGES = {"english", "spanish", "french", "german", "portuguese"}
    AUDIO_LANGUAGES = SOURCE_LANGUAGES
    ASSET_ENGINES = {"unreal", "unity"}
    BLOCKED_PATTERNS = [
        "rm -rf /",
        "shutdown",
        "format c:",
        "drop database",
        "os.system(\"rm",
        "subprocess.run(['rm",
    ]

    LANGUAGE_TOKEN_MAP: dict[str, dict[str, str]] = {
        "spanish": {
            "jugador": "player",
            "enemigo": "enemy",
            "saltar": "jump",
            "mover": "move",
            "cuando": "when",
            "si": "if",
            "colision": "collision",
            "presionado": "pressed",
            "cero": "zero",
            "guardar": "save",
            "deshabilitar": "disable",
            "sonido": "sound",
            "animacion": "animation",
        },
        "french": {
            "joueur": "player",
            "ennemi": "enemy",
            "sauter": "jump",
            "deplacer": "move",
            "quand": "when",
            "si": "if",
            "collision": "collision",
            "appuye": "pressed",
            "zero": "zero",
            "sauvegarder": "save",
            "desactiver": "disable",
            "son": "sound",
            "animation": "animation",
        },
        "german": {
            "spieler": "player",
            "gegner": "enemy",
            "springen": "jump",
            "bewegen": "move",
            "wenn": "when",
            "falls": "if",
            "kollision": "collision",
            "gedruckt": "pressed",
            "null": "zero",
            "speichern": "save",
            "deaktivieren": "disable",
            "ton": "sound",
            "animation": "animation",
        },
        "portuguese": {
            "jogador": "player",
            "inimigo": "enemy",
            "pular": "jump",
            "mover": "move",
            "quando": "when",
            "se": "if",
            "colisao": "collision",
            "pressionado": "pressed",
            "zero": "zero",
            "salvar": "save",
            "desativar": "disable",
            "som": "sound",
            "animacao": "animation",
        },
    }

    AUDIO_LANGUAGE_CODES = {
        "english": "en-US",
        "spanish": "es-ES",
        "french": "fr-FR",
        "german": "de-DE",
        "portuguese": "pt-PT",
    }

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
        self._rag_lattice: dict[tuple[int, int, int, int], list[dict[str, str]]] = {}
        self._plan_cache: dict[tuple[str, str], GenerationPlan] = {}
        self.lattice_shape = (12, 12, 12, 12)

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
        if self.planner_provider == "huggingface":
            self._last_resolved_provider = "huggingface"
            return HuggingFaceSemanticPlanner()

        try:
            self._last_resolved_provider = "huggingface"
            return HuggingFaceSemanticPlanner()
        except Exception:
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

    def _normalize_prompt_language(self, prompt: str, source_language: str = "english") -> str:
        language = source_language.lower().strip()
        if language not in self.SOURCE_LANGUAGES:
            supported = ", ".join(sorted(self.SOURCE_LANGUAGES))
            raise ValueError(f"Unsupported source_language '{source_language}'. Supported: {supported}")

        if language == "english":
            return prompt

        token_map = self.LANGUAGE_TOKEN_MAP.get(language, {})
        normalized = prompt
        for source_token, english_token in token_map.items():
            normalized = re.sub(rf"\b{re.escape(source_token)}\b", english_token, normalized, flags=re.IGNORECASE)
        return normalized

    def transcribe_audio_input(self, audio_input_path: str, source_language: str = "english") -> str:
        """Transcribe audio input into text.

        Supports `.txt` as transcript input for deterministic/local workflows.
        If speech_recognition is installed, common audio file formats are supported.
        """
        language = source_language.lower().strip()
        if language not in self.AUDIO_LANGUAGES:
            supported = ", ".join(sorted(self.AUDIO_LANGUAGES))
            raise ValueError(f"Unsupported source_language '{source_language}'. Supported: {supported}")

        source = Path(audio_input_path)
        if not source.exists():
            raise FileNotFoundError(f"Audio input does not exist: {audio_input_path}")

        if source.suffix.lower() in {".txt", ".md", ".prompt"}:
            return source.read_text(encoding="utf-8").strip()

        try:
            import speech_recognition as sr  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "speech_recognition not available. Install optional audio deps or provide a .txt transcript file."
            ) from exc

        recognizer = sr.Recognizer()
        with sr.AudioFile(str(source)) as audio_file:
            audio_data = recognizer.record(audio_file)
        return recognizer.recognize_google(audio_data, language=self.AUDIO_LANGUAGE_CODES[language])

    def synthesize_audio_output(
        self,
        text: str,
        output_audio_path: str,
        output_language: str = "english",
    ) -> str:
        """Synthesize speech audio output.

        Uses optional `pyttsx3` when available. Falls back to writing a `.txt`
        sidecar transcript if audio TTS backend is unavailable.
        """
        language = output_language.lower().strip()
        if language not in self.AUDIO_LANGUAGES:
            supported = ", ".join(sorted(self.AUDIO_LANGUAGES))
            raise ValueError(f"Unsupported output_language '{output_language}'. Supported: {supported}")

        destination = Path(output_audio_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            import pyttsx3  # type: ignore

            engine = pyttsx3.init()
            engine.save_to_file(text, str(destination))
            engine.runAndWait()
            return str(destination)
        except Exception:
            fallback = destination.with_suffix(destination.suffix + ".txt")
            fallback.write_text(text, encoding="utf-8")
            return str(fallback)

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
        cache_key = (prompt, mode)
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            return cached

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
        plan = GenerationPlan(intent=intent, ir=ir, steps=steps, state_model=state_model)
        self._plan_cache[cache_key] = plan
        if len(self._plan_cache) > 256:
            oldest = next(iter(self._plan_cache))
            self._plan_cache.pop(oldest, None)
        return plan

    def explain_plan(
        self,
        prompt: str,
        target: str,
        mode: str = "gameplay",
        source_language: str = "english",
    ) -> dict[str, Any]:
        normalized_prompt = self._normalize_prompt_language(prompt, source_language=source_language)
        plan = self.build_generation_plan(normalized_prompt, mode=mode)
        return {
            "target": target,
            "mode": mode,
            "source_language": source_language,
            "normalized_prompt": normalized_prompt,
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

    def _lattice_bucket(self, prompt: str, target: str, mode: str, source_language: str) -> tuple[int, int, int, int]:
        digest = sha256(f"{prompt}|{target}|{mode}|{source_language}".encode("utf-8")).digest()
        return tuple(digest[i] % 12 for i in range(4))

    def _rag_store(self, prompt: str, output: str, target: str, mode: str, source_language: str) -> tuple[int, int, int, int]:
        bucket = self._lattice_bucket(prompt, target, mode, source_language)
        entries = self._rag_lattice.setdefault(bucket, [])
        entries.append({"prompt": prompt, "output": output, "target": target, "mode": mode, "source_language": source_language})
        if len(entries) > 64:
            del entries[:-64]
        return bucket

    def rag_retrieve(self, prompt: str, target: str, mode: str = "gameplay", source_language: str = "english", limit: int = 3) -> list[dict[str, str]]:
        bucket = self._lattice_bucket(prompt, target, mode, source_language)
        return list(self._rag_lattice.get(bucket, [])[-limit:])

    def run_in_vm_sandbox(self, command: list[str], timeout_s: int = 20) -> tuple[bool, str]:
        if not command:
            return False, "No command provided"
        import tempfile

        with tempfile.TemporaryDirectory(prefix="nevora-vm-sandbox-") as td:
            try:
                result = subprocess.run(
                    command,
                    cwd=td,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
                if result.returncode == 0:
                    return True, output.strip() or "sandbox execution succeeded"
                return False, output.strip() or f"sandbox command failed with code {result.returncode}"
            except subprocess.TimeoutExpired:
                return False, f"sandbox command timed out after {timeout_s}s"

    def load_asset_library(self, asset_library_path: str) -> dict[str, list[dict[str, Any]]]:
        source = Path(asset_library_path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Asset library must be a JSON object keyed by engine")

        library: dict[str, list[dict[str, Any]]] = {}
        for engine in self.ASSET_ENGINES:
            entries = payload.get(engine, [])
            if not isinstance(entries, list):
                raise ValueError(f"Asset library entry '{engine}' must be a list")
            normalized: list[dict[str, Any]] = []
            for item in entries:
                if not isinstance(item, dict):
                    continue
                normalized.append({
                    "id": str(item.get("id", "")),
                    "name": str(item.get("name", "")),
                    "tags": [str(t).lower() for t in item.get("tags", []) if str(t).strip()],
                    "path": str(item.get("path", "")),
                })
            library[engine] = normalized
        return library

    def _select_assets_for_prompt(
        self,
        normalized_prompt: str,
        engine: str,
        asset_library: dict[str, list[dict[str, Any]]],
        asset_budget: int = 5,
    ) -> list[dict[str, Any]]:
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", normalized_prompt.lower()))
        candidates = asset_library.get(engine, [])
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in candidates:
            score = 0
            text_tokens = set(re.findall(r"[a-zA-Z0-9_]+", f"{item.get('name','')} {item.get('id','')}".lower()))
            tags = set(str(t).lower() for t in item.get("tags", []))
            score += len(tokens & text_tokens)
            score += 2 * len(tokens & tags)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[: max(0, asset_budget)]]

    def translate_with_asset_library(
        self,
        prompt: str,
        target: str,
        engine: str,
        asset_library_path: str,
        mode: str = "gameplay",
        source_language: str = "english",
        asset_budget: int = 5,
        use_rag_cache: bool = False,
    ) -> dict[str, Any]:
        normalized_engine = engine.strip().lower()
        if normalized_engine not in self.ASSET_ENGINES:
            raise ValueError(f"Unsupported engine '{engine}'. Supported: {', '.join(sorted(self.ASSET_ENGINES))}")

        normalized_prompt = self._normalize_prompt_language(prompt, source_language=source_language)
        library = self.load_asset_library(asset_library_path)
        selected_assets = self._select_assets_for_prompt(normalized_prompt, normalized_engine, library, asset_budget=asset_budget)
        asset_context = ""
        if selected_assets:
            asset_lines = [f"- {a.get('name') or a.get('id')} ({a.get('path')})" for a in selected_assets]
            asset_context = "\n\nAvailable assets:\n" + "\n".join(asset_lines)

        output = self.translate(
            prompt=normalized_prompt + asset_context,
            target=target,
            mode=mode,
            source_language="english",
            use_rag_cache=use_rag_cache,
        )
        return {
            "engine": normalized_engine,
            "target": target,
            "mode": mode,
            "source_language": source_language,
            "selected_assets": selected_assets,
            "output": output,
        }

    def export_engine_asset_manifest(
        self,
        prompt: str,
        target: str,
        engine: str,
        asset_library_path: str,
        output_path: str,
        mode: str = "gameplay",
        source_language: str = "english",
        asset_budget: int = 5,
    ) -> str:
        payload = self.translate_with_asset_library(
            prompt=prompt,
            target=target,
            engine=engine,
            asset_library_path=asset_library_path,
            mode=mode,
            source_language=source_language,
            asset_budget=asset_budget,
        )
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if payload["engine"] == "unreal":
            manifest = {
                "schema": "nevora.unreal.asset.manifest.v1",
                "prompt": prompt,
                "mode": mode,
                "target": target,
                "assets": payload["selected_assets"],
                "generated_code": payload["output"],
            }
        else:
            manifest = {
                "schema": "nevora.unity.asset.manifest.v1",
                "prompt": prompt,
                "mode": mode,
                "target": target,
                "assets": payload["selected_assets"],
                "generated_code": payload["output"],
            }
        destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return str(destination)

    def suggest_swarm_workers(self, batch_size: int, max_workers: int = 8) -> int:
        if batch_size <= 0:
            return 1
        cpu_count = os.cpu_count() or 2
        return max(1, min(batch_size, max_workers, cpu_count))

    def analyze_batch_report(self, report: dict[str, Any]) -> dict[str, Any]:
        success_rate = float(report.get("success_rate", 0.0))
        avg_elapsed_ms = float(report.get("avg_elapsed_ms", 0.0))
        verify_output_rate = float(report.get("verify_output_rate", 0.0))
        verify_build_rate = float(report.get("verify_build_rate", 0.0))
        total = int(report.get("total", 0))

        recommendations: list[str] = []
        if success_rate < 1.0:
            recommendations.append("Enable --batch-fail-fast while debugging failing prompts to reduce wasted runtime.")
        if verify_output_rate < 1.0:
            recommendations.append("Use clearer constraints in prompts and keep --batch-verify-output enabled to improve syntax quality.")
        if verify_build_rate < 1.0:
            recommendations.append("Review scaffold dependencies/toolchain and keep --batch-verify-build to gate buildability.")
        if avg_elapsed_ms > 250:
            recommendations.append("Consider increasing --swarm-workers and pre-warming prompts with --warm-cache-file.")
        if not recommendations:
            recommendations.append("Pipeline health is strong; increase batch volume and keep gates enabled for regression protection.")

        suggested_workers = self.suggest_swarm_workers(total)
        return {
            "total": total,
            "success_rate": success_rate,
            "avg_elapsed_ms": avg_elapsed_ms,
            "verify_output_rate": verify_output_rate,
            "verify_build_rate": verify_build_rate,
            "suggested_swarm_workers": suggested_workers,
            "recommendations": recommendations,
        }

    def benchmark_swarm_configs(
        self,
        items: list[dict[str, Any]],
        default_target: str,
        default_mode: str = "gameplay",
        worker_candidates: Optional[list[int]] = None,
        default_source_language: str = "english",
    ) -> dict[str, Any]:
        candidates = worker_candidates or [1, 2, 4]
        cleaned = sorted({max(1, int(c)) for c in candidates})
        timings: list[dict[str, Any]] = []

        for workers in cleaned:
            started = perf_counter()
            self.translate_batch(
                items,
                default_target=default_target,
                default_mode=default_mode,
                strict_safety=False,
                include_explain=False,
                fail_fast=False,
                verify_generated=False,
                verify_build=False,
                default_source_language=default_source_language,
                swarm_workers=workers,
            )
            elapsed_ms = round((perf_counter() - started) * 1000, 3)
            timings.append({"workers": workers, "elapsed_ms": elapsed_ms})

        best = min(timings, key=lambda t: t["elapsed_ms"]) if timings else {"workers": 1, "elapsed_ms": 0.0}
        return {
            "batch_size": len(items),
            "timings": timings,
            "best_workers": best["workers"],
            "best_elapsed_ms": best["elapsed_ms"],
        }

    def generate_assistant_runbook(
        self,
        prompt: str,
        target: str,
        mode: str = "gameplay",
        source_language: str = "english",
        engine: Optional[str] = None,
        has_asset_library: bool = False,
        batch_report: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        guide = self.assistant_guide(
            prompt=prompt,
            target=target,
            mode=mode,
            source_language=source_language,
            batch_report=batch_report,
        )
        checklist = [
            "Validate prompt contains clear entities/actions/conditions.",
            "Run --verify for single outputs before scaffolding.",
            "Use --batch-verify-output and --batch-verify-build for CI runs.",
        ]
        commands = [guide["suggested_command"]]

        if engine in self.ASSET_ENGINES:
            checklist.append("Ensure asset-library JSON is current and paths resolve in project.")
            if has_asset_library:
                commands.append(
                    f"python -m translator.cli --target {target} --engine {engine} --asset-library ./assets.json --prompt '{prompt}' --export-engine-manifest artifacts/{engine}_manifest.json"
                )

        if batch_report:
            analysis = self.analyze_batch_report(batch_report)
            checklist.extend(analysis["recommendations"])
            commands.append(
                f"python -m translator.cli --target {target} --batch-input batch.jsonl --swarm-workers {analysis['suggested_swarm_workers']} --batch-report artifacts/batch_report.json"
            )

        return {
            "title": "Nevora Assistant Runbook",
            "target": target,
            "mode": mode,
            "source_language": source_language,
            "engine": engine,
            "checklist": checklist,
            "commands": commands,
            "guide": guide,
        }

    def warm_plan_cache(self, prompts: list[str], mode: str = "gameplay", source_language: str = "english") -> dict[str, Any]:
        warmed = 0
        for prompt in prompts:
            normalized = self._normalize_prompt_language(prompt, source_language=source_language)
            key = (normalized, mode)
            if key in self._plan_cache:
                continue
            self.build_generation_plan(normalized, mode=mode)
            warmed += 1
        return {
            "mode": mode,
            "source_language": source_language,
            "warmed": warmed,
            "cache_size": len(self._plan_cache),
        }

    def assistant_guide(
        self,
        prompt: str,
        target: str,
        mode: str = "gameplay",
        source_language: str = "english",
        batch_report: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        normalized_prompt = self._normalize_prompt_language(prompt, source_language=source_language)
        plan = self.build_generation_plan(normalized_prompt, mode=mode)
        suggestions: list[str] = []

        if target.lower().strip() == "blueprint":
            suggestions.append("For Unreal Blueprint output, export graph payload with --export-uasset-json for editor-side import.")
        if target.lower().strip() == "csharp":
            suggestions.append("For Unity workflows, combine --engine unity with --asset-library to bind prompts to available prefabs/assets.")
        if source_language != "english":
            suggestions.append("Prompt is normalized from non-English input; add domain terms to token maps for better intent fidelity.")
        if len(plan.intent.actions) <= 1:
            suggestions.append("Add more action verbs in the prompt for richer generated behavior.")
        if "when" not in normalized_prompt.lower() and "if" not in normalized_prompt.lower():
            suggestions.append("Include explicit conditions (e.g., when/if) to improve event trigger structure.")

        perf_tip = "Enable --swarm-workers for batch speed and --enable-rag-cache for repeated related prompts."
        suggestions.append(perf_tip)

        quality_tip = "Use --verify, --verify-scaffold, and batch min-rate gates to enforce quality in CI."
        suggestions.append(quality_tip)

        report_summary = None
        report_advice = None
        if batch_report:
            report_summary = {
                "success_rate": batch_report.get("success_rate", 0.0),
                "avg_elapsed_ms": batch_report.get("avg_elapsed_ms", 0.0),
            }
            report_advice = self.analyze_batch_report(batch_report)
            if float(batch_report.get("success_rate", 0.0)) < 1.0:
                suggestions.append("Batch report indicates failures; inspect per-item errors and consider --batch-fail-fast during debugging.")

        safe_prompt = prompt.replace("'", "\\'")
        suggested_command = (
            f"python -m translator.cli --target {target} --mode {mode} "
            f"--prompt '{safe_prompt}' --verify --enable-rag-cache"
        )

        return {
            "prompt": prompt,
            "normalized_prompt": normalized_prompt,
            "target": target,
            "mode": mode,
            "source_language": source_language,
            "intent_preview": {
                "entities": plan.intent.entities,
                "actions": plan.intent.actions,
                "conditions": plan.intent.conditions,
                "outputs": plan.intent.outputs,
            },
            "suggestions": suggestions,
            "report_summary": report_summary,
            "report_advice": report_advice,
            "suggested_command": suggested_command,
        }

    def translate(
        self,
        prompt: str,
        target: str,
        mode: str = "gameplay",
        context: Optional[str] = None,
        refine: bool = False,
        strict_safety: bool = False,
        source_language: str = "english",
        use_rag_cache: bool = False,
    ) -> str:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Supported: {', '.join(sorted(self.MODES))}")

        normalized_target = target.strip().lower()
        if normalized_target not in self.supported_targets:
            supported = ", ".join(sorted(self.supported_targets))
            raise ValueError(f"Unsupported target '{target}'. Supported: {supported}")

        normalized_prompt = self._normalize_prompt_language(prompt, source_language=source_language)
        normalized_context = None
        if context is not None:
            normalized_context = self._normalize_prompt_language(context, source_language=source_language)

        combined_prompt = normalized_prompt
        if refine and normalized_context:
            combined_prompt = f"{normalized_prompt}\n\nPrevious output context:\n{normalized_context}"

        self._enforce_safety(combined_prompt, strict_safety=strict_safety)
        rag_context = ""
        if use_rag_cache:
            neighbors = self.rag_retrieve(combined_prompt, normalized_target, mode=mode, source_language=source_language, limit=2)
            if neighbors:
                rag_context = "\n\nRAG memory hints:\n" + "\n".join(n["output"][:240] for n in neighbors)
        plan = self.build_generation_plan(combined_prompt + rag_context, mode=mode)
        renderer = self.renderers[normalized_target]
        output = renderer.render(combined_prompt, plan.intent, mode=mode, plan=plan)
        self._enforce_safety(output, strict_safety=strict_safety)
        if use_rag_cache:
            self._rag_store(combined_prompt, output, normalized_target, mode, source_language)
        return output

    def _slug(self, text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        return cleaned[:48] or "item"

    def _translate_batch_item(
        self,
        idx: int,
        item: dict[str, Any],
        default_target: str,
        default_mode: str,
        strict_safety: bool,
        verify_generated: bool,
        verify_build: bool,
        default_source_language: str,
        include_explain: bool,
        artifacts_root: Path | None,
    ) -> dict[str, Any]:
        prompt = str(item.get("prompt", "")).strip()
        target = str(item.get("target", default_target)).strip()
        mode = str(item.get("mode", default_mode)).strip()
        context = item.get("context")
        refine = bool(item.get("refine", False))
        source_language = str(item.get("source_language", default_source_language)).strip().lower()

        started_at = perf_counter()
        output = self.translate(
            prompt=prompt,
            target=target,
            mode=mode,
            context=context,
            refine=refine,
            strict_safety=strict_safety,
            source_language=source_language,
            use_rag_cache=True,
        )
        elapsed_ms = round((perf_counter() - started_at) * 1000, 3)
        payload: dict[str, Any] = {
            "index": idx,
            "ok": True,
            "target": target,
            "mode": mode,
            "source_language": source_language,
            "resolved_provider": self._last_resolved_provider,
            "output": output,
            "lattice_bucket": list(self._lattice_bucket(prompt, target, mode, source_language)),
            "elapsed_ms": elapsed_ms,
        }

        if verify_generated:
            verify_ok, verify_message = self.verify_output(output, target)
            payload["verify_output_ok"] = verify_ok
            payload["verify_output_message"] = verify_message

        if include_explain:
            payload["explain"] = self.explain_plan(prompt, target=target, mode=mode, source_language=source_language)

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

        return payload

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
        default_source_language: str = "english",
        swarm_workers: int = 1,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        artifacts_root = Path(artifact_dir) if artifact_dir else None
        if artifacts_root:
            artifacts_root.mkdir(parents=True, exist_ok=True)

        def _safe_item(idx: int, item: dict[str, Any]) -> dict[str, Any]:
            try:
                return self._translate_batch_item(
                    idx,
                    item,
                    default_target,
                    default_mode,
                    strict_safety,
                    verify_generated,
                    verify_build,
                    default_source_language,
                    include_explain,
                    artifacts_root,
                )
            except Exception as exc:
                target = str(item.get("target", default_target)).strip()
                mode = str(item.get("mode", default_mode)).strip()
                source_language = str(item.get("source_language", default_source_language)).strip().lower()
                return {
                    "index": idx,
                    "ok": False,
                    "target": target,
                    "mode": mode,
                    "source_language": source_language,
                    "resolved_provider": self._last_resolved_provider,
                    "error": str(exc),
                }

        if swarm_workers <= 1 or fail_fast:
            for idx, item in enumerate(items):
                payload = _safe_item(idx, item)
                results.append(payload)
                if fail_fast and not payload.get("ok"):
                    break
            return results

        with ThreadPoolExecutor(max_workers=swarm_workers) as executor:
            futures = {executor.submit(_safe_item, idx, item): idx for idx, item in enumerate(items)}
            ordered: dict[int, dict[str, Any]] = {}
            for future in as_completed(futures):
                payload = future.result()
                ordered[payload["index"]] = payload
            for idx in range(len(items)):
                if idx in ordered:
                    results.append(ordered[idx])
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
        source_language_counts: dict[str, int] = {}
        lattice_bucket_counts: dict[str, int] = {}
        for item in batch_results:
            target = str(item.get("target", "unknown"))
            provider = str(item.get("resolved_provider", "unknown"))
            source_language = str(item.get("source_language", "unknown"))
            target_counts[target] = target_counts.get(target, 0) + 1
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            source_language_counts[source_language] = source_language_counts.get(source_language, 0) + 1
            bucket = item.get("lattice_bucket")
            if isinstance(bucket, list) and len(bucket) == 4:
                key = "x".join(str(v) for v in bucket)
                lattice_bucket_counts[key] = lattice_bucket_counts.get(key, 0) + 1

        total = len(batch_results)
        elapsed_values = sorted(float(item.get("elapsed_ms", 0.0)) for item in batch_results if item.get("ok") and item.get("elapsed_ms") is not None)
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
            "source_language_counts": source_language_counts,
            "lattice_shape": list(self.lattice_shape),
            "lattice_bucket_counts": lattice_bucket_counts,
            "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 3) if elapsed_values else 0.0,
            "p95_elapsed_ms": round(elapsed_values[min(len(elapsed_values) - 1, int(0.95 * (len(elapsed_values) - 1)))], 3) if elapsed_values else 0.0,
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
