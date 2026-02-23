from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import EnglishToCodeTranslator


def _load_batch_items(path: str) -> list[dict]:
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if source.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("Batch input JSON must be a list of objects")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="English-to-code translator")
    parser.add_argument("--target", required=True)
    parser.add_argument("--prompt", required=False, help="English description to translate")
    parser.add_argument(
        "--mode",
        default="gameplay",
        choices=["gameplay", "automation", "video-processing", "web-backend"],
    )
    parser.add_argument(
        "--planner-provider",
        default="auto",
        choices=["auto", "heuristic", "openai"],
        help="Select planner backend provider",
    )
    parser.add_argument("--strict-safety", action="store_true", help="Block unsafe content patterns")
    parser.add_argument("--context-file", help="Optional previous output context for iterative refinement")
    parser.add_argument("--refine", action="store_true", help="Enable context-aware iterative refinement")
    parser.add_argument("--verify", action="store_true", help="Run target-specific syntax checks")
    parser.add_argument("--scaffold-dir", help="Generate a starter project scaffold in this folder")
    parser.add_argument("--verify-scaffold", action="store_true", help="Verify scaffold file/build structure")
    parser.add_argument(
        "--verify-scaffold-build",
        action="store_true",
        help="Run tool-based build checks on scaffold outputs when supported",
    )
    parser.add_argument("--export-uasset-json", help="Optional path to export Unreal Blueprint graph contract JSON")
    parser.add_argument("--blueprint-name", default="BP_GeneratedFeature")
    parser.add_argument("--explain-plan", action="store_true", help="Print planner/IR explanation as JSON")
    parser.add_argument("--explain-plan-file", help="Optional file path to write explain-plan JSON")
    parser.add_argument("--batch-input", help="Path to JSON/JSONL batch prompts")
    parser.add_argument("--batch-report", help="Path to write batch run report JSON")
    parser.add_argument("--batch-fail-fast", action="store_true", help="Stop batch processing on first failed item")
    parser.add_argument(
        "--batch-min-success-rate",
        type=float,
        help="Optional required minimum batch success rate (0.0-1.0); exits non-zero if unmet",
    )
    parser.add_argument("--batch-artifact-dir", help="Folder to store per-item batch output artifacts")
    parser.add_argument("--batch-include-explain", action="store_true", help="Include explain payload for each batch item")
    parser.add_argument("--batch-verify-output", action="store_true", help="Run verify_output for each successful batch item")
    parser.add_argument("--batch-verify-build", action="store_true", help="Run scaffold build verification for each successful batch item")
    parser.add_argument(
        "--batch-min-verify-output-rate",
        type=float,
        help="Optional minimum verify_output pass rate (0.0-1.0)",
    )
    parser.add_argument(
        "--batch-min-verify-build-rate",
        type=float,
        help="Optional minimum verify_build pass rate (0.0-1.0)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    translator = EnglishToCodeTranslator(planner_provider=args.planner_provider)

    if args.batch_input:
        items = _load_batch_items(args.batch_input)
        results = translator.translate_batch(
            items,
            default_target=args.target,
            default_mode=args.mode,
            strict_safety=args.strict_safety,
            artifact_dir=args.batch_artifact_dir,
            include_explain=args.batch_include_explain,
            fail_fast=args.batch_fail_fast,
            verify_generated=args.batch_verify_output,
            verify_build=args.batch_verify_build,
        )
        print(json.dumps(results, indent=2))
        if args.batch_report:
            destination = translator.write_batch_report(results, args.batch_report)
            print(f"\n[batch-report] written: {destination}")

        if args.batch_min_success_rate is not None:
            if not (0.0 <= args.batch_min_success_rate <= 1.0):
                raise ValueError("--batch-min-success-rate must be between 0.0 and 1.0")
            total = len(results)
            ok = sum(1 for r in results if r.get("ok"))
            success_rate = (ok / total) if total else 0.0
            if success_rate < args.batch_min_success_rate:
                raise SystemExit(
                    f"Batch success rate gate failed: {success_rate:.4f} < {args.batch_min_success_rate:.4f}"
                )
            print(f"\n[batch-gate:ok] success_rate={success_rate:.4f}")

        if args.batch_min_verify_output_rate is not None:
            if not (0.0 <= args.batch_min_verify_output_rate <= 1.0):
                raise ValueError("--batch-min-verify-output-rate must be between 0.0 and 1.0")
            total = len(results)
            ok = sum(1 for r in results if r.get("verify_output_ok") is True)
            verify_rate = (ok / total) if total else 0.0
            if verify_rate < args.batch_min_verify_output_rate:
                raise SystemExit(
                    f"Batch verify-output gate failed: {verify_rate:.4f} < {args.batch_min_verify_output_rate:.4f}"
                )
            print(f"\n[batch-verify-output-gate:ok] rate={verify_rate:.4f}")

        if args.batch_min_verify_build_rate is not None:
            if not (0.0 <= args.batch_min_verify_build_rate <= 1.0):
                raise ValueError("--batch-min-verify-build-rate must be between 0.0 and 1.0")
            total = len(results)
            ok = sum(1 for r in results if r.get("verify_build_ok") is True)
            verify_rate = (ok / total) if total else 0.0
            if verify_rate < args.batch_min_verify_build_rate:
                raise SystemExit(
                    f"Batch verify-build gate failed: {verify_rate:.4f} < {args.batch_min_verify_build_rate:.4f}"
                )
            print(f"\n[batch-verify-build-gate:ok] rate={verify_rate:.4f}")
        return

    if not args.prompt:
        raise ValueError("--prompt is required unless --batch-input is provided")

    context = None
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")

    output = translator.translate(
        prompt=args.prompt,
        target=args.target,
        mode=args.mode,
        context=context,
        refine=args.refine,
        strict_safety=args.strict_safety,
    )
    print(output)

    if args.explain_plan or args.explain_plan_file:
        explanation = translator.explain_plan(args.prompt, target=args.target, mode=args.mode)
        if args.explain_plan:
            print("\n[plan]")
            print(json.dumps(explanation, indent=2))
        if args.explain_plan_file:
            destination = Path(args.explain_plan_file)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(explanation, indent=2), encoding="utf-8")
            print(f"\n[plan-file] written: {destination}")

    if args.verify:
        ok, message = translator.verify_output(output, args.target)
        status = "ok" if ok else "warn"
        print(f"\n[verify:{status}] {message}")

    scaffold_root = None
    if args.scaffold_dir:
        scaffold_root = translator.scaffold_project(args.prompt, target=args.target, output_dir=args.scaffold_dir, mode=args.mode)
        print(f"\n[scaffold] created at: {scaffold_root}")

    if args.verify_scaffold:
        if scaffold_root is None and args.scaffold_dir:
            scaffold_root = args.scaffold_dir
        if scaffold_root is None:
            print("\n[verify-scaffold:warn] --verify-scaffold used without --scaffold-dir")
        else:
            ok, message = translator.verify_scaffold(scaffold_root, args.target)
            status = "ok" if ok else "warn"
            print(f"\n[verify-scaffold:{status}] {message}")

    if args.verify_scaffold_build:
        if scaffold_root is None and args.scaffold_dir:
            scaffold_root = args.scaffold_dir
        if scaffold_root is None:
            print("\n[verify-scaffold-build:warn] --verify-scaffold-build used without --scaffold-dir")
        else:
            ok, message = translator.verify_scaffold_build(scaffold_root, args.target)
            status = "ok" if ok else "warn"
            print(f"\n[verify-scaffold-build:{status}] {message}")

    if args.export_uasset_json:
        export_path = translator.export_unreal_uasset_payload(
            prompt=args.prompt,
            output_path=args.export_uasset_json,
            blueprint_name=args.blueprint_name,
            mode=args.mode,
        )
        print(f"\n[export] Unreal graph payload written to: {export_path}")


if __name__ == "__main__":
    main()
