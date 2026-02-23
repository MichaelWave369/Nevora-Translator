from __future__ import annotations

import json
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from translator.core import EnglishToCodeTranslator


def main() -> None:
    dataset = json.loads(Path("eval/prompts/golden.json").read_text(encoding="utf-8"))
    translator = EnglishToCodeTranslator(planner_provider="heuristic")

    total = len(dataset)
    passed = 0
    structure_score = 0
    intent_score = 0
    deterministic_score = 0
    buildability_score = 0

    for case in dataset:
        mode = case.get("mode", "gameplay")
        out_a = translator.translate(case["prompt"], case["target"], mode=mode)
        out_b = translator.translate(case["prompt"], case["target"], mode=mode)

        structure_ok = all(token in out_a for token in case["must_contain"])
        if structure_ok:
            structure_score += 1

        lowered = out_a.lower()
        token_hits = sum(1 for token in case.get("intent_tokens", []) if token.lower() in lowered)
        token_total = max(1, len(case.get("intent_tokens", [])))
        intent_case_score = token_hits / token_total
        intent_score += intent_case_score

        deterministic_ok = out_a == out_b
        if deterministic_ok:
            deterministic_score += 1

        with tempfile.TemporaryDirectory(prefix="nevora-eval-") as td:
            translator.scaffold_project(case["prompt"], case["target"], output_dir=td, mode=mode)
            build_ok, _ = translator.verify_scaffold_build(td, case["target"])
            if build_ok:
                buildability_score += 1

        case_ok = structure_ok and intent_case_score >= 0.5 and deterministic_ok
        if case_ok:
            passed += 1

        print(
            f"[{'PASS' if case_ok else 'FAIL'}] target={case['target']} "
            f"structure={'ok' if structure_ok else 'fail'} "
            f"intent={intent_case_score:.2f} deterministic={'ok' if deterministic_ok else 'fail'} "
            f"build={'ok' if build_ok else 'warn'}"
        )

    print(f"\nPass rate: {passed}/{total}")
    print(f"Structure score: {structure_score}/{total}")
    print(f"Intent coverage score: {intent_score:.2f}/{total:.2f}")
    print(f"Determinism score: {deterministic_score}/{total}")
    print(f"Buildability score: {buildability_score}/{total}")


if __name__ == "__main__":
    main()
