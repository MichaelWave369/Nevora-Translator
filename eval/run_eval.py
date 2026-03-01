from __future__ import annotations

import json
import tempfile
from pathlib import Path

from translator.core import EnglishToCodeTranslator


def _simple_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    hit = 0
    for token in hyp_tokens:
        if ref_counts.get(token, 0) > 0:
            hit += 1
            ref_counts[token] -= 1

    precision = hit / len(hyp_tokens)
    brevity_penalty = min(1.0, len(hyp_tokens) / len(ref_tokens))
    return precision * brevity_penalty


def _syntax_valid(target: str, output: str) -> bool:
    if target == "python":
        try:
            compile(output, "<generated>", "exec")
            return True
        except Exception:
            return False
    if target in {"cpp", "csharp", "javascript", "gdscript", "blueprint"}:
        return len(output.strip()) > 0
    return True


def _dataset_path() -> Path:
    return Path(__file__).resolve().parent / "prompts" / "golden.json"


def main() -> None:
    dataset = json.loads(_dataset_path().read_text(encoding="utf-8"))
    translator = EnglishToCodeTranslator(planner_provider="heuristic")

    total = len(dataset)
    passed = 0
    structure_score = 0
    intent_score = 0.0
    deterministic_score = 0
    buildability_score = 0
    syntax_valid_score = 0
    bleu_total = 0.0

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

        syntax_ok = _syntax_valid(case["target"], out_a)
        if syntax_ok:
            syntax_valid_score += 1

        reference = case.get("reference", " ".join(case.get("must_contain", [])))
        bleu = _simple_bleu(reference, out_a)
        bleu_total += bleu

        with tempfile.TemporaryDirectory(prefix="nevora-eval-") as td:
            translator.scaffold_project(case["prompt"], case["target"], output_dir=td, mode=mode)
            build_ok, _ = translator.verify_scaffold_build(td, case["target"])
            if build_ok:
                buildability_score += 1

        case_ok = structure_ok and intent_case_score >= 0.5 and deterministic_ok and syntax_ok
        if case_ok:
            passed += 1

        print(
            f"[{'PASS' if case_ok else 'FAIL'}] target={case['target']} "
            f"structure={'ok' if structure_ok else 'fail'} "
            f"intent={intent_case_score:.2f} deterministic={'ok' if deterministic_ok else 'fail'} "
            f"syntax={'ok' if syntax_ok else 'fail'} bleu={bleu:.2f} "
            f"build={'ok' if build_ok else 'warn'}"
        )

    print(f"\nPass rate: {passed}/{total}")
    print(f"Structure score: {structure_score}/{total}")
    print(f"Intent coverage score: {intent_score:.2f}/{total:.2f}")
    print(f"Determinism score: {deterministic_score}/{total}")
    print(f"Syntax validity score: {syntax_valid_score}/{total}")
    print(f"BLEU-like score: {bleu_total/total:.2f}")
    print(f"Buildability score: {buildability_score}/{total}")


if __name__ == "__main__":
    main()
