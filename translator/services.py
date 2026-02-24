from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class BatchReportService:
    lattice_shape: tuple[int, int, int, int]

    def build_summary(self, batch_results: list[dict[str, Any]]) -> dict[str, Any]:
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

        return {
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


def validate_ordered_results(results: list[dict[str, Any]]) -> None:
    expected = list(range(len(results)))
    actual = [int(item.get("index", -1)) for item in results]
    if actual != expected:
        raise RuntimeError(f"translate_batch ordering mismatch: expected {expected} got {actual}")
