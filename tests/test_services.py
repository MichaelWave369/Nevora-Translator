from translator.services import BatchReportService, validate_ordered_results


def test_validate_ordered_results_passes() -> None:
    validate_ordered_results([
        {"index": 0, "ok": True},
        {"index": 1, "ok": True},
    ])


def test_batch_report_service_summary_contains_counts() -> None:
    service = BatchReportService((12, 12, 12, 12))
    summary = service.build_summary([
        {"index": 0, "ok": True, "target": "python", "resolved_provider": "heuristic", "source_language": "english", "elapsed_ms": 10.0},
        {"index": 1, "ok": False, "target": "cpp", "resolved_provider": "heuristic-fallback", "source_language": "english"},
    ])
    assert summary["total"] == 2
    assert summary["ok"] == 1
    assert summary["failed"] == 1
    assert summary["target_counts"]["python"] == 1
