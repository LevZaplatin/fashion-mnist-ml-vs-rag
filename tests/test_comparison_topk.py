def test_metrics_key_format_includes_top_k() -> None:
    mode = "hog"
    k = 11
    key = f"rag_{mode}_k{k}"
    assert key == "rag_hog_k11"


def test_write_comparison_includes_top_k_column() -> None:
    import json
    import tempfile
    from pathlib import Path

    from fashion_compare.evaluation.compare import write_comparison

    results = {
        "classifier": {
            "system": "classifier",
            "sample_count": 100,
            "accuracy": 0.9,
            "macro_f1": 0.9,
            "top_k": "-",
            "latency_ms": {"average": 1.0, "p50": 1.0, "p95": 2.0},
        },
        "rag_hog_k7": {
            "system": "rag_hog_k7",
            "sample_count": 100,
            "accuracy": 0.85,
            "macro_f1": 0.84,
            "top_k": 7,
            "latency_ms": {"average": 5.0, "p50": 4.0, "p95": 8.0},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        reports_dir = Path(tmp)
        write_comparison(results, reports_dir)
        md = (reports_dir / "comparison.md").read_text()
        assert "Top K" in md
        data = json.loads((reports_dir / "comparison.json").read_text())
        assert "rag_hog_k7" in data
