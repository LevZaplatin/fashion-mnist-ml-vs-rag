from fashion_compare.evaluation.progress import iter_with_progress, progress_points


def test_progress_points_include_final_total() -> None:
    assert 100 in progress_points(100)


def test_iter_with_progress_reports_percent_at_checkpoints() -> None:
    updates = [(processed, percent) for processed, should_log, percent in iter_with_progress(10, steps=2) if should_log]

    assert updates == [(5, 50.0), (10, 100.0)]
