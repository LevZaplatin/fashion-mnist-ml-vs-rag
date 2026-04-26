from collections.abc import Iterator


def progress_points(total: int, steps: int = 20) -> set[int]:
    if total <= 0:
        return set()
    return {max(1, round(total * idx / steps)) for idx in range(1, steps + 1)}


def iter_with_progress(total: int, steps: int = 20) -> Iterator[tuple[int, bool, float]]:
    checkpoints = progress_points(total, steps=steps)
    for processed in range(1, total + 1):
        yield processed, processed in checkpoints, processed * 100.0 / total

