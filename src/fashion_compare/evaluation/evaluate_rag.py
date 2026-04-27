from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Any

from fashion_compare.data import load_datasets
from fashion_compare.evaluation.progress import progress_points
from fashion_compare.preprocessing import image_to_unit_array
from fashion_compare.rag.retrieve import predict_rag
from fashion_compare.utils.logging import get_logger

logger = get_logger(__name__)


def _predict_one(idx: int, image: Any, label: int, mode: str, top_k: int) -> tuple[int, int, int, float]:
    array = image_to_unit_array(image)
    start = perf_counter()
    prediction = predict_rag(array, mode=mode, top_k=top_k)
    latency = (perf_counter() - start) * 1000.0
    return idx, int(label), int(prediction["label_id"]), latency


def evaluate_rag_predictions(
    mode: str, top_k: int, limit: int | None = None, workers: int | None = None
) -> tuple[list[int], list[int], list[float]]:
    from fashion_compare.config import get_settings

    settings = get_settings()
    workers = workers or settings.eval_workers
    _, _, test = load_datasets(settings, normalized=False)
    count = min(limit or len(test), len(test))
    checkpoints = progress_points(count)

    results: list[tuple[int, int, int, float]] = [None] * count  # type: ignore[list-item]
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for idx in range(count):
            image, label = test[idx]
            future = pool.submit(_predict_one, idx, image, int(label), mode, top_k)
            futures[future] = idx

        for future in as_completed(futures):
            result = future.result()
            results[result[0]] = result
            completed += 1
            if completed in checkpoints:
                logger.info(
                    "rag evaluation progress mode=%s: %s/%s (%.1f%%)",
                    mode, completed, count, completed * 100.0 / count,
                )

    y_true = [r[1] for r in results]
    y_pred = [r[2] for r in results]
    latencies = [r[3] for r in results]
    return y_true, y_pred, latencies
