from time import perf_counter

from fashion_compare.data import load_datasets
from fashion_compare.evaluation.progress import iter_with_progress
from fashion_compare.preprocessing import image_to_unit_array
from fashion_compare.rag.retrieve import predict_rag
from fashion_compare.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_rag_predictions(mode: str, top_k: int, limit: int | None = None) -> tuple[list[int], list[int], list[float]]:
    from fashion_compare.config import get_settings

    settings = get_settings()
    _, _, test = load_datasets(settings, normalized=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    latencies: list[float] = []
    count = min(limit or len(test), len(test))
    for idx, (processed, should_log, percent) in zip(range(count), iter_with_progress(count), strict=True):
        image, label = test[idx]
        array = image_to_unit_array(image)
        start = perf_counter()
        prediction = predict_rag(array, mode=mode, top_k=top_k)
        latencies.append((perf_counter() - start) * 1000.0)
        y_true.append(int(label))
        y_pred.append(int(prediction["label_id"]))
        if should_log:
            logger.info("rag evaluation progress mode=%s: %s/%s (%.1f%%)", mode, processed, count, percent)
    return y_true, y_pred, latencies
