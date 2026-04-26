from time import perf_counter
from typing import Any

import numpy as np
import torch

from fashion_compare.classifier.predict import load_classifier, predict_image
from fashion_compare.data import load_datasets
from fashion_compare.evaluation.progress import iter_with_progress
from fashion_compare.preprocessing import image_to_unit_array
from fashion_compare.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_classifier_predictions(limit: int | None = None) -> tuple[list[int], list[int], list[float]]:
    from fashion_compare.config import get_settings

    settings = get_settings()
    _, _, test = load_datasets(settings, normalized=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_classifier(device)
    y_true: list[int] = []
    y_pred: list[int] = []
    latencies: list[float] = []
    count = min(limit or len(test), len(test))
    for idx, (processed, should_log, percent) in zip(range(count), iter_with_progress(count), strict=True):
        image, label = test[idx]
        array = image_to_unit_array(image)
        start = perf_counter()
        prediction = predict_image(array, model=model, device=device)
        latencies.append((perf_counter() - start) * 1000.0)
        y_true.append(int(label))
        y_pred.append(int(prediction["label_id"]))
        if should_log:
            logger.info("classifier evaluation progress: %s/%s (%.1f%%)", processed, count, percent)
    return y_true, y_pred, latencies
