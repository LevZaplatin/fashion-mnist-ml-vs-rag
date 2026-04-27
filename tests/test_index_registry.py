import numpy as np

import fashion_compare.models  # noqa: F401
from fashion_compare.rag.index import build_vector


def test_build_vector_works_for_blur784() -> None:
    image = np.random.rand(28, 28).astype(np.float32)
    vector = build_vector(image, mode="blur784")
    assert vector.shape == (784,)
    norm = float(np.linalg.norm(vector))
    assert abs(norm - 1.0) < 1e-5


def test_build_vector_works_for_hog() -> None:
    image = np.random.rand(28, 28).astype(np.float32)
    vector = build_vector(image, mode="hog")
    assert vector.ndim == 1
