import numpy as np

import fashion_compare.models  # noqa: F401
from fashion_compare.rag.retrieve import embed_query


def test_embed_query_works_for_blur784() -> None:
    image = np.random.rand(28, 28).astype(np.float32)
    vector = embed_query(image, mode="blur784")
    assert len(vector) == 784


def test_embed_query_works_for_hog() -> None:
    image = np.random.rand(28, 28).astype(np.float32)
    vector = embed_query(image, mode="hog")
    assert len(vector) > 0
