import numpy as np


def test_blur784_embed_returns_correct_shape_and_norm() -> None:
    from fashion_compare.models.blur import embed_blur784

    image = np.random.randint(0, 255, (28, 28), dtype=np.uint8).astype(np.float32)
    vector = embed_blur784(image)
    assert vector.shape == (784,)
    assert vector.dtype == np.float32
    norm = float(np.linalg.norm(vector))
    assert abs(norm - 1.0) < 1e-5


def test_blur784_is_registered() -> None:
    import fashion_compare.models.blur  # noqa: F401
    from fashion_compare.models.registry import get_mode

    mode = get_mode("blur784")
    assert mode.dim == 784
    assert mode.requires_fit is False


def test_blur784_differs_from_raw784() -> None:
    from fashion_compare.models.blur import embed_blur784
    from fashion_compare.models.embeddings import embed_raw784

    image = np.random.RandomState(42).randint(0, 255, (28, 28)).astype(np.float32)
    raw = embed_raw784(image)
    blurred = embed_blur784(image)
    assert not np.allclose(raw, blurred, atol=1e-6)
