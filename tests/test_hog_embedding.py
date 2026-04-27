import numpy as np


def test_hog_embed_returns_l2_normalized_vector() -> None:
    from fashion_compare.models.hog import embed_hog

    image = np.random.randint(0, 255, (28, 28), dtype=np.uint8).astype(np.float32)
    vector = embed_hog(image)
    assert vector.ndim == 1
    assert vector.dtype == np.float32
    norm = float(np.linalg.norm(vector))
    assert abs(norm - 1.0) < 1e-5


def test_hog_blur_embed_returns_l2_normalized_vector() -> None:
    from fashion_compare.models.hog import embed_hog_blur

    image = np.random.randint(0, 255, (28, 28), dtype=np.uint8).astype(np.float32)
    vector = embed_hog_blur(image)
    assert vector.ndim == 1
    assert vector.dtype == np.float32
    norm = float(np.linalg.norm(vector))
    assert abs(norm - 1.0) < 1e-5


def test_hog_and_hog_blur_have_same_dimension() -> None:
    from fashion_compare.models.hog import embed_hog, embed_hog_blur

    image = np.random.randint(0, 255, (28, 28), dtype=np.uint8).astype(np.float32)
    assert embed_hog(image).shape == embed_hog_blur(image).shape


def test_hog_modes_registered() -> None:
    import fashion_compare.models.hog  # noqa: F401
    from fashion_compare.models.registry import get_mode

    hog_mode = get_mode("hog")
    assert hog_mode.requires_fit is False
    hog_blur_mode = get_mode("hog_blur")
    assert hog_blur_mode.requires_fit is False
    assert hog_mode.dim == hog_blur_mode.dim
