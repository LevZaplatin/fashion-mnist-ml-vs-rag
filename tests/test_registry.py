import numpy as np
import pytest

from fashion_compare.models.registry import (
    EmbeddingMode,
    ModeNotFoundError,
    all_mode_names,
    get_mode,
    register,
)


def _dummy_embed(image: np.ndarray) -> np.ndarray:
    return np.zeros(16, dtype=np.float32)


def test_register_and_get_mode() -> None:
    mode = EmbeddingMode(
        name="test_mode",
        dim=16,
        embed_fn=_dummy_embed,
        requires_fit=False,
        fit_fn=None,
        load_fn=None,
    )
    register(mode)
    assert get_mode("test_mode") is mode


def test_get_mode_raises_for_unknown() -> None:
    with pytest.raises(ModeNotFoundError):
        get_mode("no_such_mode_xyz")


def test_all_mode_names_includes_registered() -> None:
    mode = EmbeddingMode(
        name="test_listed",
        dim=8,
        embed_fn=_dummy_embed,
        requires_fit=False,
        fit_fn=None,
        load_fn=None,
    )
    register(mode)
    assert "test_listed" in all_mode_names()


def test_raw784_is_registered() -> None:
    import fashion_compare.models.embeddings  # noqa: F401 — triggers registration

    mode = get_mode("raw784")
    assert mode.dim == 784
    assert mode.requires_fit is False


def test_raw784_embed_fn_returns_correct_shape() -> None:
    import fashion_compare.models.embeddings  # noqa: F401

    mode = get_mode("raw784")
    image = np.random.randint(0, 255, (28, 28), dtype=np.uint8).astype(np.float32)
    vector = mode.embed_fn(image)
    assert vector.shape == (784,)
    assert vector.dtype == np.float32
    norm = float(np.linalg.norm(vector))
    assert abs(norm - 1.0) < 1e-5


def test_cnn_embedding_is_registered() -> None:
    import fashion_compare.models.embeddings  # noqa: F401

    mode = get_mode("cnn_embedding")
    assert mode.dim == 128
    assert mode.requires_fit is False
