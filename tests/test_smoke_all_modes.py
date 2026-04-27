import numpy as np

import fashion_compare.models  # noqa: F401
from fashion_compare.models.registry import all_mode_names, get_mode


EXPECTED_MODES = {
    "raw784", "blur784", "hog", "hog_blur",
    "pca64", "pca128", "pca256",
    "autoencoder64", "autoencoder128", "autoencoder256",
    "cnn_embedding",
}

TESTABLE_MODES = {"raw784", "blur784", "hog", "hog_blur"}


def test_all_preprocessing_modes_produce_correct_vectors() -> None:
    """Verify every mode that does not require fit or a CNN checkpoint."""
    image = np.random.RandomState(0).rand(28, 28).astype(np.float32)

    for name in TESTABLE_MODES:
        mode = get_mode(name)
        vector = mode.embed_fn(image)
        assert vector.shape == (mode.dim,), f"{name}: expected dim {mode.dim}, got {vector.shape}"
        assert vector.dtype == np.float32, f"{name}: expected float32, got {vector.dtype}"
        norm = float(np.linalg.norm(vector))
        assert abs(norm - 1.0) < 1e-4, f"{name}: expected unit norm, got {norm}"


def test_all_expected_modes_are_registered() -> None:
    registered = set(all_mode_names())
    assert EXPECTED_MODES <= registered, f"Missing: {EXPECTED_MODES - registered}"
