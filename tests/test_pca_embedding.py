import tempfile
from pathlib import Path

import numpy as np


def _make_train_images(n: int = 200) -> np.ndarray:
    rng = np.random.RandomState(42)
    return rng.rand(n, 28, 28).astype(np.float32)


def test_pca_fit_and_embed_roundtrip() -> None:
    from fashion_compare.models.pca import embed_pca, fit_pca, load_pca

    images = _make_train_images()
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        fit_pca(images, n_components=64, artifact_dir=artifact_dir)
        assert (artifact_dir / "pca_model.pkl").exists()

        pca = load_pca(artifact_dir)
        image = np.random.rand(28, 28).astype(np.float32)
        vector = embed_pca(image, pca=pca)
        assert vector.shape == (64,)
        assert vector.dtype == np.float32
        norm = float(np.linalg.norm(vector))
        assert abs(norm - 1.0) < 1e-5


def test_pca_modes_registered() -> None:
    import fashion_compare.models.pca  # noqa: F401
    from fashion_compare.models.registry import get_mode

    for name, dim in [("pca64", 64), ("pca128", 128), ("pca256", 256)]:
        mode = get_mode(name)
        assert mode.dim == dim
        assert mode.requires_fit is True
        assert mode.fit_fn is not None
        assert mode.load_fn is not None
