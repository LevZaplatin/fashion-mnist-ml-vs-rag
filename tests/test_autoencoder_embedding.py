import tempfile
from pathlib import Path

import numpy as np
import torch

from fashion_compare.models.autoencoder import (
    FashionAutoencoder,
    embed_autoencoder,
    fit_autoencoder,
    load_autoencoder,
)


def test_autoencoder_forward_reconstructs_shape() -> None:
    model = FashionAutoencoder(bottleneck_dim=64)
    x = torch.randn(4, 1, 28, 28)
    reconstructed = model(x)
    assert reconstructed.shape == (4, 1, 28, 28)


def test_autoencoder_encoder_output_shape() -> None:
    model = FashionAutoencoder(bottleneck_dim=128)
    x = torch.randn(2, 1, 28, 28)
    embedding = model.encode(x)
    assert embedding.shape == (2, 128)


def test_fit_and_embed_roundtrip() -> None:
    rng = np.random.RandomState(42)
    images = rng.rand(100, 28, 28).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        fit_autoencoder(images, bottleneck_dim=64, artifact_dir=artifact_dir, epochs=1)
        assert (artifact_dir / "autoencoder.pt").exists()
        assert (artifact_dir / "metadata.json").exists()

        model = load_autoencoder(artifact_dir)
        image = rng.rand(28, 28).astype(np.float32)
        vector = embed_autoencoder(image, model=model)
        assert vector.shape == (64,)
        assert vector.dtype == np.float32
        norm = float(np.linalg.norm(vector))
        assert abs(norm - 1.0) < 1e-5


def test_autoencoder_modes_registered() -> None:
    import fashion_compare.models.autoencoder  # noqa: F401
    from fashion_compare.models.registry import get_mode

    for name, dim in [("autoencoder64", 64), ("autoencoder128", 128), ("autoencoder256", 256)]:
        mode = get_mode(name)
        assert mode.dim == dim
        assert mode.requires_fit is True
        assert mode.fit_fn is not None
        assert mode.load_fn is not None
