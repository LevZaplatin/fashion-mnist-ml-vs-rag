from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA

from fashion_compare.models.registry import EmbeddingMode, register
from fashion_compare.preprocessing import image_to_unit_array, l2_normalize


def fit_pca(images: np.ndarray, n_components: int, artifact_dir: Path) -> PCA:
    flat = images.reshape(len(images), -1).astype(np.float32)
    if flat.max() > 1.0:
        flat = flat / 255.0
    pca = PCA(n_components=n_components)
    pca.fit(flat)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, artifact_dir / "pca_model.pkl")
    return pca


def load_pca(artifact_dir: Path) -> PCA:
    return joblib.load(artifact_dir / "pca_model.pkl")


def embed_pca(image: Any, pca: PCA) -> np.ndarray:
    array = image_to_unit_array(image)
    flat = array.reshape(1, -1).astype(np.float32)
    transformed = pca.transform(flat)[0].astype(np.float32)
    return l2_normalize(transformed)


def _make_fit_fn(n_components: int):
    def fit_fn(images: np.ndarray, artifact_dir: Path) -> None:
        fit_pca(images, n_components=n_components, artifact_dir=artifact_dir)
    return fit_fn


def _make_load_fn():
    def load_fn(artifact_dir: Path) -> PCA:
        return load_pca(artifact_dir)
    return load_fn


def _make_embed_fn(n_components: int):
    _cached_pca: dict[str, PCA] = {}

    def embed_fn(image: Any, artifact_dir: Path | None = None) -> np.ndarray:
        key = str(artifact_dir)
        if key not in _cached_pca:
            if artifact_dir is None:
                raise ValueError(f"pca{n_components} requires artifact_dir for first call")
            _cached_pca[key] = load_pca(artifact_dir)
        return embed_pca(image, pca=_cached_pca[key])
    return embed_fn


for _n in (64, 128, 256):
    register(EmbeddingMode(
        name=f"pca{_n}",
        dim=_n,
        embed_fn=_make_embed_fn(_n),
        requires_fit=True,
        fit_fn=_make_fit_fn(_n),
        load_fn=_make_load_fn(),
    ))
