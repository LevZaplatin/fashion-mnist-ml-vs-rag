# RAG Embedding Modes Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand RAG from 2 embedding modes to 11 with a unified registry, and assess all modes across 6 top_k values (66 combinations).

**Architecture:** A mode registry (`EmbeddingMode` dataclass + `REGISTRY` dict) replaces all hardcoded mode branching. Each mode self-registers on import. Existing modules (`qdrant_client`, `index`, `retrieve`, `api`, `compare`) become generic consumers of the registry. New embedding modules (blur, HOG, PCA, autoencoder) each define their embed/fit/load functions and register themselves.

**Tech Stack:** Python 3.12, PyTorch, scikit-image (HOG), scikit-learn (PCA), joblib, Qdrant, FastAPI, pytest

---

### Task 1: Mode Registry

**Files:**
- Create: `src/fashion_compare/models/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fashion_compare.models.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fashion_compare/models/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class ModeNotFoundError(KeyError):
    pass


@dataclass
class EmbeddingMode:
    name: str
    dim: int
    embed_fn: Callable[..., np.ndarray]
    requires_fit: bool
    fit_fn: Callable[..., None] | None = None
    load_fn: Callable[..., None] | None = None


_REGISTRY: dict[str, EmbeddingMode] = {}


def register(mode: EmbeddingMode) -> None:
    _REGISTRY[mode.name] = mode


def get_mode(name: str) -> EmbeddingMode:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ModeNotFoundError(
            f"Unknown embedding mode: {name!r}. Available: {sorted(_REGISTRY)}"
        )


def all_mode_names() -> list[str]:
    return sorted(_REGISTRY)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/fashion_compare/models/registry.py tests/test_registry.py
git commit -m "feat: add embedding mode registry"
```

---

### Task 2: Register existing raw784 mode

**Files:**
- Modify: `src/fashion_compare/models/embeddings.py`
- Test: `tests/test_registry.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py::test_raw784_is_registered -v`
Expected: FAIL — `ModeNotFoundError`

- [ ] **Step 3: Add registration to embeddings.py**

Add at the bottom of `src/fashion_compare/models/embeddings.py`:

```python
from fashion_compare.models.registry import EmbeddingMode, register

register(EmbeddingMode(
    name="raw784",
    dim=784,
    embed_fn=embed_raw784,
    requires_fit=False,
))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/fashion_compare/models/embeddings.py tests/test_registry.py
git commit -m "feat: register raw784 in mode registry"
```

---

### Task 3: Register existing cnn_embedding mode

**Files:**
- Modify: `src/fashion_compare/models/embeddings.py`
- Test: `tests/test_registry.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_cnn_embedding_is_registered() -> None:
    import fashion_compare.models.embeddings  # noqa: F401

    mode = get_mode("cnn_embedding")
    assert mode.dim == 128
    assert mode.requires_fit is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py::test_cnn_embedding_is_registered -v`
Expected: FAIL — `ModeNotFoundError`

- [ ] **Step 3: Add registration**

Add to the bottom of `src/fashion_compare/models/embeddings.py`, after the raw784 registration:

```python
register(EmbeddingMode(
    name="cnn_embedding",
    dim=FashionCNN.embedding_dim,
    embed_fn=embed_cnn,
    requires_fit=False,
))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/fashion_compare/models/embeddings.py tests/test_registry.py
git commit -m "feat: register cnn_embedding in mode registry"
```

---

### Task 4: Add blur784 mode

**Files:**
- Create: `src/fashion_compare/models/blur.py`
- Modify: `pyproject.toml` (add scipy)
- Test: `tests/test_blur_embedding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_blur_embedding.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_blur_embedding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fashion_compare.models.blur'`

- [ ] **Step 3: Add scipy dependency**

In `pyproject.toml`, add `"scipy>=1.13"` to the `dependencies` list.

- [ ] **Step 4: Write implementation**

```python
# src/fashion_compare/models/blur.py
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from fashion_compare.models.registry import EmbeddingMode, register
from fashion_compare.preprocessing import image_to_unit_array, l2_normalize

BLUR_SIGMA = 1.0


def embed_blur784(image: np.ndarray | object) -> np.ndarray:
    array = image_to_unit_array(image)
    blurred = gaussian_filter(array, sigma=BLUR_SIGMA).astype(np.float32)
    return l2_normalize(blurred.reshape(-1))


register(EmbeddingMode(
    name="blur784",
    dim=784,
    embed_fn=embed_blur784,
    requires_fit=False,
))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_blur_embedding.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/fashion_compare/models/blur.py tests/test_blur_embedding.py pyproject.toml
git commit -m "feat: add blur784 embedding mode"
```

---

### Task 5: Add HOG and HOG+blur modes

**Files:**
- Create: `src/fashion_compare/models/hog.py`
- Modify: `pyproject.toml` (add scikit-image)
- Test: `tests/test_hog_embedding.py`

- [ ] **Step 1: Add scikit-image dependency**

In `pyproject.toml`, add `"scikit-image>=0.23"` to the `dependencies` list.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_hog_embedding.py
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_hog_embedding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fashion_compare.models.hog'`

- [ ] **Step 4: Write implementation**

```python
# src/fashion_compare/models/hog.py
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import hog as skimage_hog

from fashion_compare.models.registry import EmbeddingMode, register
from fashion_compare.preprocessing import image_to_unit_array, l2_normalize

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (7, 7)
HOG_CELLS_PER_BLOCK = (1, 1)
BLUR_SIGMA = 1.0


def _hog_vector(array_28x28: np.ndarray) -> np.ndarray:
    descriptor = skimage_hog(
        array_28x28,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        feature_vector=True,
    )
    return descriptor.astype(np.float32)


def embed_hog(image: np.ndarray | object) -> np.ndarray:
    array = image_to_unit_array(image)
    return l2_normalize(_hog_vector(array))


def embed_hog_blur(image: np.ndarray | object) -> np.ndarray:
    array = image_to_unit_array(image)
    blurred = gaussian_filter(array, sigma=BLUR_SIGMA).astype(np.float32)
    return l2_normalize(_hog_vector(blurred))


_hog_dim = len(_hog_vector(np.zeros((28, 28), dtype=np.float32)))

register(EmbeddingMode(name="hog", dim=_hog_dim, embed_fn=embed_hog, requires_fit=False))
register(EmbeddingMode(name="hog_blur", dim=_hog_dim, embed_fn=embed_hog_blur, requires_fit=False))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hog_embedding.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/fashion_compare/models/hog.py tests/test_hog_embedding.py pyproject.toml
git commit -m "feat: add hog and hog_blur embedding modes"
```

---

### Task 6: Add PCA modes (pca64, pca128, pca256)

**Files:**
- Create: `src/fashion_compare/models/pca.py`
- Modify: `pyproject.toml` (add joblib)
- Test: `tests/test_pca_embedding.py`

- [ ] **Step 1: Add joblib dependency**

In `pyproject.toml`, add `"joblib>=1.4"` to the `dependencies` list.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_pca_embedding.py
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_pca_embedding.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fashion_compare.models.pca'`

- [ ] **Step 4: Write implementation**

```python
# src/fashion_compare/models/pca.py
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_pca_embedding.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/fashion_compare/models/pca.py tests/test_pca_embedding.py pyproject.toml
git commit -m "feat: add pca64, pca128, pca256 embedding modes"
```

---

### Task 7: Add autoencoder modes (autoencoder64, autoencoder128, autoencoder256)

**Files:**
- Create: `src/fashion_compare/models/autoencoder.py`
- Test: `tests/test_autoencoder_embedding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_autoencoder_embedding.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_autoencoder_embedding.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/fashion_compare/models/autoencoder.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from fashion_compare.models.registry import EmbeddingMode, register
from fashion_compare.preprocessing import image_to_unit_array, l2_normalize


class FashionAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim: int = 128) -> None:
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, bottleneck_dim),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(bottleneck_dim, 64 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(-1, 64, 7, 7)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def fit_autoencoder(
    images: np.ndarray,
    bottleneck_dim: int,
    artifact_dir: Path,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    validation_fraction: float = 0.1,
    patience: int = 3,
) -> FashionAutoencoder:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flat = images.reshape(len(images), 1, 28, 28).astype(np.float32)
    if flat.max() > 1.0:
        flat = flat / 255.0
    tensor = torch.from_numpy(flat)
    val_size = int(len(tensor) * validation_fraction)
    train_tensor = tensor[val_size:]
    val_tensor = tensor[:val_size]
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

    model = FashionAutoencoder(bottleneck_dim=bottleneck_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for (batch,) in val_loader:
                batch = batch.to(device)
                val_loss += float(criterion(model(batch), batch).item()) * len(batch)
        val_loss /= max(1, len(val_tensor))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale = 0
            _save_autoencoder(model, bottleneck_dim, artifact_dir, best_val_loss)
        else:
            stale += 1
            if stale >= patience:
                break

    return model


def _save_autoencoder(
    model: FashionAutoencoder, bottleneck_dim: int, artifact_dir: Path, val_loss: float
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "autoencoder.pt")
    metadata = {"bottleneck_dim": bottleneck_dim, "val_loss": val_loss}
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_autoencoder(artifact_dir: Path, device: torch.device | str = "cpu") -> FashionAutoencoder:
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    model = FashionAutoencoder(bottleneck_dim=metadata["bottleneck_dim"])
    model.load_state_dict(torch.load(artifact_dir / "autoencoder.pt", map_location=device))
    model.eval()
    return model


@torch.inference_mode()
def embed_autoencoder(
    image: Any, model: FashionAutoencoder, device: torch.device | str = "cpu"
) -> np.ndarray:
    array = image_to_unit_array(image)
    tensor = torch.from_numpy(array).float().view(1, 1, 28, 28).to(device)
    embedding = model.encode(tensor).detach().cpu().numpy()[0]
    return l2_normalize(embedding.astype(np.float32))


def _make_fit_fn(bottleneck_dim: int):
    def fit_fn(images: np.ndarray, artifact_dir: Path, **kwargs: Any) -> None:
        fit_autoencoder(images, bottleneck_dim=bottleneck_dim, artifact_dir=artifact_dir, **kwargs)
    return fit_fn


def _make_load_fn():
    def load_fn(artifact_dir: Path) -> FashionAutoencoder:
        return load_autoencoder(artifact_dir)
    return load_fn


def _make_embed_fn(bottleneck_dim: int):
    _cached: dict[str, FashionAutoencoder] = {}

    def embed_fn(image: Any, artifact_dir: Path | None = None) -> np.ndarray:
        key = str(artifact_dir)
        if key not in _cached:
            if artifact_dir is None:
                raise ValueError(
                    f"autoencoder{bottleneck_dim} requires artifact_dir for first call"
                )
            _cached[key] = load_autoencoder(artifact_dir)
        return embed_autoencoder(image, model=_cached[key])
    return embed_fn


for _n in (64, 128, 256):
    register(EmbeddingMode(
        name=f"autoencoder{_n}",
        dim=_n,
        embed_fn=_make_embed_fn(_n),
        requires_fit=True,
        fit_fn=_make_fit_fn(_n),
        load_fn=_make_load_fn(),
    ))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_autoencoder_embedding.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/fashion_compare/models/autoencoder.py tests/test_autoencoder_embedding.py
git commit -m "feat: add autoencoder64, autoencoder128, autoencoder256 embedding modes"
```

---

### Task 8: Auto-populate registry on import via models/__init__.py

**Files:**
- Modify: `src/fashion_compare/models/__init__.py`
- Test: `tests/test_registry.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_all_11_modes_registered_after_models_import() -> None:
    import fashion_compare.models  # noqa: F401
    from fashion_compare.models.registry import all_mode_names

    expected = {
        "raw784", "blur784", "hog", "hog_blur",
        "pca64", "pca128", "pca256",
        "autoencoder64", "autoencoder128", "autoencoder256",
        "cnn_embedding",
    }
    registered = set(all_mode_names())
    assert expected <= registered, f"Missing: {expected - registered}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py::test_all_11_modes_registered_after_models_import -v`
Expected: FAIL — missing modes

- [ ] **Step 3: Update models/__init__.py**

Replace the content of `src/fashion_compare/models/__init__.py`:

```python
# src/fashion_compare/models/__init__.py
from fashion_compare.models.cnn import FashionCNN

import fashion_compare.models.embeddings as _embeddings  # noqa: F401 — registers raw784, cnn_embedding
import fashion_compare.models.blur as _blur  # noqa: F401 — registers blur784
import fashion_compare.models.hog as _hog  # noqa: F401 — registers hog, hog_blur
import fashion_compare.models.pca as _pca  # noqa: F401 — registers pca64, pca128, pca256
import fashion_compare.models.autoencoder as _autoencoder  # noqa: F401 — registers autoencoder64/128/256

__all__ = ["FashionCNN"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add src/fashion_compare/models/__init__.py tests/test_registry.py
git commit -m "feat: auto-populate mode registry on models import"
```

---

### Task 9: Refactor qdrant_client.py to use registry

**Files:**
- Modify: `src/fashion_compare/rag/qdrant_client.py`
- Modify: `src/fashion_compare/rag/api.py`
- Create: `tests/test_qdrant_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qdrant_registry.py
import fashion_compare.models  # noqa: F401 — populate registry

from fashion_compare.rag.qdrant_client import collection_name, vector_size


def test_collection_name_works_for_all_modes() -> None:
    for mode in ["raw784", "blur784", "hog", "hog_blur", "pca64", "pca128", "pca256",
                 "autoencoder64", "autoencoder128", "autoencoder256", "cnn_embedding"]:
        name = collection_name(mode)
        assert name == f"fashion_mnist_{mode}"


def test_vector_size_matches_registry() -> None:
    from fashion_compare.models.registry import get_mode
    for mode_name in ["raw784", "blur784", "hog", "pca128", "autoencoder64", "cnn_embedding"]:
        assert vector_size(mode_name) == get_mode(mode_name).dim
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_qdrant_registry.py -v`
Expected: FAIL — `collection_name("blur784")` raises `ValueError`

- [ ] **Step 3: Refactor qdrant_client.py**

Replace the entire content of `src/fashion_compare/rag/qdrant_client.py`:

```python
# src/fashion_compare/rag/qdrant_client.py
import fashion_compare.models  # noqa: F401 — populate registry

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from fashion_compare.config import Settings, get_settings
from fashion_compare.models.registry import get_mode


def collection_name(mode: str) -> str:
    get_mode(mode)  # validates mode exists
    return f"fashion_mnist_{mode}"


def vector_size(mode: str) -> int:
    return get_mode(mode).dim


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    settings = settings or get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def recreate_collection(client: QdrantClient, mode: str) -> None:
    client.recreate_collection(
        collection_name=collection_name(mode),
        vectors_config=VectorParams(size=vector_size(mode), distance=Distance.COSINE),
    )
```

- [ ] **Step 4: Fix rag/api.py import**

In `src/fashion_compare/rag/api.py`, change line 8:

Old:
```python
from fashion_compare.rag.qdrant_client import COLLECTIONS
```

New:
```python
from fashion_compare.models.registry import all_mode_names
```

And in the `metadata` endpoint (line 28), change:

Old:
```python
return {"system": "rag", "labels": labels_metadata(), "collections": COLLECTIONS}
```

New:
```python
return {"system": "rag", "labels": labels_metadata(), "modes": all_mode_names()}
```

- [ ] **Step 5: Fix test_cnn_forward.py imports**

In `tests/test_cnn_forward.py`, the `test_embedding_dimensions_are_stable` test imports `raw_embedding_dim` and `cnn_embedding_dim` from `fashion_compare.models.embeddings`. These functions still exist, so no change needed. Verify by running:

Run: `pytest tests/test_cnn_forward.py -v`
Expected: 2 passed

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/fashion_compare/rag/qdrant_client.py src/fashion_compare/rag/api.py tests/test_qdrant_registry.py
git commit -m "refactor: qdrant_client and rag api use mode registry"
```

---

### Task 10: Add fit CLI entry point and update config

**Files:**
- Create: `src/fashion_compare/rag/fit.py`
- Modify: `src/fashion_compare/config.py`
- Test: `tests/test_fit_cli.py`

- [ ] **Step 1: Update config.py**

Add these fields to the `Settings` class in `src/fashion_compare/config.py` after the `num_workers` field:

```python
    autoencoder_epochs: int = Field(default=20, alias="AUTOENCODER_EPOCHS")
    top_k_candidates: list[int] = Field(default=[3, 5, 7, 11, 15, 21], alias="TOP_K_CANDIDATES")
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_fit_cli.py
import importlib


def test_fit_module_is_importable() -> None:
    mod = importlib.import_module("fashion_compare.rag.fit")
    assert hasattr(mod, "main")


def test_settings_has_autoencoder_epochs() -> None:
    from fashion_compare.config import get_settings
    settings = get_settings()
    assert settings.autoencoder_epochs == 20


def test_settings_has_top_k_candidates() -> None:
    from fashion_compare.config import get_settings
    settings = get_settings()
    assert settings.top_k_candidates == [3, 5, 7, 11, 15, 21]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_fit_cli.py -v`
Expected: FAIL — `ModuleNotFoundError` for fit module

- [ ] **Step 4: Write implementation**

```python
# src/fashion_compare/rag/fit.py
from __future__ import annotations

import argparse

import numpy as np
from torch.utils.data import Dataset

import fashion_compare.models  # noqa: F401 — populate registry
from fashion_compare.config import get_settings
from fashion_compare.data import load_datasets
from fashion_compare.models.registry import all_mode_names, get_mode
from fashion_compare.preprocessing import image_to_unit_array
from fashion_compare.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _dataset_to_images(dataset: Dataset) -> np.ndarray:
    images = []
    for i in range(len(dataset)):
        image, _ = dataset[i]
        images.append(image_to_unit_array(image))
    return np.stack(images)


def fit_mode(mode_name: str) -> None:
    mode = get_mode(mode_name)
    if not mode.requires_fit:
        logger.info("mode %s does not require fitting, skipping", mode_name)
        return
    if mode.fit_fn is None:
        raise ValueError(f"mode {mode_name} requires fit but has no fit_fn")

    settings = get_settings()
    artifact_dir = settings.rag_dir / mode_name
    logger.info("fitting mode=%s, saving to %s", mode_name, artifact_dir)

    train, _, _ = load_datasets(settings, normalized=False)
    images = _dataset_to_images(train)
    logger.info("loaded %s training images", len(images))

    kwargs = {}
    if "autoencoder" in mode_name:
        kwargs["epochs"] = settings.autoencoder_epochs
    mode.fit_fn(images, artifact_dir=artifact_dir, **kwargs)
    logger.info("finished fitting mode=%s", mode_name)


def main() -> None:
    configure_logging()
    fit_modes = [m for m in all_mode_names() if get_mode(m).requires_fit]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=fit_modes, default=None,
                        help="Fit a single mode. Omit to fit all modes that require it.")
    args = parser.parse_args()

    if args.mode:
        fit_mode(args.mode)
    else:
        for mode_name in fit_modes:
            fit_mode(mode_name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_fit_cli.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/fashion_compare/rag/fit.py src/fashion_compare/config.py tests/test_fit_cli.py
git commit -m "feat: add fit CLI for PCA and autoencoder modes"
```

---

### Task 11: Refactor index.py to use registry

**Files:**
- Modify: `src/fashion_compare/rag/index.py`
- Create: `tests/test_index_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index_registry.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_index_registry.py -v`
Expected: FAIL — `ValueError: Unsupported embedding mode: blur784`

- [ ] **Step 3: Refactor index.py**

Replace the content of `src/fashion_compare/rag/index.py`:

```python
# src/fashion_compare/rag/index.py
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
from qdrant_client.http.models import PointStruct
from torch.utils.data import DataLoader

import fashion_compare.models  # noqa: F401 — populate registry
from fashion_compare.classifier.predict import load_classifier
from fashion_compare.config import get_settings
from fashion_compare.data import load_datasets
from fashion_compare.labels import get_label_name
from fashion_compare.models.embeddings import embed_cnn
from fashion_compare.models.registry import all_mode_names, get_mode
from fashion_compare.rag.qdrant_client import collection_name, get_qdrant_client, recreate_collection
from fashion_compare.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def image_tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    return image.squeeze(0).detach().cpu().numpy().astype(np.float32)


def build_vector(image: Any, mode: str, **kwargs: Any) -> np.ndarray:
    if mode == "cnn_embedding":
        return embed_cnn(image, model=kwargs["model"], device=kwargs.get("device", "cpu"))
    mode_def = get_mode(mode)
    embed_kwargs: dict[str, Any] = {}
    if mode_def.requires_fit and "artifact_dir" in kwargs:
        embed_kwargs["artifact_dir"] = kwargs["artifact_dir"]
    return mode_def.embed_fn(image, **embed_kwargs)


def index_dataset(mode: str, batch_size: int | None = None) -> None:
    settings = get_settings()
    client = get_qdrant_client(settings)
    recreate_collection(client, mode)
    train, _, _ = load_datasets(settings, normalized=False)
    loader = DataLoader(
        train, batch_size=batch_size or settings.batch_size, shuffle=False, num_workers=settings.num_workers
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode_def = get_mode(mode)
    extra_kwargs: dict[str, Any] = {}
    if mode == "cnn_embedding":
        model, _ = load_classifier(device)
        extra_kwargs["model"] = model
        extra_kwargs["device"] = device
    if mode_def.requires_fit:
        extra_kwargs["artifact_dir"] = settings.rag_dir / mode

    next_id = 0
    for images, labels in loader:
        points: list[PointStruct] = []
        for image, label in zip(images, labels, strict=True):
            label_id = int(label)
            vector = build_vector(image_tensor_to_numpy(image), mode=mode, **extra_kwargs)
            points.append(
                PointStruct(
                    id=next_id,
                    vector=vector.tolist(),
                    payload={
                        "image_id": next_id,
                        "split": "train",
                        "label_id": label_id,
                        "label_name": get_label_name(label_id),
                    },
                )
            )
            next_id += 1
        client.upsert(collection_name=collection_name(mode), points=points)
        if next_id % max(1, (batch_size or settings.batch_size) * 10) == 0:
            logger.info("indexed %s train vectors into %s", next_id, collection_name(mode))
    logger.info("finished indexing %s vectors into %s", next_id, collection_name(mode))


def main() -> None:
    configure_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=all_mode_names(), default=settings.embedding_mode)
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    args = parser.parse_args()
    index_dataset(mode=args.mode, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_index_registry.py tests/ -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/fashion_compare/rag/index.py tests/test_index_registry.py
git commit -m "refactor: index.py uses mode registry for all embedding modes"
```

---

### Task 12: Refactor retrieve.py to use registry

**Files:**
- Modify: `src/fashion_compare/rag/retrieve.py`
- Create: `tests/test_retrieve_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieve_registry.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieve_registry.py -v`
Expected: FAIL — `ValueError: Unsupported embedding mode: blur784`

- [ ] **Step 3: Refactor retrieve.py**

Replace the content of `src/fashion_compare/rag/retrieve.py`:

```python
# src/fashion_compare/rag/retrieve.py
from __future__ import annotations

from typing import Any

import torch

import fashion_compare.models  # noqa: F401 — populate registry
from fashion_compare.classifier.predict import load_classifier
from fashion_compare.config import get_settings
from fashion_compare.models.embeddings import embed_cnn
from fashion_compare.models.registry import get_mode
from fashion_compare.rag.qdrant_client import collection_name, get_qdrant_client
from fashion_compare.rag.vote import vote_neighbors


def embed_query(image: Any, mode: str) -> list[float]:
    if mode == "cnn_embedding":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_classifier(device)
        return embed_cnn(image, model=model, device=device).tolist()
    mode_def = get_mode(mode)
    kwargs: dict[str, Any] = {}
    if mode_def.requires_fit:
        settings = get_settings()
        kwargs["artifact_dir"] = settings.rag_dir / mode
    return mode_def.embed_fn(image, **kwargs).tolist()


def retrieve_neighbors(image: Any, mode: str = "raw784", top_k: int = 7) -> list[dict[str, Any]]:
    client = get_qdrant_client()
    hits = query_similar_points(client, collection_name(mode), embed_query(image, mode), top_k)
    neighbors: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        neighbors.append(
            {
                "rank": rank,
                "image_id": int(payload.get("image_id", hit.id)),
                "label_id": int(payload["label_id"]),
                "label_name": str(payload["label_name"]),
                "score": float(hit.score),
            }
        )
    return neighbors


def query_similar_points(client: Any, collection: str, vector: list[float], limit: int) -> list[Any]:
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return list(response.points)
    return list(client.search(collection_name=collection, query_vector=vector, limit=limit))


def predict_rag(image: Any, mode: str = "raw784", top_k: int = 7, weighted: bool = True) -> dict[str, Any]:
    neighbors = retrieve_neighbors(image, mode=mode, top_k=top_k)
    vote = vote_neighbors(neighbors, weighted=weighted)
    return vote.to_response(mode, top_k, neighbors)
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/fashion_compare/rag/retrieve.py tests/test_retrieve_registry.py
git commit -m "refactor: retrieve.py uses mode registry for all embedding modes"
```

---

### Task 13: Extend evaluator with top_k sweep

**Files:**
- Modify: `src/fashion_compare/evaluation/compare.py`
- Create: `tests/test_comparison_topk.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_comparison_topk.py
def test_metrics_key_format_includes_top_k() -> None:
    mode = "hog"
    k = 11
    key = f"rag_{mode}_k{k}"
    assert key == "rag_hog_k11"


def test_write_comparison_includes_top_k_column() -> None:
    import json
    import tempfile
    from pathlib import Path

    from fashion_compare.evaluation.compare import write_comparison

    results = {
        "classifier": {
            "system": "classifier",
            "sample_count": 100,
            "accuracy": 0.9,
            "macro_f1": 0.9,
            "top_k": "-",
            "latency_ms": {"average": 1.0, "p50": 1.0, "p95": 2.0},
        },
        "rag_hog_k7": {
            "system": "rag_hog_k7",
            "sample_count": 100,
            "accuracy": 0.85,
            "macro_f1": 0.84,
            "top_k": 7,
            "latency_ms": {"average": 5.0, "p50": 4.0, "p95": 8.0},
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        reports_dir = Path(tmp)
        write_comparison(results, reports_dir)
        md = (reports_dir / "comparison.md").read_text()
        assert "Top K" in md
        data = json.loads((reports_dir / "comparison.json").read_text())
        assert "rag_hog_k7" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_comparison_topk.py -v`
Expected: FAIL — `write_comparison` does not have `Top K` column yet

- [ ] **Step 3: Refactor compare.py**

Replace the entire content of `src/fashion_compare/evaluation/compare.py`:

```python
# src/fashion_compare/evaluation/compare.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

import fashion_compare.models  # noqa: F401 — populate registry
from fashion_compare.config import get_settings
from fashion_compare.evaluation.evaluate_classifier import evaluate_classifier_predictions
from fashion_compare.evaluation.evaluate_rag import evaluate_rag_predictions
from fashion_compare.evaluation.plots import save_confusion_matrix
from fashion_compare.labels import CLASS_NAMES
from fashion_compare.models.registry import all_mode_names
from fashion_compare.rag.qdrant_client import collection_name, get_qdrant_client
from fashion_compare.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def metrics_from_predictions(
    name: str, y_true: list[int], y_pred: list[int], latencies: list[float]
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(CLASS_NAMES.keys()),
        zero_division=0,
    )
    per_class = []
    for idx, label_id in enumerate(CLASS_NAMES):
        per_class.append(
            {
                "label_id": label_id,
                "label_name": CLASS_NAMES[label_id],
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES.keys()))
    return {
        "system": name,
        "sample_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "latency_ms": {
            "average": float(mean(latencies)) if latencies else 0.0,
            "p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        },
    }


def save_metrics(name: str, metrics: dict[str, Any], reports_dir: Path) -> None:
    path = reports_dir / f"{name}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        title=f"{name} confusion matrix",
        path=reports_dir / f"{name}_confusion_matrix.png",
    )


def collection_exists(mode: str) -> bool:
    try:
        client = get_qdrant_client()
        names = {item.name for item in client.get_collections().collections}
        return collection_name(mode) in names
    except Exception:
        return False


def write_comparison(results: dict[str, dict[str, Any]], reports_dir: Path) -> None:
    lines = [
        "# Fashion-MNIST Classifier Comparison",
        "",
        "| System | Top K | Samples | Accuracy | Macro F1 | Avg ms | P50 ms | P95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in results.items():
        latency = metrics["latency_ms"]
        top_k = metrics.get("top_k", "-")
        lines.append(
            f"| {name} | {top_k} | {metrics['sample_count']} | {metrics['accuracy']:.4f} | "
            f"{metrics['macro_f1']:.4f} | {latency['average']:.2f} | {latency['p50']:.2f} | "
            f"{latency['p95']:.2f} |"
        )
    lines.append("")
    lines.append("Generated by `python -m fashion_compare.evaluation.compare`.")
    (reports_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (reports_dir / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    configure_logging()
    settings = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    parser.add_argument("--top-k", type=int, default=None, help="Single top_k value (overrides sweep)")
    parser.add_argument("--embedding-mode", choices=all_mode_names(), default=None)
    parser.add_argument("--tune-top-k", action="store_true", help="Sweep all top_k candidates for each mode")
    args = parser.parse_args()

    results: dict[str, dict[str, Any]] = {}

    # CNN classifier
    if settings.model_path.exists():
        logger.info("evaluating classifier")
        y_true, y_pred, latencies = evaluate_classifier_predictions(limit=args.limit)
        results["classifier"] = metrics_from_predictions("classifier", y_true, y_pred, latencies)
        results["classifier"]["top_k"] = "-"
        save_metrics("classifier", results["classifier"], settings.reports_dir)
    else:
        logger.warning("skipping classifier: checkpoint does not exist at %s", settings.model_path)

    # Determine modes and top_k values
    modes = [args.embedding_mode] if args.embedding_mode else all_mode_names()
    if args.tune_top_k:
        top_k_values = settings.top_k_candidates
    elif args.top_k:
        top_k_values = [args.top_k]
    else:
        top_k_values = [settings.top_k]

    for mode in modes:
        if not collection_exists(mode):
            logger.warning("skipping rag %s: Qdrant collection not available", mode)
            continue
        for k in top_k_values:
            key = f"rag_{mode}_k{k}"
            logger.info("evaluating rag mode=%s top_k=%s", mode, k)
            y_true, y_pred, latencies = evaluate_rag_predictions(mode=mode, top_k=k, limit=args.limit)
            results[key] = metrics_from_predictions(key, y_true, y_pred, latencies)
            results[key]["top_k"] = k
            save_metrics(key, results[key], settings.reports_dir)

    write_comparison(results, settings.reports_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/fashion_compare/evaluation/compare.py tests/test_comparison_topk.py
git commit -m "feat: evaluator supports top_k sweep across all embedding modes"
```

---

### Task 14: Update Makefile and docker-compose.yml

**Files:**
- Modify: `Makefile`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add fitter service to docker-compose.yml**

Add after the `rag-indexer` service block in `docker-compose.yml`:

```yaml
  fitter:
    build: .
    command: python -m fashion_compare.rag.fit
    env_file: .env.example
    environment:
      QDRANT_HOST: qdrant
    volumes:
      - ./data:/app/data
      - ./artifacts:/app/artifacts
      - ./reports:/app/reports
```

- [ ] **Step 2: Replace Makefile**

```makefile
.PHONY: build train index-rag index-rag-cnn index-mode index-all fit-pca fit-autoencoder fit-all api evaluate evaluate-all test clean

build:
	docker compose build

train:
	docker compose run --rm trainer

index-rag:
	EMBEDDING_MODE=raw784 docker compose run --rm rag-indexer

index-rag-cnn:
	EMBEDDING_MODE=cnn_embedding docker compose run --rm rag-indexer

index-mode:
	EMBEDDING_MODE=$(MODE) docker compose run --rm rag-indexer

index-all:
	@for mode in raw784 blur784 hog hog_blur cnn_embedding; do \
		echo "=== Indexing $$mode ==="; \
		EMBEDDING_MODE=$$mode docker compose run --rm rag-indexer; \
	done
	@for mode in pca64 pca128 pca256 autoencoder64 autoencoder128 autoencoder256; do \
		echo "=== Indexing $$mode ==="; \
		EMBEDDING_MODE=$$mode docker compose run --rm rag-indexer; \
	done

fit-pca:
	@for dim in 64 128 256; do \
		echo "=== Fitting pca$$dim ==="; \
		docker compose run --rm fitter python -m fashion_compare.rag.fit --mode pca$$dim; \
	done

fit-autoencoder:
	@for dim in 64 128 256; do \
		echo "=== Fitting autoencoder$$dim ==="; \
		docker compose run --rm fitter python -m fashion_compare.rag.fit --mode autoencoder$$dim; \
	done

fit-all: fit-pca fit-autoencoder

api:
	docker compose up -d classifier-api rag-api

evaluate:
	docker compose run --rm evaluator

evaluate-all:
	docker compose run --rm evaluator python -m fashion_compare.evaluation.compare --tune-top-k

test:
	docker compose run --rm trainer pytest -q

clean:
	docker compose down
	rm -rf artifacts/classifier/* artifacts/rag/* reports/*
```

- [ ] **Step 3: Verify Makefile syntax**

Run: `make -n build`
Expected: prints `docker compose build` without error

- [ ] **Step 4: Commit**

```bash
git add Makefile docker-compose.yml
git commit -m "feat: add fit/index-all/evaluate-all targets to Makefile and docker-compose"
```

---

### Task 15: Final dependency check in pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Verify all dependencies are present**

Ensure the `dependencies` list in `pyproject.toml` contains:

```toml
dependencies = [
  "fastapi>=0.111",
  "uvicorn[standard]>=0.30",
  "python-multipart>=0.0.9",
  "pydantic>=2.7",
  "pydantic-settings>=2.3",
  "torch>=2.3",
  "torchvision>=0.18",
  "qdrant-client>=1.9",
  "scikit-learn>=1.5",
  "matplotlib>=3.9",
  "pillow>=10.3",
  "numpy>=1.26",
  "scikit-image>=0.23",
  "scipy>=1.13",
  "joblib>=1.4",
]
```

- [ ] **Step 2: Install and verify**

Run: `pip install -e ".[dev]"`
Expected: installs successfully

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add scikit-image, scipy, joblib dependencies"
```

---

### Task 16: End-to-end smoke test

**Files:**
- Create: `tests/test_smoke_all_modes.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_smoke_all_modes.py
import numpy as np

import fashion_compare.models  # noqa: F401
from fashion_compare.models.registry import all_mode_names, get_mode


def test_all_preprocessing_modes_produce_correct_vectors() -> None:
    """Verify every mode that does not require fit or a CNN checkpoint."""
    image = np.random.RandomState(0).rand(28, 28).astype(np.float32)
    skip_modes = {
        "cnn_embedding",
        "pca64", "pca128", "pca256",
        "autoencoder64", "autoencoder128", "autoencoder256",
    }

    for name in all_mode_names():
        if name in skip_modes:
            continue
        mode = get_mode(name)
        vector = mode.embed_fn(image)
        assert vector.shape == (mode.dim,), f"{name}: expected dim {mode.dim}, got {vector.shape}"
        assert vector.dtype == np.float32, f"{name}: expected float32, got {vector.dtype}"
        norm = float(np.linalg.norm(vector))
        assert abs(norm - 1.0) < 1e-4, f"{name}: expected unit norm, got {norm}"


def test_mode_count_is_11() -> None:
    names = all_mode_names()
    assert len(names) == 11, f"Expected 11 modes, got {len(names)}: {names}"
```

- [ ] **Step 2: Run the smoke test**

Run: `pytest tests/test_smoke_all_modes.py -v`
Expected: 2 passed

- [ ] **Step 3: Run full test suite one last time**

Run: `pytest tests/ -v`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke_all_modes.py
git commit -m "test: add end-to-end smoke test for all 11 embedding modes"
```
