# RAG Embedding Modes Expansion

Expand the RAG pipeline from 2 embedding modes (raw784, cnn_embedding) to 11 modes covering preprocessing-only, unsupervised, and CNN-based strategies. Evaluate all modes across 6 top_k values (66 total combinations) to find the best configuration for each.

## Motivation

`raw784` achieves 85.3% accuracy by comparing images in raw pixel space, which poorly captures semantic similarity. The goal is to add alternative embedding strategies that improve retrieval quality without relying on the trained CNN classifier, while preserving `raw784` as a baseline.

## Embedding Modes

### Group 1: Preprocessing-only (no training required)

| Mode | Description | Dim |
|---|---|---|
| `raw784` | Existing baseline. Pixels to [0,1], flatten, L2-norm. | 784 |
| `blur784` | Gaussian blur (sigma=1.0) then same as raw784. | 784 |
| `hog` | HOG features via scikit-image (9 orientations, 7x7 cells on 28x28). L2-norm. | ~324 |
| `hog_blur` | Gaussian blur then HOG. | ~324 |

### Group 2: Unsupervised (requires fit on training data, no labels used)

| Mode | Description | Dim |
|---|---|---|
| `pca64` | PCA fit on train pixels, transform, L2-norm. | 64 |
| `pca128` | Same, 128 components. | 128 |
| `pca256` | Same, 256 components. | 256 |
| `autoencoder64` | Convolutional autoencoder trained on train images, bottleneck embedding, L2-norm. | 64 |
| `autoencoder128` | Same, 128-dim bottleneck. | 128 |
| `autoencoder256` | Same, 256-dim bottleneck. | 256 |

### Group 3: CNN-dependent (existing)

| Mode | Description | Dim |
|---|---|---|
| `cnn_embedding` | Existing. Penultimate layer of trained FashionCNN. | 128 |

## Mode Registry

New file `src/fashion_compare/models/registry.py`:

```python
@dataclass
class EmbeddingMode:
    name: str              # "raw784", "hog", "pca128", ...
    dim: int               # vector dimensionality
    embed_fn: Callable     # (image) -> np.ndarray or (image, **ctx) -> np.ndarray
    requires_fit: bool     # whether fit step is needed before indexing
    fit_fn: Callable | None  # (train_dataset, settings) -> saves artifact to disk
    load_fn: Callable | None # (settings) -> loads fitted artifact, makes embed_fn ready

REGISTRY: dict[str, EmbeddingMode] = {}

def register(mode: EmbeddingMode) -> None: ...
def get_mode(name: str) -> EmbeddingMode: ...
def all_modes() -> list[str]: ...
def all_mode_names() -> list[str]: ...
```

Each mode self-registers when its module is imported. A top-level `models/__init__.py` imports all mode modules to populate the registry.

## Fit Artifacts

Modes that require fitting save artifacts under `artifacts/rag/{mode_name}/`:

| Mode | Artifact |
|---|---|
| `pca64` | `artifacts/rag/pca64/pca_model.pkl` (joblib) |
| `pca128` | `artifacts/rag/pca128/pca_model.pkl` |
| `pca256` | `artifacts/rag/pca256/pca_model.pkl` |
| `autoencoder64` | `artifacts/rag/autoencoder64/autoencoder.pt` + `metadata.json` |
| `autoencoder128` | `artifacts/rag/autoencoder128/autoencoder.pt` + `metadata.json` |
| `autoencoder256` | `artifacts/rag/autoencoder256/autoencoder.pt` + `metadata.json` |

Fit step runs once before indexing. The `load_fn` loads the artifact at retrieval time (cached).

## Autoencoder Architecture

Convolutional autoencoder for 28x28 grayscale images:

- **Encoder**: Conv2d(1,32,3,padding=1) -> ReLU -> MaxPool(2) -> Conv2d(32,64,3,padding=1) -> ReLU -> MaxPool(2) -> Flatten -> Linear(64*7*7, bottleneck_dim)
- **Decoder**: Linear(bottleneck_dim, 64*7*7) -> Unflatten(64,7,7) -> ConvTranspose2d(64,32,2,stride=2) -> ReLU -> ConvTranspose2d(32,1,2,stride=2) -> Sigmoid
- **Loss**: MSE reconstruction loss
- **Training**: Adam, lr=1e-3, epochs configurable via `AUTOENCODER_EPOCHS` env var (default 20), early stopping on validation reconstruction loss
- **Embedding**: encoder output (bottleneck), L2-normalized

## Changes to Existing Modules

### `rag/qdrant_client.py`

- `recreate_collection(client, mode)` reads dim from `get_mode(mode).dim` instead of hardcoding.
- `collection_name(mode)` returns `f"fashion_mnist_{mode}"` for all modes.

### `rag/index.py`

- `index_dataset(mode)` becomes generic:
  1. `mode_def = get_mode(mode)`
  2. If `mode_def.requires_fit`: run `mode_def.fit_fn(train_data, settings)`
  3. For each image: `vector = mode_def.embed_fn(image)`
  4. Upsert to Qdrant
- `build_vector()` removed, replaced by `mode_def.embed_fn`.

### `rag/retrieve.py`

- `embed_query()` replaced by `get_mode(mode).embed_fn(image)`.
- For fit-based modes, `load_fn` is called once (cached) before embedding.

### `rag/api.py`

- Mode validation uses `all_mode_names()` from registry.

### `evaluation/compare.py`

- New `--tune-top-k` flag enables top_k sweep.
- When enabled, for each embedding mode: iterate `TOP_K_CANDIDATES = [3, 5, 7, 11, 15, 21]`.
- Each combination produces a separate metrics entry keyed as `rag_{mode}_k{k}`.
- Without `--tune-top-k`, behavior is unchanged (uses `TOP_K` from config).

### `config.py`

- Add `top_k_candidates: list[int]` with default `[3, 5, 7, 11, 15, 21]`.
- Add `autoencoder_epochs: int` with default `20`.

### `pyproject.toml`

- Add `scikit-image>=0.23` to dependencies.
- Add `joblib>=1.4` to dependencies (for PCA serialization).

### `docker-compose.yml`

- Add `fitter` service: `python -m fashion_compare.rag.fit --mode {mode}` for PCA/autoencoder fitting.
- Update `rag-indexer` to accept any mode via `EMBEDDING_MODE`.

### `Makefile`

- `make fit-pca`: fit all PCA modes.
- `make fit-autoencoder`: fit all autoencoder modes.
- `make fit-all`: fit PCA + autoencoder.
- `make index-all`: index all 11 modes.
- `make evaluate-all`: evaluate all 66 combinations.

## New Files

| File | Purpose |
|---|---|
| `src/fashion_compare/models/registry.py` | Mode registry, `EmbeddingMode` dataclass, register/get/list functions |
| `src/fashion_compare/models/hog.py` | HOG and HOG+blur embedding functions, registration |
| `src/fashion_compare/models/blur.py` | Gaussian blur embedding function, registration |
| `src/fashion_compare/models/pca.py` | PCA fit/load/embed functions, registration for pca64/pca128/pca256 |
| `src/fashion_compare/models/autoencoder.py` | Autoencoder architecture, train/load/embed, registration for autoencoder64/128/256 |
| `src/fashion_compare/rag/fit.py` | CLI entry point for fitting unsupervised models |

## Report Format

`reports/comparison.json` expands to include all evaluated combinations:

```json
{
  "classifier": { ... },
  "rag_raw784_k7": { "best_top_k": 7, ... },
  "rag_hog_k3": { ... },
  "rag_hog_k5": { ... },
  "rag_hog_k11": { ... },
  "rag_pca128_k7": { ... },
  ...
}
```

`reports/comparison.md` table adds a `top_k` column.

Each entry contains: system, sample_count, top_k, accuracy, macro_precision, macro_recall, macro_f1, per_class metrics, confusion_matrix, latency_ms.

## Out of Scope

- Changes to the CNN classifier or its training pipeline.
- LLM-based classification.
- Hyperparameter tuning beyond top_k (e.g., HOG cell size, blur sigma, autoencoder architecture search).
- Web UI for results comparison.
