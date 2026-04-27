from fashion_compare.models.cnn import FashionCNN

import fashion_compare.models.embeddings as _embeddings  # noqa: F401 — registers raw784, cnn_embedding
import fashion_compare.models.blur as _blur  # noqa: F401 — registers blur784
import fashion_compare.models.hog as _hog  # noqa: F401 — registers hog, hog_blur
import fashion_compare.models.pca as _pca  # noqa: F401 — registers pca64, pca128, pca256
import fashion_compare.models.autoencoder as _autoencoder  # noqa: F401 — registers autoencoder64/128/256

__all__ = ["FashionCNN"]
