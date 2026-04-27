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
