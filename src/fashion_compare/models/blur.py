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
