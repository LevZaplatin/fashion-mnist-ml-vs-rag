import base64
import io
from typing import Any

import numpy as np
from PIL import Image


def pil_to_grayscale_28(image: Image.Image) -> Image.Image:
    return image.convert("L").resize((28, 28))


def load_image_bytes(content: bytes) -> Image.Image:
    return pil_to_grayscale_28(Image.open(io.BytesIO(content)))


def base64_to_image(value: str) -> Image.Image:
    if "," in value and value.strip().startswith("data:"):
        value = value.split(",", 1)[1]
    return load_image_bytes(base64.b64decode(value))


def pixel_array_to_numpy(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (28, 28):
        raise ValueError(f"Expected 28x28 pixel array, got {array.shape}")
    return array

