from typing import Any

import numpy as np
import torch
from PIL import Image

FASHION_MNIST_MEAN = 0.2860406
FASHION_MNIST_STD = 0.35302424


def _to_numpy(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("L"), dtype=np.float32)
    else:
        array = np.asarray(image, dtype=np.float32)

    array = np.squeeze(array)
    if array.shape != (28, 28):
        raise ValueError(f"Expected 28x28 grayscale image, got {array.shape}")
    return array.astype(np.float32)


def image_to_unit_array(image: Any) -> np.ndarray:
    array = _to_numpy(image)
    if array.max(initial=0.0) > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def image_to_tensor(image: Any, normalize: bool = True) -> torch.Tensor:
    array = image_to_unit_array(image)
    if normalize:
        array = (array - FASHION_MNIST_MEAN) / FASHION_MNIST_STD
    return torch.from_numpy(array).float().view(1, 1, 28, 28)


def batch_to_tensor(images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    tensor = images.float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.max().item() > 1.0:
        tensor = tensor / 255.0
    if normalize:
        tensor = (tensor - FASHION_MNIST_MEAN) / FASHION_MNIST_STD
    return tensor


def raw784_vector(image: Any) -> np.ndarray:
    return image_to_unit_array(image).reshape(-1).astype(np.float32)


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)

