import numpy as np
import torch

from fashion_compare.preprocessing import image_to_tensor, raw784_vector


def test_image_to_tensor_returns_normalized_batch_shape() -> None:
    image = np.full((28, 28), 128, dtype=np.uint8)

    tensor = image_to_tensor(image, normalize=True)

    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_raw784_vector_range_and_dimension() -> None:
    image = np.arange(28 * 28, dtype=np.float32).reshape(28, 28)

    vector = raw784_vector(image)

    assert vector.shape == (784,)
    assert vector.dtype == np.float32
    assert float(vector.min()) >= 0.0
    assert float(vector.max()) <= 1.0

