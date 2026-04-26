from typing import Any

import numpy as np
import torch

from fashion_compare.models.cnn import FashionCNN
from fashion_compare.preprocessing import image_to_tensor, l2_normalize, raw784_vector


def raw_embedding_dim() -> int:
    return 784


def cnn_embedding_dim(model: FashionCNN | None = None) -> int:
    return (model or FashionCNN()).embedding_dim


def embed_raw784(image: Any) -> np.ndarray:
    return l2_normalize(raw784_vector(image))


@torch.inference_mode()
def embed_cnn(image: Any, model: FashionCNN, device: torch.device | str = "cpu") -> np.ndarray:
    model.eval()
    tensor = image_to_tensor(image, normalize=True).to(device)
    embedding = model.forward_embedding(tensor).detach().cpu().numpy()[0]
    return l2_normalize(embedding.astype(np.float32))

