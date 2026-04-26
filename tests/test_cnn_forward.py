import torch

from fashion_compare.models.cnn import FashionCNN
from fashion_compare.models.embeddings import cnn_embedding_dim, raw_embedding_dim


def test_cnn_forward_returns_logits_for_10_classes() -> None:
    model = FashionCNN()

    logits = model(torch.zeros(4, 1, 28, 28))

    assert logits.shape == (4, 10)


def test_embedding_dimensions_are_stable() -> None:
    assert raw_embedding_dim() == 784
    assert cnn_embedding_dim(FashionCNN()) == 128

