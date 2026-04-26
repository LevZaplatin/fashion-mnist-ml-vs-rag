import argparse
import itertools
from typing import Any

import numpy as np
import torch
from qdrant_client.http.models import PointStruct
from torch.utils.data import DataLoader

from fashion_compare.classifier.predict import load_classifier
from fashion_compare.config import get_settings
from fashion_compare.data import load_datasets
from fashion_compare.labels import get_label_name
from fashion_compare.models.embeddings import embed_cnn, embed_raw784
from fashion_compare.rag.qdrant_client import collection_name, get_qdrant_client, recreate_collection
from fashion_compare.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def image_tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    return image.squeeze(0).detach().cpu().numpy().astype(np.float32)


def build_vector(image: Any, mode: str, model: Any = None, device: torch.device | str = "cpu") -> np.ndarray:
    if mode == "raw784":
        return embed_raw784(image)
    if mode == "cnn_embedding":
        if model is None:
            raise ValueError("cnn_embedding mode requires a trained classifier checkpoint")
        return embed_cnn(image, model=model, device=device)
    raise ValueError(f"Unsupported embedding mode: {mode}")


def index_dataset(mode: str, batch_size: int | None = None) -> None:
    settings = get_settings()
    client = get_qdrant_client(settings)
    recreate_collection(client, mode)
    train, _, _ = load_datasets(settings, normalized=False)
    loader = DataLoader(train, batch_size=batch_size or settings.batch_size, shuffle=False, num_workers=settings.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if mode == "cnn_embedding":
        model, _ = load_classifier(device)

    next_id = 0
    for images, labels in loader:
        points: list[PointStruct] = []
        for image, label in zip(images, labels, strict=True):
            label_id = int(label)
            vector = build_vector(image_tensor_to_numpy(image), mode=mode, model=model, device=device)
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
    parser.add_argument("--mode", choices=["raw784", "cnn_embedding"], default=settings.embedding_mode)
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    args = parser.parse_args()
    index_dataset(mode=args.mode, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

