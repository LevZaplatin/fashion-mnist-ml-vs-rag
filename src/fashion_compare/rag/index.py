from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
from qdrant_client.http.models import PointStruct
from torch.utils.data import DataLoader

import fashion_compare.models  # noqa: F401 — populate registry
from fashion_compare.classifier.predict import load_classifier
from fashion_compare.config import get_settings
from fashion_compare.data import load_datasets
from fashion_compare.labels import get_label_name
from fashion_compare.models.embeddings import embed_cnn
from fashion_compare.models.registry import all_mode_names, get_mode
from fashion_compare.rag.qdrant_client import collection_name, get_qdrant_client, recreate_collection
from fashion_compare.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def image_tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    return image.squeeze(0).detach().cpu().numpy().astype(np.float32)


def build_vector(image: Any, mode: str, **kwargs: Any) -> np.ndarray:
    if mode == "cnn_embedding":
        return embed_cnn(image, model=kwargs["model"], device=kwargs.get("device", "cpu"))
    mode_def = get_mode(mode)
    embed_kwargs: dict[str, Any] = {}
    if mode_def.requires_fit and "artifact_dir" in kwargs:
        embed_kwargs["artifact_dir"] = kwargs["artifact_dir"]
    return mode_def.embed_fn(image, **embed_kwargs)


def index_dataset(mode: str, batch_size: int | None = None) -> None:
    settings = get_settings()
    client = get_qdrant_client(settings)
    recreate_collection(client, mode)
    train, _, _ = load_datasets(settings, normalized=False)
    loader = DataLoader(
        train, batch_size=batch_size or settings.batch_size, shuffle=False, num_workers=settings.num_workers
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode_def = get_mode(mode)
    extra_kwargs: dict[str, Any] = {}
    if mode == "cnn_embedding":
        model, _ = load_classifier(device)
        extra_kwargs["model"] = model
        extra_kwargs["device"] = device
    if mode_def.requires_fit:
        extra_kwargs["artifact_dir"] = settings.rag_dir / mode

    next_id = 0
    for images, labels in loader:
        points: list[PointStruct] = []
        for image, label in zip(images, labels, strict=True):
            label_id = int(label)
            vector = build_vector(image_tensor_to_numpy(image), mode=mode, **extra_kwargs)
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
    parser.add_argument("--mode", choices=all_mode_names(), default=settings.embedding_mode)
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    args = parser.parse_args()
    index_dataset(mode=args.mode, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
