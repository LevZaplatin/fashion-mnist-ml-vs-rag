from __future__ import annotations

from typing import Any

import torch

import fashion_compare.models  # noqa: F401 — populate registry
from fashion_compare.classifier.predict import load_classifier
from fashion_compare.config import get_settings
from fashion_compare.models.embeddings import embed_cnn
from fashion_compare.models.registry import get_mode
from fashion_compare.rag.qdrant_client import collection_name, get_qdrant_client
from fashion_compare.rag.vote import vote_neighbors


def embed_query(image: Any, mode: str) -> list[float]:
    if mode == "cnn_embedding":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_classifier(device)
        return embed_cnn(image, model=model, device=device).tolist()
    mode_def = get_mode(mode)
    kwargs: dict[str, Any] = {}
    if mode_def.requires_fit:
        settings = get_settings()
        kwargs["artifact_dir"] = settings.rag_dir / mode
    return mode_def.embed_fn(image, **kwargs).tolist()


def retrieve_neighbors(image: Any, mode: str = "raw784", top_k: int = 7) -> list[dict[str, Any]]:
    client = get_qdrant_client()
    hits = query_similar_points(client, collection_name(mode), embed_query(image, mode), top_k)
    neighbors: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        neighbors.append(
            {
                "rank": rank,
                "image_id": int(payload.get("image_id", hit.id)),
                "label_id": int(payload["label_id"]),
                "label_name": str(payload["label_name"]),
                "score": float(hit.score),
            }
        )
    return neighbors


def query_similar_points(client: Any, collection: str, vector: list[float], limit: int) -> list[Any]:
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return list(response.points)
    return list(client.search(collection_name=collection, query_vector=vector, limit=limit))


def predict_rag(image: Any, mode: str = "raw784", top_k: int = 7, weighted: bool = True) -> dict[str, Any]:
    neighbors = retrieve_neighbors(image, mode=mode, top_k=top_k)
    vote = vote_neighbors(neighbors, weighted=weighted)
    return vote.to_response(mode, top_k, neighbors)
