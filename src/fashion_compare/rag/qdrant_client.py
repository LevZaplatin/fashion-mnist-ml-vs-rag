from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from fashion_compare.config import Settings, get_settings
from fashion_compare.models.embeddings import cnn_embedding_dim, raw_embedding_dim

COLLECTIONS = {
    "raw784": "fashion_mnist_raw784",
    "cnn_embedding": "fashion_mnist_cnn_embedding",
}


def collection_name(mode: str) -> str:
    if mode not in COLLECTIONS:
        raise ValueError(f"Unsupported embedding mode: {mode}")
    return COLLECTIONS[mode]


def vector_size(mode: str) -> int:
    if mode == "raw784":
        return raw_embedding_dim()
    if mode == "cnn_embedding":
        return cnn_embedding_dim()
    raise ValueError(f"Unsupported embedding mode: {mode}")


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    settings = settings or get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def recreate_collection(client: QdrantClient, mode: str) -> None:
    client.recreate_collection(
        collection_name=collection_name(mode),
        vectors_config=VectorParams(size=vector_size(mode), distance=Distance.COSINE),
    )

