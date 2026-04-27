import fashion_compare.models  # noqa: F401 — populate registry

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from fashion_compare.config import Settings, get_settings
from fashion_compare.models.registry import get_mode


def collection_name(mode: str) -> str:
    get_mode(mode)  # validates mode exists
    return f"fashion_mnist_{mode}"


def vector_size(mode: str) -> int:
    return get_mode(mode).dim


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    settings = settings or get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def recreate_collection(client: QdrantClient, mode: str) -> None:
    client.recreate_collection(
        collection_name=collection_name(mode),
        vectors_config=VectorParams(size=vector_size(mode), distance=Distance.COSINE),
    )
