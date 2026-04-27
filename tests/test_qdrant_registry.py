import fashion_compare.models  # noqa: F401 — populate registry

from fashion_compare.rag.qdrant_client import collection_name, vector_size


def test_collection_name_works_for_all_modes() -> None:
    for mode in ["raw784", "blur784", "hog", "hog_blur", "pca64", "pca128", "pca256",
                 "autoencoder64", "autoencoder128", "autoencoder256", "cnn_embedding"]:
        name = collection_name(mode)
        assert name == f"fashion_mnist_{mode}"


def test_vector_size_matches_registry() -> None:
    from fashion_compare.models.registry import get_mode
    for mode_name in ["raw784", "blur784", "hog", "pca128", "autoencoder64", "cnn_embedding"]:
        assert vector_size(mode_name) == get_mode(mode_name).dim
