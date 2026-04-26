from types import SimpleNamespace

from fashion_compare.rag.retrieve import query_similar_points


def test_query_similar_points_uses_query_points_client_api() -> None:
    class Client:
        def query_points(self, collection_name, query, limit, with_payload):
            assert collection_name == "collection"
            assert query == [0.1, 0.2]
            assert limit == 2
            assert with_payload is True
            return SimpleNamespace(points=["a", "b"])

    assert query_similar_points(Client(), "collection", [0.1, 0.2], 2) == ["a", "b"]


def test_query_similar_points_supports_legacy_search_client_api() -> None:
    class Client:
        def search(self, collection_name, query_vector, limit):
            assert collection_name == "collection"
            assert query_vector == [0.1, 0.2]
            assert limit == 2
            return ["legacy"]

    assert query_similar_points(Client(), "collection", [0.1, 0.2], 2) == ["legacy"]
