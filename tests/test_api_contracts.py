from fashion_compare.classifier.predict import build_classifier_response
from fashion_compare.rag.vote import vote_neighbors


def test_classifier_response_schema_has_required_fields() -> None:
    response = build_classifier_response([0.7, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0])

    assert response["system"] == "classifier"
    assert {"label_id", "label_name", "confidence", "probabilities"} <= response.keys()
    assert len(response["probabilities"]) == 10


def test_rag_prediction_schema_has_required_fields() -> None:
    neighbors = [
        {"rank": 1, "image_id": 1, "label_id": 8, "label_name": "Bag", "score": 0.8},
        {"rank": 2, "image_id": 2, "label_id": 8, "label_name": "Bag", "score": 0.6},
    ]

    vote = vote_neighbors(neighbors)
    response = vote.to_response("raw784", top_k=2, neighbors=neighbors)

    assert response["system"] == "rag"
    assert response["embedding_mode"] == "raw784"
    assert {"label_id", "label_name", "confidence", "neighbors", "votes"} <= response.keys()
