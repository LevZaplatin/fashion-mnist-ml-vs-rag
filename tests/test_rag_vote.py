from fashion_compare.rag.vote import vote_neighbors


def test_weighted_vote_chooses_highest_total_score_label() -> None:
    neighbors = [
        {"label_id": 1, "score": 0.6, "label_name": "Trouser", "image_id": 10},
        {"label_id": 2, "score": 0.5, "label_name": "Pullover", "image_id": 11},
        {"label_id": 2, "score": 0.4, "label_name": "Pullover", "image_id": 12},
    ]

    result = vote_neighbors(neighbors, weighted=True)

    assert result.label_id == 2
    assert result.label_name == "Pullover"
    assert round(result.confidence, 4) == 0.6


def test_unweighted_vote_uses_neighbor_counts() -> None:
    neighbors = [
        {"label_id": 1, "score": 0.9, "label_name": "Trouser", "image_id": 10},
        {"label_id": 2, "score": 0.2, "label_name": "Pullover", "image_id": 11},
        {"label_id": 2, "score": 0.1, "label_name": "Pullover", "image_id": 12},
    ]

    result = vote_neighbors(neighbors, weighted=False)

    assert result.label_id == 2
    assert round(result.confidence, 4) == 0.6667

