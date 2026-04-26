from dataclasses import dataclass
from typing import Any

from fashion_compare.labels import get_label_name


@dataclass(frozen=True)
class VoteResult:
    label_id: int
    label_name: str
    confidence: float
    votes: list[dict[str, float | int | str]]

    def to_response(self, embedding_mode: str, top_k: int, neighbors: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "system": "rag",
            "embedding_mode": embedding_mode,
            "top_k": top_k,
            "label_id": self.label_id,
            "label_name": self.label_name,
            "confidence": self.confidence,
            "neighbors": neighbors,
            "votes": self.votes,
        }


def vote_neighbors(neighbors: list[dict[str, Any]], weighted: bool = True) -> VoteResult:
    if not neighbors:
        raise ValueError("Cannot vote without neighbors")

    totals: dict[int, float] = {}
    names: dict[int, str] = {}
    for item in neighbors:
        label_id = int(item["label_id"])
        score = float(item.get("score", 0.0)) if weighted else 1.0
        totals[label_id] = totals.get(label_id, 0.0) + max(score, 0.0)
        names[label_id] = str(item.get("label_name") or get_label_name(label_id))

    total_score = sum(totals.values()) or float(len(neighbors))
    label_id = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
    votes = [
        {"label_id": lid, "label_name": names[lid], "score": score}
        for lid, score in sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
    ]
    return VoteResult(
        label_id=label_id,
        label_name=names[label_id],
        confidence=float(totals[label_id] / total_score),
        votes=votes,
    )

