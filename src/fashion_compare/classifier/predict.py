from typing import Any

import numpy as np
import torch

from fashion_compare.config import get_settings
from fashion_compare.labels import get_label_name, labels_metadata
from fashion_compare.models.cnn import FashionCNN
from fashion_compare.preprocessing import image_to_tensor


def build_classifier_response(probabilities: list[float] | np.ndarray) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float64)
    label_id = int(probs.argmax())
    ordered = [
        {
            "label_id": int(idx),
            "label_name": get_label_name(int(idx)),
            "probability": float(prob),
        }
        for idx, prob in enumerate(probs.tolist())
    ]
    return {
        "system": "classifier",
        "label_id": label_id,
        "label_name": get_label_name(label_id),
        "confidence": float(probs[label_id]),
        "probabilities": ordered,
    }


def load_classifier(device: torch.device | str | None = None) -> tuple[FashionCNN, dict[str, Any]]:
    settings = get_settings()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(settings.model_path, map_location=device)
    model = FashionCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint.get("metadata", {})


@torch.inference_mode()
def predict_image(image: Any, model: FashionCNN | None = None, device: torch.device | str | None = None) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model, _ = load_classifier(device)
    tensor = image_to_tensor(image, normalize=True).to(device)
    probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    return build_classifier_response(probs)


def classifier_metadata(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"system": "classifier", "labels": labels_metadata(), "checkpoint": metadata or {}}

