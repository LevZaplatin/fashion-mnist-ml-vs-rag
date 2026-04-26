import json
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import Adam

from fashion_compare.config import get_settings
from fashion_compare.data import create_loaders
from fashion_compare.labels import CLASS_NAMES
from fashion_compare.models.cnn import FashionCNN
from fashion_compare.preprocessing import FASHION_MNIST_MEAN, FASHION_MNIST_STD
from fashion_compare.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    validation_fraction: float
    early_stopping_patience: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate_model(model: FashionCNN, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        total_loss += float(criterion(logits, targets).item()) * images.size(0)
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return {
        "loss": total_loss / max(1, len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train() -> None:
    configure_logging()
    settings = get_settings()
    set_seed(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_loaders(settings)
    model = FashionCNN().to(device)
    optimizer = Adam(model.parameters(), lr=settings.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_config = TrainConfig(
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=settings.learning_rate,
        seed=settings.seed,
        validation_fraction=settings.validation_fraction,
        early_stopping_patience=settings.early_stopping_patience,
    )

    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}
    stale_epochs = 0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, targets in loaders.train:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)

        val_metrics = evaluate_model(model, loaders.val, device)
        logger.info(
            "epoch=%s train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch,
            running_loss / max(1, seen),
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics
            stale_epochs = 0
            save_checkpoint(model, train_config, best_metrics)
        else:
            stale_epochs += 1
            if stale_epochs >= settings.early_stopping_patience:
                logger.info("early stopping after %s stale epochs", stale_epochs)
                break


def save_checkpoint(model: FashionCNN, train_config: TrainConfig, metrics: dict[str, float]) -> None:
    settings = get_settings()
    metadata = {
        "labels": CLASS_NAMES,
        "normalization": {"mean": FASHION_MNIST_MEAN, "std": FASHION_MNIST_STD},
        "architecture": "FashionCNN",
        "train_config": asdict(train_config),
        "final_metrics": metrics,
    }
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, settings.model_path)
    (settings.classifier_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train()

