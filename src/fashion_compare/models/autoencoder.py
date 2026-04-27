from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from fashion_compare.models.registry import EmbeddingMode, register
from fashion_compare.preprocessing import image_to_unit_array, l2_normalize


class FashionAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim: int = 128) -> None:
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, bottleneck_dim),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(bottleneck_dim, 64 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(-1, 64, 7, 7)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def fit_autoencoder(
    images: np.ndarray,
    bottleneck_dim: int,
    artifact_dir: Path,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    validation_fraction: float = 0.1,
    patience: int = 3,
) -> FashionAutoencoder:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flat = images.reshape(len(images), 1, 28, 28).astype(np.float32)
    if flat.max() > 1.0:
        flat = flat / 255.0
    tensor = torch.from_numpy(flat)
    val_size = int(len(tensor) * validation_fraction)
    train_tensor = tensor[val_size:]
    val_tensor = tensor[:val_size]
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

    model = FashionAutoencoder(bottleneck_dim=bottleneck_dim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()

        # switch to inference mode for validation
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for (batch,) in val_loader:
                batch = batch.to(device)
                val_loss += float(criterion(model(batch), batch).item()) * len(batch)
        val_loss /= max(1, len(val_tensor))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale = 0
            _save_autoencoder(model, bottleneck_dim, artifact_dir, best_val_loss)
        else:
            stale += 1
            if stale >= patience:
                break

    return model


def _save_autoencoder(
    model: FashionAutoencoder, bottleneck_dim: int, artifact_dir: Path, val_loss: float
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "autoencoder.pt")
    metadata = {"bottleneck_dim": bottleneck_dim, "val_loss": val_loss}
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_autoencoder(artifact_dir: Path, device: torch.device | str = "cpu") -> FashionAutoencoder:
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    model = FashionAutoencoder(bottleneck_dim=metadata["bottleneck_dim"])
    model.load_state_dict(torch.load(artifact_dir / "autoencoder.pt", map_location=device))
    # set model to inference mode
    model.eval()
    return model


@torch.inference_mode()
def embed_autoencoder(
    image: Any, model: FashionAutoencoder, device: torch.device | str = "cpu"
) -> np.ndarray:
    array = image_to_unit_array(image)
    tensor = torch.from_numpy(array).float().view(1, 1, 28, 28).to(device)
    embedding = model.encode(tensor).detach().cpu().numpy()[0]
    return l2_normalize(embedding.astype(np.float32))


def _make_fit_fn(bottleneck_dim: int):
    def fit_fn(images: np.ndarray, artifact_dir: Path, **kwargs: Any) -> None:
        fit_autoencoder(images, bottleneck_dim=bottleneck_dim, artifact_dir=artifact_dir, **kwargs)
    return fit_fn


def _make_load_fn():
    def load_fn(artifact_dir: Path) -> FashionAutoencoder:
        return load_autoencoder(artifact_dir)
    return load_fn


def _make_embed_fn(bottleneck_dim: int):
    _cached: dict[str, FashionAutoencoder] = {}

    def embed_fn(image: Any, artifact_dir: Path | None = None) -> np.ndarray:
        key = str(artifact_dir)
        if key not in _cached:
            if artifact_dir is None:
                raise ValueError(
                    f"autoencoder{bottleneck_dim} requires artifact_dir for first call"
                )
            _cached[key] = load_autoencoder(artifact_dir)
        return embed_autoencoder(image, model=_cached[key])
    return embed_fn


for _n in (64, 128, 256):
    register(EmbeddingMode(
        name=f"autoencoder{_n}",
        dim=_n,
        embed_fn=_make_embed_fn(_n),
        requires_fit=True,
        fit_fn=_make_fit_fn(_n),
        load_fn=_make_load_fn(),
    ))
