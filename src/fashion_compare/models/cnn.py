import torch
from torch import nn


class FashionCNN(nn.Module):
    embedding_dim = 128

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(0.25)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, self.embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_embedding(x))

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.embedding(self.dropout(features))

