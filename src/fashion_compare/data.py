from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from fashion_compare.config import Settings
from fashion_compare.preprocessing import FASHION_MNIST_MEAN, FASHION_MNIST_STD


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def fashion_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
        ]
    )


def raw_fashion_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


def load_datasets(settings: Settings, normalized: bool = True) -> tuple[Dataset, Dataset, Dataset]:
    transform = fashion_transform() if normalized else raw_fashion_transform()
    full_train = datasets.FashionMNIST(
        root=str(settings.data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    test = datasets.FashionMNIST(
        root=str(settings.data_dir),
        train=False,
        download=True,
        transform=transform,
    )
    val_size = int(len(full_train) * settings.validation_fraction)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(settings.seed)
    train, val = random_split(full_train, [train_size, val_size], generator=generator)
    return train, val, test


def create_loaders(settings: Settings) -> DataLoaders:
    train, val, test = load_datasets(settings, normalized=True)
    kwargs = {
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return DataLoaders(
        train=DataLoader(train, shuffle=True, **kwargs),
        val=DataLoader(val, shuffle=False, **kwargs),
        test=DataLoader(test, shuffle=False, **kwargs),
    )

