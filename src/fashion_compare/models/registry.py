from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class ModeNotFoundError(KeyError):
    pass


@dataclass
class EmbeddingMode:
    name: str
    dim: int
    embed_fn: Callable[..., np.ndarray]
    requires_fit: bool
    fit_fn: Callable[..., None] | None = None
    load_fn: Callable[..., None] | None = None


_REGISTRY: dict[str, EmbeddingMode] = {}


def register(mode: EmbeddingMode) -> None:
    _REGISTRY[mode.name] = mode


def get_mode(name: str) -> EmbeddingMode:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ModeNotFoundError(
            f"Unknown embedding mode: {name!r}. Available: {sorted(_REGISTRY)}"
        )


def all_mode_names() -> list[str]:
    return sorted(_REGISTRY)
