from __future__ import annotations

import argparse

import numpy as np
from torch.utils.data import Dataset

import fashion_compare.models  # noqa: F401 — populate registry
from fashion_compare.config import get_settings
from fashion_compare.data import load_datasets
from fashion_compare.models.registry import all_mode_names, get_mode
from fashion_compare.preprocessing import image_to_unit_array
from fashion_compare.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _dataset_to_images(dataset: Dataset) -> np.ndarray:
    images = []
    for i in range(len(dataset)):
        image, _ = dataset[i]
        images.append(image_to_unit_array(image))
    return np.stack(images)


def fit_mode(mode_name: str) -> None:
    mode = get_mode(mode_name)
    if not mode.requires_fit:
        logger.info("mode %s does not require fitting, skipping", mode_name)
        return
    if mode.fit_fn is None:
        raise ValueError(f"mode {mode_name} requires fit but has no fit_fn")

    settings = get_settings()
    artifact_dir = settings.rag_dir / mode_name
    logger.info("fitting mode=%s, saving to %s", mode_name, artifact_dir)

    train, _, _ = load_datasets(settings, normalized=False)
    images = _dataset_to_images(train)
    logger.info("loaded %s training images", len(images))

    kwargs = {}
    if "autoencoder" in mode_name:
        kwargs["epochs"] = settings.autoencoder_epochs
    mode.fit_fn(images, artifact_dir=artifact_dir, **kwargs)
    logger.info("finished fitting mode=%s", mode_name)


def main() -> None:
    configure_logging()
    fit_modes = [m for m in all_mode_names() if get_mode(m).requires_fit]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=fit_modes, default=None,
                        help="Fit a single mode. Omit to fit all modes that require it.")
    args = parser.parse_args()

    if args.mode:
        fit_mode(args.mode)
    else:
        for mode_name in fit_modes:
            fit_mode(mode_name)


if __name__ == "__main__":
    main()
