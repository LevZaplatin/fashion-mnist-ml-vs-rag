import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for name in ("httpx", "httpcore"):
        noisy = logging.getLogger(name)
        noisy.setLevel(logging.WARNING)
        noisy.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

