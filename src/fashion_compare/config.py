from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    artifacts_dir: Path = Field(default=Path("artifacts"), alias="ARTIFACTS_DIR")
    reports_dir: Path = Field(default=Path("reports"), alias="REPORTS_DIR")
    qdrant_host: str = Field(default="qdrant", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    epochs: int = Field(default=5, alias="EPOCHS")
    batch_size: int = Field(default=128, alias="BATCH_SIZE")
    learning_rate: float = Field(default=1e-3, alias="LEARNING_RATE")
    top_k: int = Field(default=7, alias="TOP_K")
    embedding_mode: str = Field(default="raw784", alias="EMBEDDING_MODE")
    seed: int = Field(default=42, alias="SEED")
    validation_fraction: float = Field(default=0.1, alias="VALIDATION_FRACTION")
    early_stopping_patience: int = Field(default=3, alias="EARLY_STOPPING_PATIENCE")
    num_workers: int = Field(default=0, alias="NUM_WORKERS")
    autoencoder_epochs: int = Field(default=20, alias="AUTOENCODER_EPOCHS")
    top_k_candidates: list[int] = Field(default=[3, 5, 7, 11, 15, 21], alias="TOP_K_CANDIDATES")
    eval_workers: int = Field(default=4, alias="EVAL_WORKERS")

    @property
    def classifier_dir(self) -> Path:
        return self.artifacts_dir / "classifier"

    @property
    def rag_dir(self) -> Path:
        return self.artifacts_dir / "rag"

    @property
    def model_path(self) -> Path:
        return self.classifier_dir / "model.pt"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.classifier_dir.mkdir(parents=True, exist_ok=True)
    settings.rag_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    return settings

