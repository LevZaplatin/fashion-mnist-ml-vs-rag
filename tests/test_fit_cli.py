import importlib


def test_fit_module_is_importable() -> None:
    mod = importlib.import_module("fashion_compare.rag.fit")
    assert hasattr(mod, "main")


def test_settings_has_autoencoder_epochs() -> None:
    from fashion_compare.config import get_settings
    settings = get_settings()
    assert settings.autoencoder_epochs == 20


def test_settings_has_top_k_candidates() -> None:
    from fashion_compare.config import get_settings
    settings = get_settings()
    assert settings.top_k_candidates == [3, 5, 7, 11, 15, 21]
