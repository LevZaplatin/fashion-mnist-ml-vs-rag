CLASS_NAMES: dict[int, str] = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


def get_label_name(label_id: int) -> str:
    try:
        return CLASS_NAMES[int(label_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown Fashion-MNIST label id: {label_id}") from exc


def labels_metadata() -> list[dict[str, int | str]]:
    return [{"label_id": label_id, "label_name": name} for label_id, name in CLASS_NAMES.items()]

