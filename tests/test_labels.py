from fashion_compare.labels import CLASS_NAMES, get_label_name


def test_label_mapping_is_fashion_mnist_order() -> None:
    assert CLASS_NAMES == {
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
    assert get_label_name(9) == "Ankle boot"

