from typing import Literal

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Metadata:
    name: str
    description: str

    modality: dict[str, str]
    labels: dict[str, str]

    num_training: int
    num_validation: int
    num_test: int

    training: list[dict[str, Path]]
    validation: list[dict[str, Path]]
    test: list[dict[str, Path]]

    author: str | None = None
    contact: str | None = None
    reference: str | None = None
    licence: str | None = None
    release: str | None = None
    quantitative: str | None = None
    tensor_image_size: Literal["3D"] | Literal["4D"] | None = None

    @staticmethod
    def load(root_dir: str | Path) -> "Metadata":
        if isinstance(root_dir, str):
            base_folder = Path(root_dir)
        elif isinstance(root_dir, Path):
            base_folder = root_dir
        else:
            raise ValueError(f"Invalid root_dir {root_dir}")

        assert base_folder.exists()

        with (base_folder / "dataset.json").open() as f:
            d = json.load(f)

        def replace_key(old: str, new: str):
            if old not in d:
                return

            d[new] = d[old]
            del d[old]

        replace_key("tensorImageSize", "tensor_image_size")
        replace_key("numTraining", "num_training")
        replace_key("numValidation", "num_validation")
        replace_key("numTest", "num_test")

        if "tensor_image_size" in d:
            assert d["tensor_image_size"] in ("3D", "4D")

        training = []

        for X in d["training"]:
            if isinstance(X, str):
                image = base_folder / X

                training.append(dict(image=image))
            elif isinstance(X, dict):
                image = base_folder / X["image"]
                label = base_folder / X["label"]

                assert image.exists() and label.exists()

                training.append(dict(image=image, label=label))
            else:
                assert False

        d["training"] = training

        if "validation" in d:
            validation = []

            for X in d["validation"]:
                if isinstance(X, str):
                    image = base_folder / X

                    validation.append(dict(image=image))
                elif isinstance(X, dict):
                    image = base_folder / X["image"]
                    label = base_folder / X["label"]

                    assert image.exists() and label.exists()

                    validation.append(dict(image=image, label=label))
                else:
                    assert False

            d["validation"] = validation
        else:
            d["num_validation"] = 0
            d["validation"] = []

        test = []

        assert len(d["test"]) == d["num_test"], f"{len(d['test'])} != {d['num_test']} {root_dir}"

        for X in d["test"]:
            if isinstance(X, str):
                image = base_folder / X
            elif isinstance(X, dict):
                image = base_folder / X["image"]
            else:
                assert False

            assert image.exists()

            test.append(dict(image=image))

        d["test"] = test

        if "relase" in d:
            # spellchecking is totally overrated
            replace_key("relase", "release")

        assert d["num_training"] == len(
            d["training"]
        ), f"{d['num_training']} != {len(d['training'])}"
        assert d["num_validation"] == len(
            d["validation"]
        ), f"{d['num_validation']} != {len(d['validation'])}"
        assert d["num_test"] == len(d["test"]), f"{d['num_test']} != {len(d['test'])}"

        return Metadata(**d)

    def __repr__(self) -> str:
        s = "Metadata:\n"

        s += f"\tName: {self.name}\n"
        s += f"\tDescription: {self.description}\n"
        s += "\n"
        s += f"\tTensor Image Size: {self.tensor_image_size}\n"
        s += f"\tModality: {self.modality}\n"
        s += f"\tLabels: {self.labels}\n"
        s += "\n"
        s += f"\tNumber of training: {self.num_training}\n"
        s += f"\tNumber of validation: {self.num_validation}\n"
        s += f"\tNumber of test: {self.num_test}\n"

        return s
