import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self


@dataclass
class Metadata:
    name: str
    description: str

    tensor_image_size: Literal["3D"] | Literal["4D"]
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

    @staticmethod
    def load(root_dir: str) -> Self:
        base_folder = Path(root_dir)

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

        assert d["tensor_image_size"] in ("3D", "4D")

        d["training"] = [
            dict(image=(base_folder / X["image"]), label=(base_folder / X["label"]))
            for X in d["training"]
        ]
        if "validation" in d:
            d["validation"] = [
                dict(image=(base_folder / X["image"]), label=(base_folder / X["label"]))
                for X in d["validation"]
            ]
        else:
            d["num_validation"] = 0
            d["validation"] = []

        match d["test"][0]:
            case str():
                d["test"] = [dict(image=(base_folder / path)) for path in d["test"]]
            case dict():
                d["test"] = [dict(image=(base_folder / X["image"])) for X in d["test"]]
            case _:
                assert False, f"Unexpected type {type(d['test'][0])} for test entries"

        if "relase" in d:
            # typechecking stuff is totally overrated
            replace_key("relase", "release")

        return Metadata(**d)
