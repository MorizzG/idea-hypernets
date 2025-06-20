from typing import Any

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


# @dataclass(frozen=True)
class Metadata(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    name: str
    description: str

    modality: dict[int, str]
    labels: dict[int, str]

    num_training: int
    num_validation: int
    num_test: int

    training: list[dict[str, Path]] = Field(repr=False)
    validation: list[dict[str, Path]] = Field(repr=False)
    test: list[dict[str, Path]] = Field(repr=False)

    # author: str | None = None
    # contact: str | None = None
    # reference: str | None = None
    # licence: str | None = None
    # release: str | None = None
    # quantitative: str | None = None
    # tensor_image_size: Literal["3D"] | Literal["4D"] | None = None

    __pydantic_extra__: dict[str, Any] = {}
    # extra: dict[str, str] = Field(alias="__pydantic_extra__")

    @property
    def extra(self) -> dict[str, Any]:
        return self.__pydantic_extra__

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
            dataset_json = json.load(f)

        dataset_json["modality"] = {
            int(key): value for key, value in dataset_json["modality"].items()
        }
        dataset_json["labels"] = {int(key): value for key, value in dataset_json["labels"].items()}

        assert (list(dataset_json["modality"].keys())) == list(
            range(len(dataset_json["modality"]))
        ), f"{list(dataset_json['modality'].keys())} != {range(len(dataset_json['modality']))}"
        assert list(dataset_json["labels"].keys()) == list(range(len(dataset_json["labels"]))), (
            f"{list(dataset_json['labels'].keys())} != {range(len(dataset_json['labels']))}"
        )

        def replace_key(old: str, new: str):
            if old not in dataset_json:
                return

            dataset_json[new] = dataset_json[old]
            del dataset_json[old]

        replace_key("tensorImageSize", "tensor_image_size")
        replace_key("numTraining", "num_training")
        replace_key("numValidation", "num_validation")
        replace_key("numTest", "num_test")

        if "tensor_image_size" in dataset_json:
            assert dataset_json["tensor_image_size"] in ("3D", "4D")

        training = []

        for X in dataset_json["training"]:
            if isinstance(X, str):
                image = base_folder / X

                training.append(dict(image=image))
            elif isinstance(X, dict):
                d = {name: base_folder / item for name, item in X.items()}

                for path in d.values():
                    assert path.exists(), f"path {path} does not exist"

                training.append(d)
            else:
                assert False

        dataset_json["training"] = training
        del training

        if "validation" in dataset_json:
            validation = []

            for X in dataset_json["validation"]:
                if isinstance(X, str):
                    image = base_folder / X

                    validation.append(dict(image=image))
                elif isinstance(X, dict):
                    d = {name: base_folder / item for name, item in X.items()}

                    for path in d.values():
                        assert path.exists()

                    validation.append(d)
                else:
                    assert False

            dataset_json["validation"] = validation
            del validation
        else:
            dataset_json["num_validation"] = 0
            dataset_json["validation"] = []

        test = []

        for X in dataset_json["test"]:
            if isinstance(X, str):
                image = base_folder / X

                assert image.exists()

                test.append(dict(image=image))
            elif isinstance(X, dict):
                d = {name: base_folder / item for name, item in X.items()}

                for path in d.values():
                    assert path.exists()

                test.append(d)
            else:
                assert False

        dataset_json["test"] = test
        del test

        if "relase" in dataset_json:
            # spellchecking is totally overrated
            replace_key("relase", "release")

        assert dataset_json["num_training"] == len(dataset_json["training"]), (
            f"{dataset_json['num_training']} != {len(dataset_json['training'])}"
        )
        assert dataset_json["num_validation"] == len(dataset_json["validation"]), (
            f"{dataset_json['num_validation']} != {len(dataset_json['validation'])}"
        )
        assert dataset_json["num_test"] == len(dataset_json["test"]), (
            f"{dataset_json['num_test']} != {len(dataset_json['test'])}"
        )

        if "license" in dataset_json:
            # oops
            replace_key("license", "licence")

        return Metadata(**dataset_json)

    def __str__(self) -> str:
        modality_s = yaml.dump(self.modality)
        labels_s = yaml.dump(self.labels)

        # prepend each line by 2 tabs for indentation
        modality_s = "\n".join("\t\t" + line for line in modality_s.split("\n"))
        labels_s = "\n".join("\t\t" + line for line in labels_s.split("\n"))

        if self.extra:
            extra_s = yaml.dump(self.extra)
            extra_s = "\n".join("\t" + line for line in extra_s.split("\n"))

            extra_s = f"\n{extra_s}"
        else:
            extra_s = ""

        return f"""Metadata:
\tName: {self.name}
\tDescription: {self.description}
{extra_s}
\tModality:
{modality_s}
\tLabels:
{labels_s}

\tNumber of training: {self.num_training}
\tNumber of validation: {self.num_validation}
\tNumber of test: {self.num_test}
        """
