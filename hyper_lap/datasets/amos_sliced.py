from typing import Literal

from pathlib import Path

import numpy as np

from .base import Dataset
from .metadata import Metadata


class AmosSliced(Dataset):
    _metadata: Metadata

    _dataset: list[dict[str, Path]]

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @property
    def name(self) -> str:
        return "AMOS " + self.metadata.name.split(" ")[1]

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "validation", "test"] = "train",
        # preload: bool = False,
    ):
        super().__init__()

        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split: {split}")

        self._metadata = Metadata.load(root_dir)

        self._split = split

        match split:
            case "train":
                self._dataset = self.metadata.training
            case "validation":
                self._dataset = self.metadata.validation
            case "test":
                self._dataset = self.metadata.test
            case _:
                assert False

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}:\n"

        s += f"\tName: {self._metadata.name}\n"
        s += f"\tDescription: {self._metadata.description}\n"
        s += f"\tTensor Image Size: {self._metadata.tensor_image_size}\n"

        s += "\n"

        s += f"\tNumber of training: {self._metadata.num_training}\n"
        s += f"\tNumber of validation: {self._metadata.num_validation}\n"
        s += f"\tNumber of test: {self._metadata.num_test}\n"

        s += "\n"

        s += "\tModalities:\n"
        for i, modality in self._metadata.modality.items():
            s += f"\t\t{i}: {modality}\n"

        s += "\n"

        s += "\tLabels:\n"
        for i, label in self._metadata.labels.items():
            s += f"\t\t{i}: {label}\n"

        return s

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range")

        entry = self._dataset[index]["image"]

        X = np.load(entry)

        return {key: np.array(value) for key, value in X.items()}
        # return dict(X)
