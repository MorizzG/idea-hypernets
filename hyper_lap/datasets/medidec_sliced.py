from typing import Literal

from pathlib import Path

import numpy as np

from .base import Dataset
from .metadata import Metadata


class MediDecSliced(Dataset):
    _metadata: Metadata

    _dataset: list[dict[str, Path]]

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Metadata):
        if not isinstance(metadata, Metadata):
            raise ValueError

        self._metadata = metadata

    @property
    def name(self) -> str:
        # "MediDec " +
        return self.metadata.name.split(" ")[0]

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
                # raise ValueError("MediDec sliced dataset has no validation set")
                self._dataset = self.metadata.validation
            case "test":
                self._dataset = self.metadata.test
            case _:
                assert False

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range")

        entry = self._dataset[index]["image"]

        X = np.load(entry)

        return {key: np.array(value) for key, value in X.items()}
        # return dict(X)
