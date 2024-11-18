from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np

from .metadata import Metadata


class NIfTIDataset:
    _metadata: Metadata

    _split: Literal["train", "validation", "test"]

    _dataset: list[dict[str, Path]]

    def __init__(self, root_dir: str, split: Literal["train", "validation", "test"] = "train"):
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

    @property
    def metadata(self) -> Metadata:
        return self._metadata

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

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        image = nib.load(entry["image"]).get_fdata().astype(np.float32)  # type: ignore

        if "label" in entry:
            label = nib.load(entry["label"]).get_fdata().astype(np.uint8)  # type: ignore
        else:
            label = None

        # expand image with channel axis if it doesn't exist yet
        if self.metadata.tensor_image_size == "3D":
            assert image.ndim == 3
            image = image[..., None]

        assert image.ndim == 4

        image = np.moveaxis(image, 3, 0)

        if label is not None:
            return {"image": image, "label": label}
        else:
            return {"image": image}
