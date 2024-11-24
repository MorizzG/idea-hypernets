from pathlib import Path
from typing import Literal, Optional

import nibabel as nib
import numpy as np

from .metadata import Metadata


class NiftiDataset:
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

    def get_image_shape(self, idx: int) -> tuple[int, ...]:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        return entry["image"].shape

    def get_image(self, idx: int) -> np.ndarray:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        image = np.asanyarray(nib.load(entry["image"]).dataobj).astype(np.float32)

        # expand image with channel axis if it doesn't exist yet
        if self.metadata.tensor_image_size == "3D":
            assert image.ndim == 3
            image = image[..., None]

        assert image.ndim == 4

        image = np.moveaxis(image, -1, 0)

        return image

    def get_image_slice(self, idx: int, slice_idx: int) -> np.ndarray:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        image = (
            np.asanyarray(nib.load(entry["image"]).slicer[:, :, slice_idx : slice_idx + 1].dataobj)
            .squeeze(2)
            .astype(np.float32)
        )

        # expand image with channel axis if it doesn't exist yet
        if self.metadata.tensor_image_size == "3D":
            assert image.ndim == 2
            image = image[..., None]

        assert image.ndim == 3

        image = np.moveaxis(image, -1, 0)

        return image

    def get_label_shape(self, idx: int) -> tuple[int, ...]:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        assert "label" in entry

        return entry["label"].shape

    def get_label(self, idx: int) -> Optional[np.ndarray]:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        if "label" not in entry:
            return None

        label = np.asanyarray(nib.load(entry["label"]).dataobj).astype(np.uint8)

        return label

    def get_label_slice(self, idx: int, slice_idx: int) -> Optional[np.ndarray]:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        if "label" not in entry:
            return None

        label = (
            np.asanyarray(nib.load(entry["label"]).slicer[:, :, slice_idx : slice_idx + 1].dataobj)
            .squeeze(2)
            .astype(np.float32)
        )

        return label

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        image = self.get_image(idx)
        label = self.get_label(idx)

        if label is not None:
            return {"image": image, "label": label}
        else:
            return {"image": image}

    def get_slice(self, idx: int, slice_idx: int) -> dict[str, np.ndarray]:
        image = self.get_image_slice(idx, slice_idx)
        label = self.get_label_slice(idx, slice_idx)

        if label is not None:
            return {"image": image, "label": label}
        else:
            return {"image": image}
