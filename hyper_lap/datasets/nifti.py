from jaxtyping import DTypeLike
from typing import Literal, Optional

from pathlib import Path

import nibabel as nib
import numpy as np

from .base import Dataset
from .metadata import Metadata


class NiftiDataset(Dataset):
    _metadata: Metadata

    _split: Literal["train", "validation", "test"]

    _dataset: list[dict[str, Path]]

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

        if len(self._dataset) == 0:
            raise ValueError(f"dataset {root_dir} does not appear to have a {split} split")

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: Metadata):
        self._metadata = metadata

    @property
    def name(self) -> str:
        return self.metadata.name

    def __len__(self) -> int:
        return len(self._dataset)

    @staticmethod
    def load_nib(path: Path) -> nib.nifti1.Nifti1Image:
        image = nib.load(path)  # type: ignore

        assert isinstance(image, nib.nifti1.Nifti1Image)

        return image

    @staticmethod
    def load_array(path: Path, dtype: DTypeLike = np.float32):
        nib = NiftiDataset.load_nib(path)

        assert nib.ndim == 3

        return np.asarray(nib.dataobj, dtype=dtype)

    @staticmethod
    def load_array_slice(
        path: Path, slice_idx: int, slice_axis: int = 2, *, dtype: DTypeLike = np.float32
    ):
        if not isinstance(slice_axis, int) or not 0 <= slice_axis <= 2:
            raise ValueError(f"Invalid slice_axis {slice_axis}")

        nib = NiftiDataset.load_nib(path)

        assert nib.ndim == 3

        if slice_axis == 0:
            sliced = nib.slicer[slice_idx : slice_idx + 1, :, :]
        elif slice_axis == 1:
            sliced = nib.slicer[:, slice_idx : slice_idx + 1, :]
        elif slice_axis == 2:
            sliced = nib.slicer[:, :, slice_idx : slice_idx + 1]
        else:
            assert False

        return np.asarray(sliced.dataobj, dtype=dtype).squeeze(slice_axis)

    def get_image_shape(self, idx: int) -> tuple[int, ...]:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        shape = self.load_nib(entry["image"]).shape

        if len(shape) == 3:
            shape = (1,) + shape

        return shape

    def get_image(self, idx: int) -> np.ndarray:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        image = self.load_array(entry["image"])

        # expand image with channel axis if it doesn't exist yet
        if image.ndim == 3:
            image = image[..., None]

        assert image.ndim == 4

        image = np.moveaxis(image, -1, 0)

        return image

    def get_image_slice(self, idx: int, slice_idx: int) -> np.ndarray:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        image = self.load_array_slice(entry["image"], slice_idx)

        # expand image with channel axis if it doesn't exist yet
        if image.ndim == 2:
            image = image[..., None]

        assert image.ndim == 3

        image = np.moveaxis(image, -1, 0)

        return image

    def get_label_shape(self, idx: int) -> tuple[int, ...]:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        if "label" in entry:
            return self.load_nib(entry["label"]).shape

        if self.metadata.labels[0] in entry:
            return (len(self.metadata.labels),) + self.load_nib(
                entry[self.metadata.labels[0]]
            ).shape

        raise RuntimeError(f"Failed to find label in entry {entry}")

    def get_label(self, idx: int) -> Optional[np.ndarray]:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        if "label" in entry:
            return self.load_array(entry["label"], dtype=np.uint8)

        if self.metadata.labels[0] in entry:
            labels = self.metadata.labels

            shape = (len(self.metadata.labels),) + self.load_nib(
                entry[self.metadata.labels[0]]
            ).shape

            label = np.empty(shape, dtype=np.uint8)

            for i, name in labels.items():
                label[i] = self.load_array(entry[name], dtype=np.uint8)

            return label

        return None

    def get_label_slice(self, idx: int, slice_idx: int) -> Optional[np.ndarray]:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        if "label" in entry:
            return self.load_array_slice(entry["label"], slice_idx, dtype=np.uint8)

        if self.metadata.labels[0] in entry:
            labels = self.metadata.labels

            shape = (len(self.metadata.labels),) + self.load_nib(
                entry[self.metadata.labels[0]]
            ).shape

            label = np.empty(shape[:-1], dtype=np.uint8)

            for i, name in labels.items():
                label[i] = self.load_array_slice(entry[name], slice_idx, dtype=np.uint8)

            return label

        return None

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if not 0 <= idx < len(self):
            raise IndexError("index out of range")

        image = self.get_image(idx)
        label = self.get_label(idx)

        if label is not None:
            return {"image": image, "label": label}
        else:
            return {"image": image}

    def get_slice(self, idx: int, slice_idx: int) -> dict[str, np.ndarray]:
        if not 0 <= idx < len(self):
            raise IndexError("index out of range")

        image = self.get_image_slice(idx, slice_idx)
        label = self.get_label_slice(idx, slice_idx)

        if label is not None:
            return {"image": image, "label": label}
        else:
            return {"image": image}
