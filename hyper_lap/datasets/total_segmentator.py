from typing import Literal, Optional

from pathlib import Path

import numpy as np

from hyper_lap.datasets.nifti import NiftiDataset


class TotalSegmentator(NiftiDataset):
    min_axes: list[int]

    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "validation", "test"] = "train",
    ):
        super().__init__(root_dir, split)

        self.min_axes = []

        for idx in range(len(self)):
            shape = super().get_image_shape(idx)

            min_axis = 1 + int(np.argmin(shape[1:]))

            self.min_axes.append(min_axis)

    def get_image_shape(self, idx: int) -> tuple[int, ...]:
        shape = super().get_image_shape(idx)

        min_axis = self.min_axes[idx]

        shape = shape[:min_axis] + shape[min_axis + 1 :] + (shape[min_axis],)

        return shape

    def get_image(self, idx: int) -> np.ndarray:
        image = super().get_image(idx)

        min_axis = self.min_axes[idx]

        # move smallest axis last
        image = np.moveaxis(image, min_axis, -1)

        return image

    def get_image_slice(self, idx: int, slice_idx: int) -> np.ndarray:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        slice_axis = self.min_axes[idx] - 1

        image = self.load_array_slice(entry["image"], slice_idx, slice_axis=slice_axis)

        if image.ndim == 2:
            # expand image with channel axis if it doesn't exist yet
            # add it at last position, since we move last to first later
            image = image[..., None]

        assert image.ndim == 3

        image = np.moveaxis(image, -1, 0)

        return image

    def get_label_shape(self, idx: int) -> tuple[int, ...]:
        shape = super().get_label_shape(idx)

        min_axis = self.min_axes[idx]

        shape = shape[:min_axis] + shape[min_axis + 1 :] + (shape[min_axis],)

        return shape

    def get_label(self, idx: int) -> Optional[np.ndarray]:
        label = super().get_label(idx)

        if label is None:
            return label

        min_axis = self.min_axes[idx]

        # move smallest axis last
        label = np.moveaxis(label, min_axis, -1)

        return label

    def get_label_slice(self, idx: int, slice_idx: int) -> Optional[np.ndarray]:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        entry = self._dataset[idx]

        labels = self.metadata.labels

        if labels[0] not in entry:
            return None

        # label nib has no channel axis
        slice_axis = self.min_axes[idx] - 1

        label = np.empty(self.get_label_shape(idx)[:-1], dtype=np.uint8)

        for i, name in labels.items():
            label[i] = self.load_array_slice(
                entry[name], slice_idx, slice_axis=slice_axis, dtype=np.uint8
            )

        return label
