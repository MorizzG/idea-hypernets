from typing import Optional
import numpy as np
from torch.utils.data import Dataset

from .nifti import NiftiDataset


class SlicedDataset(Dataset):
    samples_per_volume: int

    dataset: NiftiDataset

    indices: list[tuple[int, int]]

    @staticmethod
    def _make_slice_dist(
        image: np.ndarray, label: np.ndarray | None, target: Optional[int]
    ) -> np.ndarray | None:
        if label is not None:
            if target is None:
                counts = np.count_nonzero(label != 0, axis=(0, 1))
            else:
                counts = np.count_nonzero(label == target, axis=(0, 1))

            if counts.sum() == 0:
                return None

            p = counts / counts.sum()
        else:
            image_shape = image.shape

            n_slices = image_shape[-1]

            p = np.full([n_slices], 1 / n_slices)

        return p

    def __init__(self, dataset: NiftiDataset, target: Optional[int] = None):
        super().__init__()

        self.dataset = dataset

        rng = np.random.default_rng(seed=42)

        self.indices = []

        from tqdm import tqdm

        for i, X in enumerate(tqdm(dataset)):
            image = X["image"]
            label = X.get("label", None)

            p = self._make_slice_dist(image, label, target)

            if p is None:
                continue

            samples_per_volume = min(np.count_nonzero(p) // 4, 2)

            for _ in range(samples_per_volume):
                slice_idx = rng.choice(np.arange(p.shape[0]), p=p)

                self.indices.append((i, slice_idx))

    def __len__(self) -> int:
        # return self.multiplier * len(self.dataset)
        return len(self.indices)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if not 0 <= idx < len(self):
            raise IndexError("index out of range")

        dataset_idx, slice_idx = self.indices[idx]

        return self.dataset.get_slice(dataset_idx, slice_idx)

    # def _make_slice_dist(self, idx: int) -> np.ndarray:
    #     label = self.dataset.get_label(idx)
    #
    #     if label is not None:
    #         counts = np.count_nonzero(label != 0, axis=(0, 1))
    #         p = counts / counts.sum()
    #     else:
    #         label_shape = self.dataset.get_label_shape(idx)
    #
    #         n_slices = label_shape[2]
    #
    #         p = np.full([n_slices], 1 / n_slices)
    #
    #     return p

    # def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
    #     assert 0 <= idx < len(self), f"idx {idx} out of range"
    #
    #     idx = idx % len(self.dataset)
    #
    #     if self.slice_dists[idx] is None:
    #         self._make_slice_dist(idx)
    #
    #     # # X = self.dataset[idx]
    #     # #
    #     # # image = X["image"]
    #     # # label = X.get("label", None)
    #     # #
    #     # # c, h, w, d = image.shape
    #     # #
    #     # # if label is None:
    #     # #     image = self.rng.choice(image, axis=3)
    #     # #
    #     # #     return {"image": image}
    #     # #
    #     # # assert label.shape == (h, w, d)
    #     # #
    #     # # counts = np.count_nonzero(label != 0, axis=(0, 1))
    #     # # counts = counts / counts.sum()
    #     # #
    #     # # idx = self.rng.choice(np.arange(counts.size), p=counts)
    #     #
    #     # image = image[..., idx]
    #     # label = label[..., idx]
    #
    #     p = self.slice_dists[idx]
    #
    #     slice_idx = self.rng.choice(np.arange(p.shape[0]), p=p)
    #
    #     image = self.dataset.get_image_slice(idx, slice_idx)
    #     label = self.dataset.get_label_slice(idx, slice_idx)
    #
    #     if label is not None:
    #         return {"image": image, "label": label}
    #     else:
    #         return {"image": image}
