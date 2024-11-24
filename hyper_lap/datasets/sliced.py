import numpy as np

from .nifti import NiftiDataset


class SlicedDataset:
    dataset: NiftiDataset

    slice_dists: list[np.ndarray]

    rng: np.random.Generator = np.random.default_rng()

    def __init__(self, dataset: NiftiDataset):
        super().__init__()

        self.dataset = dataset

        # self.slice_dists = len(self.dataset) * [None]

        self.slice_dists = [self._make_slice_dist(idx) for idx in range(len(self.dataset))]

    def __len__(self) -> int:
        return len(self.dataset)

    def _make_slice_dist(self, idx: int) -> np.ndarray:
        label = self.dataset.get_label(idx)

        if label is not None:
            counts = np.count_nonzero(label != 0, axis=(0, 1))
            p = counts / counts.sum()
        else:
            label_shape = self.dataset.get_label_shape(idx)

            n_slices = label_shape[2]

            p = np.full([n_slices], 1 / n_slices)

        return p

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        assert 0 <= idx < len(self), f"idx {idx} out of range"

        if self.slice_dists[idx] is None:
            self._make_slice_dist(idx)

        # # X = self.dataset[idx]
        # #
        # # image = X["image"]
        # # label = X.get("label", None)
        # #
        # # c, h, w, d = image.shape
        # #
        # # if label is None:
        # #     image = self.rng.choice(image, axis=3)
        # #
        # #     return {"image": image}
        # #
        # # assert label.shape == (h, w, d)
        # #
        # # counts = np.count_nonzero(label != 0, axis=(0, 1))
        # # counts = counts / counts.sum()
        # #
        # # idx = self.rng.choice(np.arange(counts.size), p=counts)
        #
        # image = image[..., idx]
        # label = label[..., idx]

        p = self.slice_dists[idx]

        slice_idx = self.rng.choice(np.arange(p.shape[0]), p=p)

        image = self.dataset.get_image_slice(idx, slice_idx)
        label = self.dataset.get_label_slice(idx, slice_idx)

        if label is not None:
            return {"image": image, "label": label}
        else:
            return {"image": image}
