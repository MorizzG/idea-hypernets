import numpy as np

from .amos import Amos as Amos
from .medidec import MediDec as MediDec
from .nifti import NIfTIDataset as NiftiDataset


class SlicedDataset:
    dataset: NiftiDataset

    rng: np.random.Generator

    def __init__(self, dataset: NiftiDataset):
        super().__init__()

        self.dataset = dataset

        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        assert 0 <= idx < len(self), f"idx {idx} out of range"

        X = self.dataset[idx]

        image = X["image"]
        label = X.get("label", None)

        c, h, w, d = image.shape

        if label is None:
            image = self.rng.choice(image, axis=3)

            return {"image": image}

        assert label.shape == (h, w, d)

        counts = np.count_nonzero(label != 0, axis=(0, 1))
        counts = counts / counts.sum()

        idx = self.rng.choice(np.arange(counts.size), p=counts)

        image = image[..., idx]
        label = label[..., idx]

        return {"image": image, "label": label}
