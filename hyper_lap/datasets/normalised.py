import numpy as np

from .base import Dataset
from .metadata import Metadata


class NormalisedDataset(Dataset):
    dataset: Dataset

    @property
    def metadata(self) -> Metadata:
        return self.dataset.metadata

    @property
    def name(self) -> str:
        return self.dataset.name

    @staticmethod
    def image_to_imagenet(image: np.ndarray):
        c = image.shape[0]

        if c == 3:
            pass
        elif c == 1:
            image = np.repeat(image, 3, 0)
        else:
            raise RuntimeError(f"unexpected number of channels {c}")

        ndims = image.ndim

        spatial_axes = tuple(range(ndims)[1:])

        mean = np.mean(image, axis=spatial_axes, keepdims=True)
        std = np.std(image, axis=spatial_axes, keepdims=True)

        imagenet_mean = np.expand_dims(
            np.array([0.48145466, 0.4578275, 0.40821073], dtype=image.dtype), spatial_axes
        )
        imagenet_std = np.expand_dims(
            np.array([0.26862954, 0.26130258, 0.27577711], dtype=image.dtype), spatial_axes
        )

        image_normed = (image - mean) / (std + 1e-5)

        image_normed = imagenet_std * image_normed + imagenet_mean

        return image_normed

    def __init__(self, dataset: Dataset):
        super().__init__()

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if not 0 <= idx < len(self):
            raise IndexError("index out of range")

        X = self.dataset[idx]

        # TODO: sth better than just slice out first channel here?
        image = X["image"][0:1, ...]

        X["image"] = NormalisedDataset.image_to_imagenet(image)

        return X
