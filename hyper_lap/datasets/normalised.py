import numpy as np
import PIL.Image as Image

from .base import Dataset
from .metadata import Metadata


class NormalisedDataset(Dataset):
    dataset: Dataset

    target_size: tuple[int, int]

    @property
    def metadata(self) -> Metadata:
        return self.dataset.metadata

    @property
    def name(self) -> str:
        return self.dataset.name

    @staticmethod
    def renormalise_to_clip(image: np.ndarray):
        if image.ndim == 2:
            image = np.expand_dims(image, 0)
        elif image.ndim == 3:
            pass
        else:
            assert False, image.shape

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

        openai_clip_mean = np.expand_dims(
            np.array([0.48145466, 0.4578275, 0.40821073], dtype=image.dtype), spatial_axes
        )
        openai_clip_std = np.expand_dims(
            np.array([0.26862954, 0.26130258, 0.27577711], dtype=image.dtype), spatial_axes
        )

        image_normed = (image - mean) / (std + 1e-5)

        image_normed = openai_clip_std * image_normed + openai_clip_mean

        return image_normed

    def __init__(self, dataset: Dataset, *, target_shape: tuple[int, int] = (336, 336)):
        super().__init__()

        self.dataset = dataset

        self.target_size = target_shape

    def resize(self, x: np.ndarray) -> np.ndarray:
        x_orig = x

        if x.ndim == 3:
            assert x.shape[0] == 1, f"{x.shape}"

            x = x.transpose(1, 2, 0)
        elif x.ndim == 2:
            pass
        else:
            raise ValueError(f"don't know how to deal with {x.ndim=}")

        img = Image.fromarray(x)

        img = img.resize(self.target_size, resample=Image.Resampling.BICUBIC)

        x = np.array(img)

        if x.ndim == 3:
            x = x.transpose(2, 0, 1)

            assert x.shape[0] == 1, f"{x.shape}"
        elif x.ndim == 2:
            pass
        else:
            assert False

        assert x.shape == x_orig.shape[:-2] + self.target_size, f"{x_orig.shape=}    {x.shape=}"

        return x

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if not 0 <= idx < len(self):
            raise IndexError("index out of range")

        X = self.dataset[idx]

        image = X.pop("image")
        label = X.pop("label")

        # TODO: sth better than just slice out first channel here?
        image = image[0, ...]

        image = self.resize(image)
        label = self.resize(label)

        image = self.renormalise_to_clip(image)

        label = (label != 0).astype(np.uint8)

        X |= {"image": image, "label": label}

        return X
