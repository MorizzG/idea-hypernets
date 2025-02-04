import numpy as np

from .base import Dataset
from .metadata import Metadata


class PreloadedDataset(Dataset):
    dataset: list[dict[str, np.ndarray]]

    orig_dataset: Dataset

    @property
    def metadata(self) -> Metadata:
        return self.orig_dataset.metadata

    @property
    def name(self) -> str:
        return self.orig_dataset.name

    def __init__(self, dataset):
        super().__init__()

        from tqdm import tqdm

        self.dataset = []
        self.dataset.extend(tqdm(dataset))

        self.orig_dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < len(self.dataset):
            raise IndexError("Index out of range")

        return self.dataset[index]
