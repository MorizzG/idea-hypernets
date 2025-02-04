import numpy as np

from .base import Dataset
from .metadata import Metadata


class DegenerateDataset(Dataset):
    len: int
    item: dict[str, np.ndarray]

    orig_dataset: Dataset

    @property
    def metadata(self) -> Metadata:
        return self.orig_dataset.metadata

    @property
    def name(self) -> str:
        return self.orig_dataset.name

    def __init__(self, dataset, idx: int = 0):
        super().__init__()

        self.len = len(dataset)

        self.item = dataset[idx]

        self.orig_dataset = dataset

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if not 0 <= idx < len(self):
            raise IndexError("index out of range")

        return self.item
