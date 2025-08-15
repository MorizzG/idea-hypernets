import concurrent.futures
from os import cpu_count

import numpy as np
from tqdm import tqdm

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

        self.orig_dataset = dataset

        # self.dataset = []
        # self.dataset.extend(tqdm(dataset))

        n = len(dataset)

        max_workers = min((cpu_count() or 32) + 4, 32)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            self.dataset = list(tqdm(executor.map(lambda i: dataset[i], range(n)), total=n))

        # self.dataset = thread_map(lambda i: dataset[i], range(len(dataset)))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < len(self.dataset):
            raise IndexError("Index out of range")

        return self.dataset[index]
