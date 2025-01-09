import numpy as np
from torch.utils.data import Dataset


class DegenerateDataset(Dataset):
    len: int
    item: dict[str, np.ndarray]

    orig_dataset: Dataset

    def __init__(self, dataset):
        super().__init__()

        self.len = len(dataset)

        self.item = dataset[0]

        self.orig_dataset = dataset

    def __len__(self) -> int:
        return self.len

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if not 0 <= idx < len(self):
            raise IndexError("index out of range")

        return self.item
