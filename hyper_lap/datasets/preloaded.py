import numpy as np
from torch.utils.data import Dataset


class PreloadedDataset(Dataset):
    dataset: list[dict[str, np.ndarray]]

    orig_dataset: Dataset

    def __init__(self, dataset):
        super().__init__()

        from tqdm import tqdm

        self.dataset = []
        self.dataset.extend(tqdm(dataset))

        self.orig_dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < len(self.dataset):
            raise IndexError("Index out of range")

        return self.dataset[index]
