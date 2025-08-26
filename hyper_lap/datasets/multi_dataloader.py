from typing import Generator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .base import Dataset


class MultiDataLoader:
    datasets: list[Dataset]
    dataloaders: list[DataLoader]

    rng: np.random.Generator

    def __init__(
        self,
        *datasets: Dataset,
        num_samples: Optional[int] = None,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()

        self.rng = np.random.default_rng()

        self.datasets = list(datasets)

        self.dataloaders = []

        for dataset in datasets:
            generator = torch.Generator()
            generator.manual_seed(42)

            sampler = RandomSampler(dataset, num_samples=num_samples, generator=generator)

            data_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=(num_workers != 0),
            )

            _ = next(iter(data_loader))

            self.dataloaders.append(data_loader)

    def __len__(self) -> int:
        return sum(len(dataloader) for dataloader in self.dataloaders)

    def __iter__(self) -> Generator[dict[str, np.ndarray], None, None]:
        iters = {i: iter(dataloader) for i, dataloader in enumerate(self.dataloaders)}

        while iters:
            idx = self.rng.choice(list(iters.keys()))

            next_iter = iters[idx]

            try:
                batch: dict[str, np.ndarray] = next(next_iter)

                batch["dataset_idx"] = np.array(idx)

                batch = {name: np.asarray(val) for name, val in batch.items()}

                yield batch
            except StopIteration:
                iters.pop(idx)
