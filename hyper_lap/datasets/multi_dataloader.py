from typing import Any, Callable, Generator

import numpy as np
from torch.utils.data import DataLoader, Dataset


class MultiDataLoader:
    dataloaders: list[DataLoader]

    rng: np.random.Generator

    def __init__(self, *datasets: Dataset, **dataloader_args: Callable | Any):
        super().__init__()

        self.rng = np.random.default_rng()

        self.dataloaders = []
        self.empty_dataloaders = []

        for dataset in datasets:
            kw_args: dict[str, Any] = {
                key: (value(dataset) if callable(value) else value)
                for key, value in dataloader_args.items()
            }

            self.dataloaders.append((DataLoader(dataset, **kw_args)))

    def __len__(self) -> int:
        return sum(len(dataloader) for dataloader in self.dataloaders)

    def __iter__(self) -> Generator[dict[str, np.ndarray], None, None]:
        # iters = [(idx, iter(dataloader)) for idx, dataloader in self.dataloaders]

        iters = {i: iter(dataloader) for i, dataloader in enumerate(self.dataloaders)}
        # iters = {i: islice(iter(dataloader), 0, 5) for i, dataloader in enumerate(self.dataloaders)}

        while iters:
            idx = self.rng.choice(list(iters.keys()))

            next_iter = iters[idx]

            try:
                batch: dict[str, np.ndarray] = next(next_iter)

                yield batch
            except StopIteration:
                iters.pop(idx)
