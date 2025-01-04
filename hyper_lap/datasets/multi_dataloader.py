from typing import Any, Callable, Generator

import numpy as np
from torch.utils.data import DataLoader


class MultiDataLoader:
    dataloaders: list[DataLoader]

    rng: np.random.Generator

    def __init__(self, *datasets, **dataloader_args: dict[str, Callable | Any]):
        super().__init__()

        self.rng = np.random.default_rng()

        self.dataloaders = []
        self.empty_dataloaders = []

        for dataset in datasets:
            kw_args: dict[str, Any] = {
                key: (value(dataset) if callable(value) else value)
                for key, value in dataloader_args.items()
            }

            self.dataloaders.append(DataLoader(dataset, **kw_args))

    def __len__(self) -> int:
        return sum(len(dataloader) for dataloader in self.dataloaders)

    def __iter__(self) -> Generator[Any, None, None]:
        iters = [iter(dataloader) for dataloader in self.dataloaders]

        while iters:
            iter_idx = self.rng.choice(len(iters))

            next_iter = iters[iter_idx]

            try:
                batch = next(next_iter)

                yield batch
            except StopIteration:
                iters.pop(iter_idx)
