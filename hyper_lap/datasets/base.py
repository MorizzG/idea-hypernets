from typing import Generator

from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from .metadata import Metadata


class Dataset(TorchDataset, ABC):
    @property
    @abstractmethod
    def metadata(self) -> Metadata: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, np.ndarray]: ...

    def __iter__(self) -> Generator[dict[str, np.ndarray], None, None]:
        for i in range(len(self)):
            yield self[i]
