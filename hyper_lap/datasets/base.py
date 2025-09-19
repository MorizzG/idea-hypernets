from collections.abc import Generator

from abc import ABC, abstractmethod

import numpy as np

from .metadata import Metadata


class Dataset(ABC):
    @property
    @abstractmethod
    def metadata(self) -> Metadata: ...

    @metadata.setter
    @abstractmethod
    def metadata(self, metadata: Metadata): ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, np.ndarray]: ...

    def __iter__(self) -> Generator[dict[str, np.ndarray], None, None]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"""{self.__class__.__name__}
{self.metadata}
"""
