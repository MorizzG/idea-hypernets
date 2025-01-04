from typing import Literal

from pathlib import Path

from .nifti import NiftiDataset


class Amos(NiftiDataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: Literal["train", "validation", "test"] = "train",
    ):
        super().__init__(root_dir, split)
