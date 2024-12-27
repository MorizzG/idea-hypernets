from typing import Literal

from .nifti import NiftiDataset


class Amos(NiftiDataset):
    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "validation", "test"] = "train",
    ):
        super().__init__(root_dir, split)
