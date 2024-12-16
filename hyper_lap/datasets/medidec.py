from typing import Literal

from .nifti import NiftiDataset


class MediDec(NiftiDataset):
    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "validation", "test"] = "train",
        # preload: bool = False,
    ):
        if split == "validation":
            raise ValueError("MediDec dataset has no validation split")

        super().__init__(root_dir, split)
