from typing import Literal

from .nifti import NIfTIDataset


class MediDec(NIfTIDataset):
    def __init__(self, root_dir: str, split: Literal["train", "validation", "test"] = "train"):
        if split == "validation":
            raise ValueError("MediDec dataset has no validation split")

        super().__init__(root_dir, split)
