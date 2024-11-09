import nibabel as nib
import numpy as np

from .metadata import Metadata, load_metadata


class MediDec:
    _metadata: Metadata

    test: bool

    def __init__(self, root_dir: str, test: bool = False):
        super().__init__()

        self._metadata = load_metadata(root_dir)

        self.test = test

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    def __len__(self) -> int:
        if self.test:
            return len(self._metadata.test)
        else:
            return len(self._metadata.training)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        assert 0 <= idx < len(self), f"index {idx} out of range"

        if self.test:
            entry = self._metadata.test[idx]

            image = nib.load(entry["image"]).get_fdata().astype(np.float32)  # type: ignore

            return {"image": image}
        else:
            entry = self._metadata.training[idx]

            image = nib.load(entry["image"]).get_fdata().astype(np.float32)  # type: ignore
            label = nib.load(entry["label"]).get_fdata().astype(np.uint8)  # type: ignore

            return {"image": image, "label": label}
