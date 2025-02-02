from jaxtyping import PyTree

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .base import Dataset


class LidcIdri(Dataset):
    @dataclass
    class LidcIdriSlice:
        image: str

        mask0: str
        mask1: str
        mask2: str
        mask3: str

    leftover: list[LidcIdriSlice] | None = None

    items: list[LidcIdriSlice]

    def __init__(self, data_dir: str):
        super().__init__()

        self.items = []

        base_dir = Path(data_dir)

        assert base_dir.exists() and base_dir.is_dir()

        def get_slice(folder: Path, n: int) -> str:
            path = folder / f"slice-{n}.png"

            assert path.exists() and path.is_file()

            return str(path.absolute())

        for sub_folder in base_dir.iterdir():
            assert sub_folder.is_dir()

            for nodule_dir in sub_folder.iterdir():
                assert nodule_dir.is_dir()

                images_dir = nodule_dir / "images"

                mask0_dir = nodule_dir / "mask-0"
                mask1_dir = nodule_dir / "mask-1"
                mask2_dir = nodule_dir / "mask-2"
                mask3_dir = nodule_dir / "mask-3"

                n_images = len(list(images_dir.iterdir()))

                for n in range(n_images):
                    image = get_slice(images_dir, n)

                    mask0 = get_slice(mask0_dir, n)
                    mask1 = get_slice(mask1_dir, n)
                    mask2 = get_slice(mask2_dir, n)
                    mask3 = get_slice(mask3_dir, n)

                    item = LidcIdri.LidcIdriSlice(
                        image=image, mask0=mask0, mask1=mask1, mask2=mask2, mask3=mask3
                    )

                    self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> PyTree:
        if not 0 <= idx < len(self.items):
            raise IndexError(f"Index {idx} is out of range")

        def load_image(path: str) -> np.ndarray:
            with Image.open(path) as img:
                img = np.array(img)
                # uint8 -> float32
                img = img.astype(np.float32) / 255.0
                # add channel axis
                img = img[None, ...]
                return img

        def load_mask(path: str) -> np.ndarray:
            with Image.open(path) as mask:
                mask = np.array(mask)
                # normalise mask to 0, 1
                mask = np.where(mask != 0, 1, 0).astype(np.uint8)
                return mask

        slice = self.items[idx]

        X = {
            "image": load_image(slice.image),
            "masks": [
                load_mask(slice.mask0),
                load_mask(slice.mask1),
                load_mask(slice.mask2),
                load_mask(slice.mask3),
            ],
        }

        return X
