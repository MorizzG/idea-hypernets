from jaxtyping import Array
from typing import Literal, Optional

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import tqdm

from hyper_lap.datasets import Amos

jax.config.update("jax_platform_name", "cpu")

AMOS_FOLDER = Path("./datasets/AMOS/amos22/")

BASE_FOLDER = Path("./datasets/AmosSliced")

AMOS_TRAIN = Amos(AMOS_FOLDER, split="train")
AMOS_VAL = Amos(AMOS_FOLDER, split="validation")
AMOS_TEST = Amos(AMOS_FOLDER, split="test")

LABELS = AMOS_TRAIN.metadata.labels

NUM_CLASSES = len(LABELS)

_key = jr.PRNGKey(0)


def consume():
    global _key
    _key, consume = jr.split(_key)

    return consume


@dataclass
class Dataset:
    i: int
    name: str

    dataset_folder: Path

    split_folders: dict[str, Path]

    counters: dict[str, int]


def make_datasets():
    datasets = []

    for i_str, name in LABELS.items():
        i = int(i_str)

        if i == 0:
            continue

        name = name.replace("/", "-")

        dataset_folder = BASE_FOLDER / f"{i:02} {name}"
        dataset_folder.mkdir()

        train_folder = dataset_folder / "training"
        train_folder.mkdir()

        val_folder = dataset_folder / "validation"
        val_folder.mkdir()

        test_folder = dataset_folder / "test"
        test_folder.mkdir()

        dataset = Dataset(
            i=i,
            name=name,
            dataset_folder=dataset_folder,
            split_folders=dict(train=train_folder, val=val_folder, test=test_folder),
            counters=dict(train=0, val=0, test=0),
        )

        datasets.append(dataset)

    return datasets


def make_slice_dist(label: Array | None, target: int) -> Array | None:
    if label is None:
        return None

    label = label[:, :, :, target]

    assert label.ndim == 3, f"label has shape {label.shape}"

    counts = jnp.count_nonzero(label, axis=(0, 1))

    # if label if empty for entire volume, fall back to uniform distribution,
    # i.e. also sample empty volumes!
    if counts.sum() == 0:
        return None

    return counts / counts.sum()


@jax.jit
def normalise(image: Array, label: Optional[Array]) -> tuple[Array, Optional[Array]] | None:  # type: ignore
    c, h, w, d = image.shape

    if label is not None:
        assert label.shape == (h, w, d, NUM_CLASSES)

    if h < 256 and w < 256:
        # volume too small -> continue
        return None

    if h != 512 or w != 512:
        # resize

        image = jax.image.resize(image, (c, 512, 512, d), method="cubic")

        if label is not None:
            label = label.astype(jnp.float32)

            label: Array = jax.image.resize(label, (512, 512, d, NUM_CLASSES), method="cubic")

            label = (label > 0.5).astype(jnp.uint8)

    return image, label


def make_slices(datasets: list[Dataset], amos: Amos, split: Literal["train", "val", "test"]):
    for n_item, X in enumerate((pbar := tqdm(amos))):  # type: ignore
        image = jnp.asarray(X["image"])

        if "label" in X:
            label = jnp.asarray(X["label"])

            label = jax.nn.one_hot(label, NUM_CLASSES, dtype=jnp.uint8)
        else:
            label = None

        result = normalise(image, label)

        if result is None:
            continue

        image, label = result

        c, h, w, d = image.shape

        if label is not None:
            assert label.shape == (h, w, d, NUM_CLASSES)

        for dataset in datasets:
            i = dataset.i

            split_folder = dataset.split_folders[split]

            p = make_slice_dist(label, i)

            if label is not None and p is None:
                pbar.write(f"item {n_item} has empty label {i}")

            if p is not None:
                num_candidates = jnp.count_nonzero(p).item()

                assert num_candidates != 0

                num_samples = max(num_candidates // 4, 2)
            else:
                num_samples = 2

            slice_idxs = jr.choice(consume(), d, (num_samples,), replace=False, p=p)

            image_slice = image[:, :, :, slice_idxs]

            if label is not None:
                label_slice = label[:, :, slice_idxs, i]

                if p is not None:
                    assert jnp.all(
                        jnp.sum(label_slice[..., :num_candidates], axis=(0, 1)) != 0  # type: ignore
                    ), f"{jnp.sum(label_slice, axis=(0, 1))}"
            else:
                label_slice = None

            assert image_slice.shape == (c, h, w, num_samples)

            assert label_slice is None or label_slice.shape == (h, w, num_samples)

            for k in range(num_samples):
                pbar.refresh()
                counter = dataset.counters[split]

                file = split_folder / f"{counter:04}.npz"

                assert not file.exists()

                if label_slice is not None:
                    np.savez_compressed(file, image=image_slice[..., k], label=label_slice[..., k])
                else:
                    np.savez_compressed(file, image=image_slice[..., k])

                dataset.counters[split] = counter + 1


def make_json(datasets: list[Dataset]):
    with (AMOS_FOLDER / "dataset.json").open("r") as f:
        amos_json = json.load(f)

    for dataset in datasets:
        dataset_folder = dataset.dataset_folder

        assert dataset_folder.exists(), f"Path {dataset_folder} doesn't exist"

        name = dataset_folder.name[3:]

        dataset_json = deepcopy(amos_json)
        dataset_json["name"] += f" {name} sliced"
        dataset_json["labels"] = {"0": "background", "1": name}

        splits = {}
        nums = {}

        for split in ["training", "validation", "test"]:
            folder = dataset_folder / split
            assert folder.exists(), f"folder {folder} doesn't exist"

            paths = [str(file.relative_to(dataset_folder)) for file in folder.iterdir()]

            split_cap = split[0:1].upper() + split[1:]

            nums["num" + split_cap] = len(paths)
            splits[split] = paths

        dataset_json |= nums
        dataset_json |= splits

        with (dataset_folder / "dataset.json").open("w") as f:
            json.dump(dataset_json, f, indent=4)


def main():
    BASE_FOLDER.mkdir(parents=True, exist_ok=False)

    datasets = make_datasets()

    make_slices(datasets, AMOS_TRAIN, "train")
    make_slices(datasets, AMOS_VAL, "val")
    make_slices(datasets, AMOS_TEST, "test")

    make_json(datasets)


if __name__ == "__main__":
    main()
