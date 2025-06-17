from jaxtyping import Array
from typing import Literal, Optional

import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import tqdm

from hyper_lap.datasets import MediDec

jax.config.update("jax_platform_name", "cpu")


_key = jr.PRNGKey(0)


def consume():
    global _key
    _key, consume = jr.split(_key)

    return consume


@dataclass
class Dataset:
    i: int
    name: str

    num_classes: int

    orig_dataset_folder: Path

    sliced_dataset_folder: Path

    split_medidec: dict[str, MediDec]

    split_folders: dict[str, Path]


def make_datasets(base_folder: Path):
    medidec_folder = Path("./datasets/MediDec")

    datasets = []

    for folder in medidec_folder.iterdir():
        if not folder.is_dir():
            continue

        i = int(folder.name[5:6])
        name = folder.name[7:]

        trainset = MediDec(folder, split="train")
        testset = MediDec(folder, split="test")

        dataset_folder = base_folder / f"{i:02}_{name}"

        train_folder = dataset_folder / "training"

        val_folder = dataset_folder / "validation"

        test_folder = dataset_folder / "test"

        dataset = Dataset(
            i=i,
            name=name,
            num_classes=len(trainset.metadata.labels),
            split_medidec=dict(train=trainset, test=testset),
            orig_dataset_folder=folder,
            sliced_dataset_folder=dataset_folder,
            split_folders=dict(train=train_folder, validation=val_folder, test=test_folder),
        )

        datasets.append(dataset)

    return datasets


def make_slice_dist(label: Array | None) -> Array | None:
    if label is None:
        return None

    assert label.ndim == 3, f"label has shape {label.shape}"

    counts = jnp.count_nonzero(label, axis=(0, 1))

    # if label if empty for entire volume, fall back to uniform distribution,
    # i.e. also sample empty volumes!
    if counts.sum() == 0:
        return None

    return counts / counts.sum()


@partial(jax.jit, static_argnums=(2, 3))
def normalise(
    image: Array,
    label: Optional[Array],
    num_classes: int,
    target_size: int,
) -> tuple[Array, Optional[Array]]:
    c, h, w, d = image.shape

    target_h = target_size
    target_w = target_size

    if label is not None:
        assert label.shape == (h, w, d)

    if h != target_h or w != target_w:
        image = jax.image.resize(image, (c, target_h, target_w, d), method="cubic")

        if label is not None:
            label_onehot = jax.nn.one_hot(label, num_classes, dtype=jnp.float32)

            assert label_onehot.shape == (h, w, d, num_classes)

            label_onehot = jax.image.resize(
                label_onehot, (target_h, target_w, d, num_classes), method="cubic"
            )

            # label = (label > 0.5).astype(jnp.uint8)

            # undo one-hot encoding
            label = jnp.argmax(label_onehot, axis=-1)

            assert label.shape == (target_h, target_w, d), f"{label.shape=}"

    return image, label


def make_slices(dataset: Dataset, split: Literal["train", "validation", "test"], target_size: int):
    n = 0

    split_folder = dataset.split_folders[split]

    if split_folder.exists():
        print(f"folder {split_folder} exists, skipping...")

        return

    split_folder.mkdir(parents=True, exist_ok=False)

    match split:
        case "train":
            all_items = dataset.split_medidec["train"]

            n = len(all_items)

            cutoff = int(0.8 * n)

            def get_item(i):
                return all_items[i]

            items = (get_item(i) for i in range(0, cutoff))

            total = cutoff
        case "validation":
            all_items = dataset.split_medidec["train"]

            n = len(all_items)

            cutoff = int(0.8 * n)

            def get_item(i):
                return all_items[i]

            items = (get_item(i) for i in range(cutoff, n))

            total = n - cutoff
        case "test":
            items = dataset.split_medidec["test"]

            total = None

    for n_item, X in enumerate(
        pbar := tqdm(items, leave=True, desc=f"{dataset.name} {split}", total=total)
    ):  # type: ignore
        image = jnp.asarray(X["image"])

        if "label" in X:
            label = jnp.asarray(X["label"])
        else:
            label = None

        image, label = normalise(image, label, dataset.num_classes, target_size)

        c, h, w, d = image.shape

        if label is not None:
            assert label.shape == (h, w, d)

        p = make_slice_dist(label)

        if label is not None and p is None:
            pbar.write(f"item {n_item} has empty label")

        if p is not None:
            num_candidates = jnp.count_nonzero(p).item()

            assert num_candidates != 0

            num_samples = max(num_candidates // 4, 2)
        else:
            num_samples = 2

        slice_idxs = jr.choice(consume(), d, (num_samples,), replace=False, p=p)

        image_slice = image[:, :, :, slice_idxs]

        if label is not None:
            label_slice = label[:, :, slice_idxs]

            if p is not None:
                assert jnp.all(
                    jnp.sum(label_slice[..., :num_candidates], axis=(0, 1)) != 0  # type: ignore
                ), f"{jnp.sum(label_slice, axis=(0, 1))}"
        else:
            label_slice = None

        assert image_slice.shape == (c, h, w, num_samples)

        assert label_slice is None or label_slice.shape == (h, w, num_samples)

        for k in range(num_samples):
            file = split_folder / f"{n:04}.npz"

            assert not file.exists()

            if label_slice is not None:
                np.savez_compressed(file, image=image_slice[..., k], label=label_slice[..., k])
            else:
                np.savez_compressed(file, image=image_slice[..., k])

            n += 1

            assert n <= 9999


def make_json(dataset: Dataset):
    with (dataset.orig_dataset_folder / "dataset.json").open("r") as f:
        dataset_json = json.load(f)

    dataset_folder = dataset.sliced_dataset_folder

    assert dataset_folder.exists(), f"Path {dataset_folder} doesn't exist"

    dataset_json["name"] += " sliced"

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
    target_size = int(sys.argv[1])

    base_folder = Path(f"./datasets/MediDecSliced-{target_size}")

    base_folder.mkdir(parents=True, exist_ok=True)

    datasets = make_datasets(base_folder)

    for dataset in datasets:
        print(f"Starting dataset {dataset.name}")

        make_slices(dataset, "train", target_size)
        make_slices(dataset, "validation", target_size)
        make_slices(dataset, "test", target_size)

        make_json(dataset)


if __name__ == "__main__":
    main()
