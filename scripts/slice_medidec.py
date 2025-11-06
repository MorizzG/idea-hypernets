from jaxtyping import Array
from typing import Literal

import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import Queue, ShutDown
from threading import Event, Thread

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from grain import MapDataset, ReadOptions
from tqdm import tqdm

from hyper_lap.datasets import MediDec

type Split = Literal["train", "validation", "test"]

jax.config.update("jax_platform_name", "cpu")

MEDIDEC_FOLDER = Path("./datasets/MediDec")

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

    input_channel: int

    modality: str

    orig_dataset_folder: Path

    sliced_dataset_folder: Path

    sources: dict[Split, MapDataset]

    split_folders: dict[Split, Path]

    counters: dict[str, int]


class AsyncFileWriter:
    q: Queue
    thread: Thread

    shutdown: Event

    def __init__(self):
        self.q = Queue(3)
        self.thread = Thread(target=self._worker)
        self.shutdown = Event()

        self.thread.start()

    def _worker(self):
        """Worker thread that writes files from queue"""
        while True:
            try:
                item = self.q.get()
            except ShutDown:
                break

            path, image, label = item

            np.savez_compressed(path, image=image, label=label)

            self.q.task_done()

    def queue(self, path: str | Path, image: np.ndarray, label: np.ndarray):
        """Queue a BytesIO buffer to be written to file"""
        if self.shutdown.is_set():
            raise RuntimeError("Writer is shut down")

        assert image.ndim == 2 and image.dtype == np.float32
        assert label.ndim == 2 and label.dtype == np.uint8

        self.q.put((path, image, label))

    def wait(self):
        """Wait for all queued writes to complete"""

        self.q.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown.set()
        self.q.shutdown()

        self.q.join()
        self.thread.join()


def make_datasets(base_folder: Path):
    datasets = []

    for folder in sorted(list(MEDIDEC_FOLDER.iterdir())):
        if not folder.is_dir():
            continue

        i = int(folder.name[5:6])
        name = folder.name[7:]

        trainset = MediDec(folder, split="train")

        source = MapDataset.source(trainset)  # pyright: ignore

        total_len = len(source)

        train_source = source[: int(0.6 * total_len)]
        val_source = source[int(0.6 * total_len) : int(0.8 * total_len)]
        test_source = source[int(0.8 * total_len) :]

        if len(trainset.metadata.modality) > 1:
            for channel, subname in trainset.metadata.modality.items():
                dataset_folder = base_folder / f"{i:02}_{name}_{subname}"

                train_folder = dataset_folder / "training"

                val_folder = dataset_folder / "validation"

                test_folder = dataset_folder / "test"

                dataset = Dataset(
                    i=i,
                    name=f"{name}_{subname}",
                    input_channel=channel,
                    modality=subname,
                    num_classes=len(trainset.metadata.labels),
                    sources={"train": train_source, "validation": val_source, "test": test_source},
                    orig_dataset_folder=folder,
                    sliced_dataset_folder=dataset_folder,
                    split_folders={
                        "train": train_folder,
                        "validation": val_folder,
                        "test": test_folder,
                    },
                    counters=dict(train=0, validation=0, test=0),
                )

                datasets.append(dataset)

            continue

        dataset_folder = base_folder / f"{i:02}_{name}"

        train_folder = dataset_folder / "training"

        val_folder = dataset_folder / "validation"

        test_folder = dataset_folder / "test"

        dataset = Dataset(
            i=i,
            name=name,
            input_channel=0,
            modality=trainset.metadata.modality[0],
            num_classes=len(trainset.metadata.labels),
            sources={"train": train_source, "validation": val_source, "test": test_source},
            orig_dataset_folder=folder,
            sliced_dataset_folder=dataset_folder,
            split_folders={"train": train_folder, "validation": val_folder, "test": test_folder},
            counters=dict(train=0, validation=0, test=0),
        )

        datasets.append(dataset)

    return datasets


def make_slice_dist(label: Array) -> Array | None:
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
    label: Array,
    num_classes: int,
    target_size: int,
) -> tuple[Array, Array]:
    h, w, d = image.shape

    assert label.shape == (h, w, d)

    if h != target_size or w != target_size:
        image = jax.image.resize(image, (target_size, target_size, d), method="cubic")

        label_onehot = jax.nn.one_hot(label, num_classes, dtype=jnp.float32)

        assert label_onehot.shape == (h, w, d, num_classes)

        label_onehot = jax.image.resize(
            label_onehot, (target_size, target_size, d, num_classes), method="cubic"
        )

        # label = (label > 0.5).astype(jnp.uint8)

        # undo one-hot encoding
        label = jnp.argmax(label_onehot, axis=-1).astype(jnp.uint8)

        assert label.shape == (target_size, target_size, d), f"{label.shape=}"

    return image, label


def make_slices(
    dataset: Dataset,
    split: Split,
    *,
    target_size: int,
):
    source = dataset.sources[split]

    split_folder = dataset.split_folders[split]

    if split_folder.exists():
        print(f"folder {split_folder} exists, skipping...")

        return

    split_folder.mkdir(parents=True, exist_ok=False)

    with AsyncFileWriter() as async_writer:
        for n_item, X in enumerate(
            tqdm(
                source.to_iter_dataset(
                    read_options=ReadOptions(num_threads=16, prefetch_buffer_size=16)
                ),
                total=len(source),
            )
        ):
            image = jnp.asarray(X["image"])
            label = jnp.asarray(X["label"])

            assert image.ndim == 4
            assert label.ndim == 3

            image = image[dataset.input_channel, ...]

            image, label = normalise(image, label, dataset.num_classes, target_size)

            h, w, d = image.shape

            assert label.shape == (h, w, d)

            p = make_slice_dist(label)

            if p is not None:
                num_candidates = jnp.count_nonzero(p).item()

                assert num_candidates != 0

                num_samples = max(num_candidates // 4, 2)
            else:
                tqdm.write(f"item {n_item} has empty label")

                num_samples = 2

            slice_idxs = jr.choice(consume(), d, (num_samples,), replace=False, p=p)

            image_slice = image[:, :, slice_idxs]

            label_slice = label[:, :, slice_idxs]

            if p is not None:
                assert jnp.all(
                    jnp.sum(label_slice[..., :num_candidates], axis=(0, 1)) != 0  # pyright: ignore
                ), f"{jnp.sum(label_slice, axis=(0, 1))}"

            assert image_slice.shape == (h, w, num_samples)

            assert label_slice.shape == (h, w, num_samples)

            for k in range(num_samples):
                counter = dataset.counters[split]

                path = split_folder / f"{counter:04}.npz"

                assert not path.exists()

                # if label_slice is not None:
                #     np.savez_compressed(
                #         path, image=image_slice[..., k], label=label_slice[..., k]
                #     )
                # else:
                #     np.savez_compressed(path, image=image_slice[..., k])

                async_writer.queue(
                    path, np.asarray(image_slice[..., k]), np.asarray(label_slice[..., k])
                )

                dataset.counters[split] = counter + 1

                assert counter <= 9999


def make_json(dataset: Dataset):
    with (dataset.orig_dataset_folder / "dataset.json").open("r") as f:
        dataset_json = json.load(f)

    dataset_folder = dataset.sliced_dataset_folder

    assert dataset_folder.exists(), f"Path {dataset_folder} doesn't exist"

    dataset_json["name"] = dataset.name + " sliced"
    dataset_json["modality"] = {"0": dataset.modality}

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

        make_slices(dataset, "train", target_size=target_size)
        make_slices(dataset, "validation", target_size=target_size)
        make_slices(dataset, "test", target_size=target_size)

        make_json(dataset)


if __name__ == "__main__":
    main()
