from jaxtyping import Array
from typing import Literal

import multiprocessing
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.random as jr
import numpy as np
import PIL.Image as Image
import yaml

from hyper_lap.datasets import AmosSliced, Dataset, MediDecSliced, NormalisedDataset
from hyper_lap.hyper import HyperNet, HyperNetConfig
from hyper_lap.models import Unet, UnetConfig


@dataclass
class Args:
    dataset: Literal["amos", "medidec"]

    degenerate: bool
    wandb: bool

    batch_size: int
    epochs: int
    num_workers: int

    embedder: Literal["vit", "convnext", "resnet", "clip", "learned"]


@dataclass
class Config:
    seed: int

    dataset: Literal["amos", "medidec"]
    embedder: Literal["vit", "convnext", "resnet", "clip", "learned"]

    epochs: int

    learning_rate: float

    unet: UnetConfig
    hypernet: HyperNetConfig


DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_NUM_WORKERS = min(multiprocessing.cpu_count() // 2, 64)


dataset_dir = Path("/vol/ideadata/eg94ifeh/idea-laplacian-hypernet/datasets")

if not dataset_dir.exists():
    dataset_dir = Path("/media/LinuxData/datasets")

if not dataset_dir.exists():
    raise RuntimeError("Could not determine root_dir")


def parse_args() -> Args:
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset", type=str, choices=["amos", "medidec"], help="Dataset to train on"
    )

    parser.add_argument("--degenerate", action="store_true", help="Use degenerate dataset")
    parser.add_argument("--wandb", action="store_true", help="Run with W&B logging")

    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs to train for"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of dataloader worker threads",
    )
    parser.add_argument(
        "--embedder", type=str, choices=["vit", "convnext", "resnet", "clip", "learned"]
    )

    args = parser.parse_args()

    return Args(**vars(args))


def load_amos_datasets(normalised: bool = True) -> dict[str, Dataset]:
    amos_dir = dataset_dir / "AmosSliced"

    if not amos_dir.exists():
        raise RuntimeError("AmosSliced dir doesn't exist")

    datasets = {}

    for sub_dir in sorted(amos_dir.iterdir()):
        if not sub_dir.is_dir():
            continue

        dataset = AmosSliced(sub_dir, split="train")

        if normalised:
            dataset = NormalisedDataset(dataset)

        datasets[dataset.name] = dataset

    return datasets


def load_medidec_datasets(normalised: bool = True) -> dict[str, Dataset]:
    medidec_sliced = dataset_dir / "MediDecSliced"

    if not medidec_sliced.exists():
        raise RuntimeError("MediDecSliced dir doesn't exist")

    datasets = {}

    for sub_dir in sorted(medidec_sliced.iterdir()):
        if not sub_dir.is_dir():
            continue

        # exclude Hippocampus dataset: too small
        # exclude Prostate dataset: bad performance
        # TODO: figure out why Protate dataset performs so badly
        if "Hippocampus" in sub_dir.name or "Prostate" in sub_dir.name:
            continue

        dataset = MediDecSliced(sub_dir, split="train")

        if normalised:
            dataset = NormalisedDataset(dataset)

        datasets[dataset.name] = dataset

    return datasets


def make_hypernet(config: Config) -> HyperNet:
    print(yaml.dump(asdict(config), indent=2, width=60, default_flow_style=None, sort_keys=False))

    key = jr.PRNGKey(config.seed)
    unet_key, hypernet_key = jr.split(key)

    unet = Unet(**asdict(config.unet), key=unet_key)
    hypernet = HyperNet(unet, **asdict(config.hypernet), key=hypernet_key)

    return hypernet


def to_PIL(img: np.ndarray | Array) -> Image.Image:
    image: np.ndarray = np.array(img)

    assert image.ndim == 3 and image.shape[0] == 3

    image -= image.min()
    image /= image.max()

    image = (255 * image).astype(np.uint8)

    image = image.transpose(1, 2, 0)

    return Image.fromarray(image, mode="RGB")
