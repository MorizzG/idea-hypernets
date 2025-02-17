from jaxtyping import Array
from typing import Any, Literal

import json
import multiprocessing
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import jax.random as jr
import numpy as np
import PIL.Image as Image
import yaml

import wandb
from hyper_lap.datasets import AmosSliced, Dataset, MediDecSliced, NormalisedDataset
from hyper_lap.hyper import HyperNet
from hyper_lap.models import Unet


@dataclass
class CommonArgs:
    wandb: bool

    degenerate: bool

    lr: float
    batch_size: int
    epochs: int
    num_workers: int


@dataclass
class TrainArgs(CommonArgs):
    dataset: Literal["amos", "medidec"]

    embedder: Literal["vit", "convnext", "resnet", "clip", "learned"]

    resume: str | None


@dataclass
class ResumeArgs(CommonArgs):
    artifact: str


DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_NUM_WORKERS = min(multiprocessing.cpu_count() // 2, 64)

dataset_paths = [
    "/vol/ideadata/eg94ifeh/idea-laplacian-hypernet/datasets",
    "/media/LinuxData/datasets",
    "./datasets",
]

for dataset_path in dataset_paths:
    dataset_dir = Path(dataset_path)

    if dataset_dir.exists():
        break
else:
    raise RuntimeError("Could not determine root_dir")


def parse_args() -> TrainArgs | ResumeArgs:
    parser = ArgumentParser()

    parser.add_argument("--wandb", action="store_true", help="Run with W&B logging")

    parser.add_argument("--degenerate", action="store_true", help="Use degenerate dataset")

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of dataloader worker threads",
    )

    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train")

    train_parser.add_argument(
        "--dataset", type=str, choices=["amos", "medidec"], help="Dataset to train on"
    )
    train_parser.add_argument(
        "--embedder", type=str, choices=["vit", "convnext", "resnet", "clip", "learned"]
    )

    resume_parser = subparsers.add_parser("resume")

    resume_parser.add_argument("artifact", type=str)

    args = vars(parser.parse_args())

    match args.pop("command"):
        case "train":
            return TrainArgs(**args)
        case "resume":
            return ResumeArgs(**args)
        case cmd:
            raise RuntimeError(f"Found unexpected command {cmd}")


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


def make_hypernet(config: dict[str, Any]) -> HyperNet:
    print(yaml.dump(config, indent=2, width=60, default_flow_style=None, sort_keys=False))

    key = jr.PRNGKey(config["seed"])
    unet_key, hypernet_key = jr.split(key)

    unet = Unet(**config["unet"], key=unet_key)
    hypernet = HyperNet(unet, **config["hypernet"], key=hypernet_key)

    return hypernet


def to_PIL(img: np.ndarray | Array) -> Image.Image:
    image: np.ndarray = np.array(img)

    assert image.ndim == 3 and image.shape[0] == 3

    image -= image.min()
    image /= image.max()

    image = (255 * image).astype(np.uint8)

    image = image.transpose(1, 2, 0)

    return Image.fromarray(image, mode="RGB")


def load_model_artifact(name: str) -> tuple[dict, Path]:
    api = wandb.Api()

    artifact = api.artifact(name)

    artifact_dir = Path(artifact.download())

    config_path = None
    weights_path = None

    for file in artifact_dir.iterdir():
        if not file.is_file():
            raise RuntimeError(f"Unexpected dir entry {file}")

        match file.suffix:
            case ".json":
                config_path = file
            case ".safetensors":
                weights_path = file
            case _:
                raise RuntimeError("Unexpected file {file}")

    assert config_path is not None and weights_path is not None

    with config_path.open("r") as f:
        config = json.load(f)

    return config, weights_path
