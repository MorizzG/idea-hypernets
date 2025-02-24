from jaxtyping import Array
from typing import Any

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

import jax.random as jr
import numpy as np
import PIL.Image as Image
import yaml
from omegaconf import DictConfig, OmegaConf

import wandb
from hyper_lap.datasets import AmosSliced, Dataset, MediDecSliced, NormalisedDataset
from hyper_lap.hyper import HyperNet
from hyper_lap.models import Unet

DEFAULT_NUM_WORKERS = min((cpu_count() or -1) // 2, 64)


@dataclass
class Args:
    wandb: bool

    num_workers: int

    command: str

    artifact: str | None = None


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


def parse_args() -> tuple[Args, DictConfig]:
    parser = ArgumentParser()

    parser.add_argument("--wandb", action="store_true", help="Run with W&B logging")

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of dataloader worker threads",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    _train_parser = subparsers.add_parser("train")

    resume_parser = subparsers.add_parser("resume", help="Resume training from an artifact")

    resume_parser.add_argument("artifact", type=str)

    args, unknown_args = parser.parse_known_args()

    arg_config = OmegaConf.from_cli(unknown_args)

    return Args(**vars(args)), arg_config


def print_config(config: Any):
    print(yaml.dump(config, indent=2, width=60, default_flow_style=None, sort_keys=False))


def load_amos_datasets(normalised: bool = True) -> tuple[dict[str, Dataset], dict[str, Dataset]]:
    amos_dir = dataset_dir / "AmosSliced"

    if not amos_dir.exists():
        raise RuntimeError("AmosSliced dir doesn't exist")

    trainsets = {}
    valsets = {}

    for sub_dir in sorted(amos_dir.iterdir()):
        if not sub_dir.is_dir():
            continue

        trainset = AmosSliced(sub_dir, split="train")
        valset = AmosSliced(sub_dir, split="validation")

        if normalised:
            trainset = NormalisedDataset(trainset)
            valset = NormalisedDataset(valset)

        trainsets[trainset.name] = trainset
        valsets[valset.name] = valset

    return trainsets, valsets


def load_medidec_datasets(normalised: bool = True) -> tuple[dict[str, Dataset], dict[str, Dataset]]:
    medidec_sliced = dataset_dir / "MediDecSliced"

    if not medidec_sliced.exists():
        raise RuntimeError("MediDecSliced dir doesn't exist")

    trainsets = {}
    valsets = {}

    for sub_dir in sorted(medidec_sliced.iterdir()):
        if not sub_dir.is_dir():
            continue

        # exclude Hippocampus dataset: too small
        # exclude Prostate dataset: bad performance
        # TODO: figure out why Protate dataset performs so badly
        if "Hippocampus" in sub_dir.name or "Prostate" in sub_dir.name:
            continue

        trainset = MediDecSliced(sub_dir, split="train")
        valset = MediDecSliced(sub_dir, split="validation")

        if normalised:
            trainset = NormalisedDataset(trainset)
            valset = NormalisedDataset(valset)

        trainsets[trainset.name] = trainset
        valsets[valset.name] = valset

    return trainsets, valsets


def make_hypernet(config: dict[str, Any]) -> HyperNet:
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
