from jaxtyping import Array
from typing import Any, Literal

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from time import time

import jax.random as jr
import numpy as np
import optax
import PIL.Image as Image
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf

from hyper_lap.datasets import AmosSliced, Dataset, MediDecSliced, NormalisedDataset
from hyper_lap.hyper import HyperNet
from hyper_lap.models import Unet
from hyper_lap.serialisation.safetensors import load_config, load_pytree

DEFAULT_NUM_WORKERS = min((cpu_count() or -1) // 2, 64)


class Timer:
    def __init__(self, msg: str, pbar=None):
        super().__init__()

        self.msg = msg

        self.pbar = pbar

    def __enter__(self):
        self.start_time = time()

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time()
        start_time = self.start_time

        if self.pbar:
            self.pbar.write(f"{self.msg}: {end_time - start_time:.2}s")
        else:
            print(f"{self.msg}: {end_time - start_time:.2}s")


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
    DATASET_DIR = Path(dataset_path)

    if DATASET_DIR.exists():
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


def load_amos_datasets(
    split: Literal["train", "validation", "test"], normalised: bool = True
) -> dict[str, Dataset]:
    amos_dir = DATASET_DIR / "AmosSliced"

    if not amos_dir.exists():
        raise RuntimeError("AmosSliced dir doesn't exist")

    datasets = {}

    for sub_dir in sorted(amos_dir.iterdir()):
        if not sub_dir.is_dir():
            continue

        dataset = AmosSliced(sub_dir, split=split)

        if normalised:
            dataset = NormalisedDataset(dataset)

        datasets[dataset.name] = dataset

    return datasets


def load_medidec_datasets(
    split: Literal["train", "validation", "test"], normalised: bool = True, size: int = 336
) -> dict[str, Dataset]:
    medidec_sliced = DATASET_DIR / f"MediDecSliced-{size}"

    if not medidec_sliced.exists():
        raise RuntimeError("MediDecSliced dir doesn't exist")

    datasets = {}

    for sub_dir in sorted(medidec_sliced.iterdir()):
        if not sub_dir.is_dir():
            continue

        # exclude Hippocampus dataset: too small
        # TODO: figure out why Protate dataset performs so badly
        if "Hippocampus" in sub_dir.name:
            continue

        dataset = MediDecSliced(sub_dir, split=split)

        if normalised:
            dataset = NormalisedDataset(dataset)

        datasets[dataset.name] = dataset

    return datasets


def make_lr_schedule(lr: float, epochs: int, len_train_loader) -> optax.Schedule:
    total_updates = epochs * len_train_loader

    # 20% warmup, then 80% cosine decay
    return optax.schedules.warmup_cosine_decay_schedule(
        lr / 1e3,
        lr,
        total_updates // 5,
        total_updates - total_updates // 5,
        end_value=lr / 1e-3,
    )


def make_hypernet(config: dict[str, Any]) -> HyperNet:
    key = jr.PRNGKey(config["seed"])
    unet_key, hypernet_key = jr.split(key)

    unet = Unet(**config["unet"], key=unet_key)
    hypernet = HyperNet(unet, **config["hypernet"], key=hypernet_key)

    return hypernet


def load_hypernet_safetensors(path: str | Path) -> HyperNet:
    if isinstance(path, str):
        path = Path(path)
    elif isinstance(path, Path):
        pass
    else:
        raise ValueError(f"Unexpected path {path}")

    config_path = path.with_suffix(".json")
    safetensors_path = path.with_suffix(".safetensors")

    if not (safetensors_path.exists()):
        raise ValueError(f"Path {safetensors_path} does not exist")

    config = load_config(config_path)

    hypernet = make_hypernet(config)

    hypernet = load_pytree(safetensors_path, hypernet)

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
    if wandb.run is not None:
        artifact = wandb.run.use_artifact(name)
    else:
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
