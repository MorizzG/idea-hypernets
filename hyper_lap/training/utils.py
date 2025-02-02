from typing import Any

import multiprocessing
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.random as jr
import yaml

from hyper_lap.datasets.amos_sliced import AmosSliced
from hyper_lap.datasets.medidec_sliced import MediDecSliced
from hyper_lap.hyper.hypernet import HyperNet, HyperNetConfig
from hyper_lap.models.unet import Unet, UnetConfig


@dataclass
class Args:
    degenerate: bool
    batch_size: int
    epochs: int
    num_workers: int

    embedder: str


@dataclass
class HyperParams:
    seed: int

    unet: UnetConfig
    hypernet: HyperNetConfig

    def to_dict(self) -> dict[str, Any]:
        hyper_params_dict = asdict(self)
        hyper_params_dict["unet"] = hyper_params_dict["unet"].to_dict()
        hyper_params_dict["hypernet"] = hyper_params_dict["hypernet"].to_dict()

        return hyper_params_dict


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

    parser.add_argument("--degenerate", action="store_true", help="Use degenerate dataset")
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


def load_amos_datasets() -> list[AmosSliced]:
    amos_dir = dataset_dir / "AmosSliced"

    if not amos_dir.exists():
        raise RuntimeError("AmosSliced dir doesn't exist")

    datasets = []

    for sub_dir in sorted(amos_dir.iterdir()):
        if not sub_dir.is_dir():
            continue

        dataset = AmosSliced(sub_dir, split="train")

        datasets.append(dataset)

    return datasets


def load_medidec_datasets() -> list[MediDecSliced]:
    medidec_sliced = dataset_dir / "MediDecSliced"

    if not medidec_sliced.exists():
        raise RuntimeError("MediDecSliced dir doesn't exist")

    datasets = []

    for sub_dir in sorted(medidec_sliced.iterdir()):
        if not sub_dir.is_dir():
            continue

        # exclude Hippocampus dataset: too small
        # exclude Prostate dataset: bad performance
        # TODO: figure out why Protate dataset performs so badly
        if "Hippocampus" in sub_dir.name or "Prostate" in sub_dir.name:
            continue

        dataset = MediDecSliced(sub_dir, split="train")

        datasets.append(dataset)

    return datasets


def make_hypernet(hyper_params: HyperParams) -> tuple[Unet, HyperNet]:
    print(
        yaml.dump(
            hyper_params.to_dict(), indent=2, width=60, default_flow_style=None, sort_keys=False
        )
    )

    key = jr.PRNGKey(hyper_params.seed)
    unet_key, hypernet_key = jr.split(key)

    model_template = Unet(**hyper_params.unet.to_dict(), key=unet_key)
    hypernet = HyperNet(model_template, **hyper_params.hypernet.to_dict(), key=hypernet_key)

    return model_template, hypernet
