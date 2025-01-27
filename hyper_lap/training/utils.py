import json
import multiprocessing
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import equinox as eqx
import jax.random as jr

from hyper_lap.datasets.amos_sliced import AmosSliced
from hyper_lap.datasets.medidec_sliced import MediDecSliced
from hyper_lap.hyper.hypernet import HyperNet
from hyper_lap.models import Unet


@dataclass
class Args:
    degenerate: bool
    batch_size: int
    epochs: int
    num_workers: int

    embedder: str


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

        dataset = MediDecSliced(sub_dir, split="train")

        datasets.append(dataset)

    return datasets


def make_hypernet(hyper_params: dict) -> tuple[Unet, HyperNet]:
    pprint(hyper_params)

    unet_key, hypernet_key = jr.split(jr.PRNGKey(hyper_params["seed"]))

    model_template = Unet(**hyper_params["unet"], key=unet_key)
    hypernet = HyperNet(model_template, **hyper_params["hypernet"], key=hypernet_key)

    return model_template, hypernet


def save_hypernet(path: str | Path, hyper_params: dict, hypernet: HyperNet):
    if isinstance(path, Path):
        pass
    elif isinstance(path, str):
        path = Path(path)
    else:
        raise ValueError(f"invalid path {path}")

    with path.open("wb") as f:
        hyper_params_str = json.dumps(hyper_params)

        f.write((hyper_params_str + "\n").encode())

        eqx.tree_serialise_leaves(f, hypernet)


def load_hypernet(path: str | Path) -> tuple[Unet, HyperNet]:
    if isinstance(path, Path):
        pass
    elif isinstance(path, str):
        path = Path(path)
    else:
        raise ValueError(f"invalid path {path}")

    with path.open("rb") as f:
        hyper_params = json.loads(f.readline().decode())

        (unet, hypernet) = make_hypernet(hyper_params)

        return unet, eqx.tree_deserialise_leaves(f, hypernet)
