import multiprocessing
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from hyper_lap.datasets.amos_sliced import AmosSliced
from hyper_lap.datasets.medidec_sliced import MediDecSliced


@dataclass
class Args:
    degenerate: bool
    batch_size: int
    epochs: int
    num_workers: int


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

    args = parser.parse_args()

    return Args(**vars(args))


def load_amos_datasets() -> list[AmosSliced]:
    amos_dir = dataset_dir / "AmosSliced"

    if not amos_dir.exists():
        raise RuntimeError("AmosSliced dir doesn't exist")

    datasets = []

    for sub_dir in amos_dir.iterdir():
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

    for sub_dir in medidec_sliced.iterdir():
        if not sub_dir.is_dir():
            continue

        dataset = MediDecSliced(sub_dir, split="train")

        datasets.append(dataset)

    return datasets
