from jaxtyping import Array, PyTree
from typing import Any, Literal

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from time import time

import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import optax
import PIL.Image as Image
import wandb
import yaml
from grain import MapDataset
from omegaconf import MISSING, DictConfig, OmegaConf
from tqdm import tqdm

from hyper_lap.datasets import (
    AmosSliced,
    Dataset,
    MediDecSliced,
    NormalisedDataset,
)
from hyper_lap.models import HyperNet, Unet
from hyper_lap.serialisation import load_config, load_pytree

COMMON_CONFIG = {
    "seed": 42,
    "dataset": MISSING,
    "trainsets": MISSING,
    "testset": MISSING,
    "degenerate": False,
    "epochs": MISSING,
    "batch_size": MISSING,
    "grad_accu": 1,
    "num_validation_batches": MISSING,
    "optim": {
        "optimizer": "adamw",
        "lr": 5e-4,
        "scheduler": "cosine",
        "epochs": "${epochs}",
        "grad_clip": 1.0,
    },
}

EMBEDDER_CONFIG = {
    "kind": "clip",
    "emb_size": 1024,
}

UNET_CONFIG = {
    "base_channels": 8,
    "channel_mults": [1, 2, 4],
    "in_channels": 3,
    "out_channels": 2,
    "use_weight_standardized_conv": False,
}

HYPERNET_CONFIG = {
    "block_size": 8,
    "input_emb_size": "${embedder.emb_size}",
    "pos_emb_size": "${embedder.emb_size}",
    "kernel_size": 3,
    "generator_kind": "basic",
}

ATTN_HYPERNET_CONFIG = {
    "block_size": 8,
    "emb_size": "${embedder.emb_size}",
    "transformer_depth": 2,
    "kernel_size": 3,
    "generator_kind": "basic",
}

FILMUNET_CONFIG = {
    "base_channels": 8,
    "channel_mults": [1, 2, 4],
    "in_channels": 3,
    "out_channels": 2,
    "emb_size": "${embedder.emb_size}",
    "use_weight_standardized_conv": False,
}

VITSEG_CONFIG = {
    "image_size": 336,
    "patch_size": 16,
    "d_model": 512,
    "depth": 6,
    "num_heads": 8,
    "dim_head": None,
    "in_channels": 3,
    "out_channels": 2,
}


def make_base_config(
    model: Literal[
        "unet",
        "hypernet",
        "res_hypernet",
        "film_unet",
        "vit_seg",
        "attn_hypernet",
        "attn_res_hypernet",
    ],
) -> DictConfig:
    config = COMMON_CONFIG

    match model:
        case "unet":
            config |= {
                "unet": UNET_CONFIG,
            }
        case "hypernet":
            config |= {
                "unet": UNET_CONFIG,
                "hypernet": HYPERNET_CONFIG,
                "embedder": EMBEDDER_CONFIG,
            }
        case "res_hypernet":
            config |= {
                "unet_artifact": "morizzg/idea-laplacian-hypernet/unet-medidec:v98",
                "hypernet": HYPERNET_CONFIG,
                "embedder": EMBEDDER_CONFIG,
            }
        case "attn_hypernet":
            config |= {
                "unet": UNET_CONFIG,
                "hypernet": ATTN_HYPERNET_CONFIG,
                "embedder": EMBEDDER_CONFIG,
            }
        case "attn_res_hypernet":
            config |= {
                "unet_artifact": "morizzg/idea-laplacian-hypernet/unet-medidec:v98",
                "hypernet": ATTN_HYPERNET_CONFIG,
                "embedder": EMBEDDER_CONFIG,
            }
        case "film_unet":
            config |= {
                "film_unet": FILMUNET_CONFIG,
                "embedder": EMBEDDER_CONFIG,
            }
        case "vit_seg":
            config |= {
                "vit_seg": VITSEG_CONFIG,
            }

    base_config = OmegaConf.create(config)

    OmegaConf.set_readonly(base_config, True)
    OmegaConf.set_struct(base_config, True)

    return base_config


class Timer:
    msg: str
    pbar: tqdm | None
    start_time: float | None

    def __init__(self, msg: str, pbar: tqdm | None = None):
        super().__init__()

        self.msg = msg

        self.pbar = pbar

        self.start_time = None

    def __enter__(self):
        self.start_time = time()

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time()
        start_time = self.start_time

        assert start_time is not None

        if self.pbar:
            self.pbar.write(f"{self.msg}: {end_time - start_time:.2}s")
        else:
            print(f"{self.msg}: {end_time - start_time:.2}s")


@dataclass
class Args:
    wandb: bool
    no_umap: bool

    num_workers: int

    run_name: str | None

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
    parser.add_argument("--no-umap", action="store_true", help="Disable Umap generation")

    parser.add_argument(
        "--num-workers", type=int, default=16, help="Number of dataloader worker threads"
    )

    parser.add_argument("--run-name", type=str, default=None, help="Run name on W&B")

    subparsers = parser.add_subparsers(dest="command", required=True)

    _train_parser = subparsers.add_parser("train")

    resume_parser = subparsers.add_parser("resume", help="Resume training from an artifact")

    resume_parser.add_argument("artifact", type=str)

    args, unknown_args = parser.parse_known_args()

    args = Args(**vars(args))

    arg_config = OmegaConf.from_cli(unknown_args)

    if args.run_name and not args.wandb:
        raise ValueError("Can't set run name without wandb")

    return args, arg_config


def print_config(config: Any):
    print(yaml.dump(config, indent=2, width=60, sort_keys=False))


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

        if dataset.name == "BRATS":
            # special case: make FLAIR, T1, T2 variants of BRATS

            assert dataset.metadata.modality[0] == "FLAIR"
            assert dataset.metadata.modality[1] == "T1w"
            assert dataset.metadata.modality[3] == "T2w"

            dataset_flair = NormalisedDataset(dataset, channel=0)
            dataset_t1 = NormalisedDataset(dataset, channel=1)
            dataset_t2 = NormalisedDataset(dataset, channel=3)

            dataset_flair.metadata = dataset_flair.metadata.model_copy(
                update={"name": "BRATS-FLAIR"}
            )
            dataset_t1.metadata = dataset_flair.metadata.model_copy(update={"name": "BRATS-T1"})
            dataset_t2.metadata = dataset_flair.metadata.model_copy(update={"name": "BRATS-T2"})

            datasets["BRATS-FLAIR"] = dataset_flair
            datasets["BRATS-T1"] = dataset_t1
            datasets["BRATS-T2"] = dataset_t2

            continue

        if normalised:
            dataset = NormalisedDataset(dataset)

        datasets[dataset.name] = dataset

    return datasets


def make_dataloaders(
    dataset: Literal["amos", "medidec"],
    trainset_names: list[str],
    testset_name: str | None,
    *,
    batch_size: int,
    degenerate: bool = False,
) -> tuple[list[MapDataset], list[MapDataset], MapDataset | None]:
    match dataset:
        case "amos":
            trainsets = load_amos_datasets("train")
            valsets = load_amos_datasets("validation")
        case "medidec":
            trainsets = load_medidec_datasets("train")
            valsets = load_medidec_datasets("validation")
        case _:
            raise ValueError(f"Invalid dataset {dataset}")

    assert trainsets.keys() == valsets.keys(), (
        "trainsets and valsets have different keys: "
        f"{', '.join(trainsets.keys())} vs {', '.join(valsets.keys())}"
    )

    if not set(trainset_names) <= trainsets.keys():
        raise ValueError(
            f"invalid trainsets {trainset_names}. valid names are: {', '.join(trainsets.keys())}"
        )

    if testset_name is not None:
        testset = trainsets[testset_name]
    else:
        testset = None

    trainsets = {name: dataset for name, dataset in trainsets.items() if name in trainset_names}
    valsets = {name: dataset for name, dataset in valsets.items() if name in trainset_names}

    print(f"Trainsets: {', '.join([trainset.name for trainset in trainsets.values()])}")

    if testset is not None:
        print(f"Testset:   {testset.name}")

    # trainsets = list(trainsets.values())
    # valsets = list(valsets.values())

    def make_map_dataset(i: int, dataset: Dataset, *, batch_size: int) -> MapDataset:
        return (
            MapDataset.source(dataset)
            .slice(slice(None) if not degenerate else slice(0, 1))
            .seed(0)
            .repeat()
            .shuffle()
            .batch(batch_size)
            .map(
                lambda X, idx=jnp.array(i): X
                | {
                    "dataset_idx": idx,
                    "name": dataset.name,
                }
            )
        )

    trainsets_grain = [
        make_map_dataset(i, dataset, batch_size=batch_size)
        for i, dataset in enumerate(trainsets.values())
    ]
    valsets_grain = [
        make_map_dataset(i, dataset, batch_size=2 * batch_size)
        for i, dataset in enumerate(valsets.values())
    ]

    if testset is not None:
        testset_grain = make_map_dataset(len(trainsets), testset, batch_size=4 * batch_size)
    else:
        testset_grain = None

    # mixed_trainset = MapDataset.mix(trainsets_grain)
    # mixed_valset = MapDataset.mix(valsets_grain)

    return trainsets_grain, valsets_grain, testset_grain


def make_lr_schedule(
    len_train_loader: int,
    *,
    lr: float,
    epochs: int,
    scheduler: Literal["constant", "cosine", "exponential"],
) -> optax.Schedule:
    total_steps = epochs * len_train_loader

    warmup_steps = total_steps // 5
    transition_steps = total_steps - warmup_steps

    match scheduler:
        case "constant":
            return optax.schedules.warmup_constant_schedule(lr / 1e3, lr, warmup_steps)
        case "cosine":
            return optax.schedules.warmup_cosine_decay_schedule(
                lr / 1e3,
                lr,
                warmup_steps,
                transition_steps,
                end_value=lr / 1e3,
            )
        case "exponential":
            return optax.schedules.warmup_exponential_decay_schedule(
                lr / 1e3, lr, warmup_steps, transition_steps, 1e-3
            )
        case scheduler:
            raise ValueError(f"invalid scheduler: {scheduler}")


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
    # first try to interpret artifact as a path
    config_path = Path(name).with_suffix(".json")
    weights_path = Path(name).with_suffix(".safetensors")

    if not (config_path.exists() and weights_path.exists()):
        # if paths do not exists, interpret artifact as a W&B artifact and try to load it
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
    assert config_path.exists() and weights_path.exists()

    with config_path.open("r") as f:
        config = json.load(f)

    return config, weights_path


def global_norm(updates: PyTree) -> Array:
    """Compute the global norm across a nested structure of tensors."""
    # return jnp.sqrt(sum(jnp.sum(x**2) for x in jt.leaves(updates)))

    return jnp.sqrt(jt.reduce(lambda c, x: c + jnp.sum(x**2), updates, jnp.array(0.0)))
