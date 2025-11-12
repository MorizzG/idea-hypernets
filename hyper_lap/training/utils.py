from jaxtyping import Array, PyTree
from typing import Any, Callable, Literal

import json
import time
from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import equinox as eqx
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

COMMON_CONFIG = {
    "seed": 42,
    "dataset": MISSING,
    "trainsets": MISSING,
    "oodsets": "",
    "degenerate": False,
    "epochs": 100,
    "batch_size": MISSING,
    "loss_fn": "CE",
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
    "channel_mults": [1, 2, 4, 6, 6],
    "in_channels": 3,
    "out_channels": 2,
    "kernel_size": 3,
    "use_weight_standardized_conv": False,
}

HYPERNET_CONFIG = {
    "block_size": 8,
    "input_emb_size": "${embedder.emb_size}",
    "pos_emb_size": "${embedder.emb_size}",
    "generator_kind": "basic",
}

ATTN_HYPERNET_CONFIG = {
    "block_size": 8,
    "emb_size": "${embedder.emb_size}",
    "transformer_depth": 2,
    "generator_kind": "basic",
}

FILMUNET_CONFIG = {
    "base_channels": 8,
    "channel_mults": [1, 2, 4, 6, 6],
    "in_channels": 3,
    "out_channels": 2,
    "emb_size": "${embedder.emb_size}",
    "kernel_size": 3,
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


type ModelType = Literal[
    "unet",
    "hypernet",
    "res_hypernet",
    "film_unet",
    "vit_seg",
    "attn_hypernet",
    "attn_res_hypernet",
]


def make_base_config(
    model: ModelType,
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
                "unet_artifact": MISSING,
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
                "unet_artifact": MISSING,
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


@dataclass
class Args:
    wandb: bool
    no_umap: bool

    num_workers: int
    batches_per_epoch: int

    run_name: str | None

    artifact: str | None = None


def parse_args(model: ModelType) -> tuple[Args, DictConfig]:
    parser = ArgumentParser()

    parser.add_argument("--wandb", action="store_true", help="Run with W&B logging")
    parser.add_argument("--no-umap", action="store_true", help="Disable Umap generation")

    parser.add_argument(
        "--num-workers", type=int, default=16, help="Number of dataloader worker threads"
    )
    parser.add_argument(
        "--batches-per-epoch", type=int, default=50, help="Number of batches per dataset per epoch"
    )

    parser.add_argument("--run-name", type=str, default=None, help="Run name on W&B")

    args, unknown_args = parser.parse_known_args()

    args = Args(**vars(args))

    if args.run_name and not args.wandb:
        raise ValueError("Can't set run name without wandb")

    base_config = make_base_config(model)
    arg_config = OmegaConf.from_cli(unknown_args)

    config: DictConfig = OmegaConf.merge(base_config, arg_config)  # pyright: ignore[reportAssignmentType]

    if missing_keys := OmegaConf.missing_keys(config):
        raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

    return args, config


@contextmanager
def timer(msg: str, *, use_tqdm: bool = False):
    start_time = time.perf_counter()

    try:
        yield
    finally:
        end_time = time.perf_counter()

        s = f"{msg}: {end_time - start_time:.3g}s"

        if use_tqdm:
            tqdm.write(s)
        else:
            print(s)


def with_timer(fn: Callable | None, *, msg: str, use_tqdm: bool = False):
    if fn is None:
        return partial(with_timer, msg=msg, use_tqdm=use_tqdm)

    def wrapped_fn(*args, **kw_args):
        with timer(msg, use_tqdm=use_tqdm):
            return fn(*args, **kw_args)

    return wrapped_fn


def print_config(config: Any):
    if isinstance(config, DictConfig):
        config = OmegaConf.to_object(config)

    print(yaml.dump(config, indent=2, width=60, sort_keys=False))


def load_amos_datasets(
    split: Literal["train", "validation", "test"], normalised: bool = True
) -> dict[str, Dataset]:
    amos_dir = Path("./datasets/AmosSliced")

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
    split: Literal["train", "validation", "test"], normalised: bool = True, size: int = 512
) -> dict[str, Dataset]:
    medidec_sliced = Path(f"./datasets/MediDecSliced-{size}")

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

        # if dataset.name == "BRATS":
        #     # special case: make FLAIR, T1, T2 variants of BRATS

        #     assert dataset.metadata.modality[0] == "FLAIR"
        #     assert dataset.metadata.modality[1] == "T1w"
        #     assert dataset.metadata.modality[3] == "T2w"

        #     dataset_flair = NormalisedDataset(dataset, channel=0)
        #     dataset_t1 = NormalisedDataset(dataset, channel=1)
        #     dataset_t2 = NormalisedDataset(dataset, channel=3)

        #     dataset_flair.metadata = dataset_flair.metadata.model_copy(
        #         update={"name": "BRATS-FLAIR"}
        #     )
        #     dataset_t1.metadata = dataset_flair.metadata.model_copy(update={"name": "BRATS-T1"})
        #     dataset_t2.metadata = dataset_flair.metadata.model_copy(update={"name": "BRATS-T2"})

        #     datasets["BRATS-FLAIR"] = dataset_flair
        #     datasets["BRATS-T1"] = dataset_t1
        #     datasets["BRATS-T2"] = dataset_t2

        #     continue

        if normalised:
            dataset = NormalisedDataset(dataset)

        datasets[dataset.name] = dataset

    return datasets


def get_datasets(
    dataset: Literal["amos", "medidec"],
    trainset_names: list[str],
    oodset_names: list[str],
    *,
    batch_size: int,
    degenerate: bool = False,
) -> tuple[list[MapDataset], list[MapDataset], list[MapDataset]]:
    # filter out empty strings
    trainset_names = list(filter(None, trainset_names))
    oodset_names = list(filter(None, oodset_names))

    match dataset:
        case "amos":
            all_trainsets = load_amos_datasets("train")
            all_valsets = load_amos_datasets("validation")
        case "medidec":
            all_trainsets = load_medidec_datasets("train")
            all_valsets = load_medidec_datasets("validation")
        case _:
            raise ValueError(f"Invalid dataset {dataset}")

    assert all_trainsets.keys() == all_valsets.keys(), (
        "trainsets and valsets have different keys: "
        f"{', '.join(all_trainsets.keys())} vs {', '.join(all_valsets.keys())}"
    )

    if not set(trainset_names) <= all_trainsets.keys():
        raise ValueError(
            f"invalid trainsets {trainset_names}. "
            f"valid names are: {', '.join(all_trainsets.keys())}"
        )

    if not set(oodset_names) <= all_trainsets.keys():
        raise ValueError(
            f"invalid testsets {oodset_names}. valid names are: {', '.join(all_trainsets.keys())}"
        )

    if intersect := set(trainset_names).intersection(oodset_names):
        raise ValueError(
            "Intersection between training sets and ood sets is not empty: "
            f"{', '.join(intersect)} is/are in both sets"
        )

    trainsets = {name: dataset for name, dataset in all_trainsets.items() if name in trainset_names}
    valsets = {name: dataset for name, dataset in all_valsets.items() if name in trainset_names}
    oodsets = {name: dataset for name, dataset in all_valsets.items() if name in oodset_names}

    print(f"Training Sets: {', '.join([trainset.name for trainset in trainsets.values()])}")

    if oodset_names:
        print(f"OOD Sets:      {', '.join([ood_set.name for ood_set in oodsets.values()])}")

    def make_map_dataset(i: int, dataset: Dataset, *, batch_size: int) -> MapDataset:
        return (
            MapDataset.source(dataset)
            .slice(slice(None) if not degenerate else slice(0, 1))
            .seed(0)
            .repeat()
            .shuffle()
            .batch(batch_size + 1)
            .map(
                lambda X, idx=i: {
                    "image": X["image"][1:],
                    "label": X["label"][1:],
                    "example_image": X["image"][0],
                    "example_label": X["label"][0],
                    "dataset_idx": jnp.array(idx),
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

    oodsets_grain = [
        make_map_dataset(len(trainsets_grain) + i, dataset, batch_size=2 * batch_size)
        for i, dataset in enumerate(oodsets.values())
    ]

    return trainsets_grain, valsets_grain, oodsets_grain


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
                total_steps,
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


def to_PIL(img: np.ndarray | Array) -> Image.Image:
    image: np.ndarray = np.array(img)

    assert image.ndim == 3 and image.shape[0] == 3

    image -= image.min()
    image /= image.max()

    image = (255 * image).astype(np.uint8)

    image = image.transpose(1, 2, 0)

    return Image.fromarray(image, mode="RGB")


def load_model_artifact(artifact_or_path: str) -> tuple[DictConfig, Path]:
    # first try to interpret artifact as a path
    config_path = Path(artifact_or_path).with_suffix(".json")
    weights_path = Path(artifact_or_path).with_suffix(".safetensors")

    if not (config_path.exists() and weights_path.exists()):
        # if paths do not exists, interpret artifact as a W&B artifact and try to load it
        if wandb.run is not None:
            artifact = wandb.run.use_artifact(artifact_or_path)
        else:
            api = wandb.Api()

            artifact = api.artifact(artifact_or_path)

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
        config: DictConfig = OmegaConf.create(json.load(f))  # pyright: ignore

    return config, weights_path


def global_norm(updates: PyTree) -> Array:
    """Compute the global norm across a nested structure of tensors."""
    # return jnp.sqrt(sum(jnp.sum(x**2) for x in jt.leaves(updates)))

    return jnp.sqrt(jt.reduce(lambda c, x: c + jnp.sum(x**2), updates, jnp.array(0.0)))


def peak_memory(f, *args, **kw_args):
    return (
        eqx.filter_jit(f)
        .lower(*args, **kw_args)  # pyright: ignore[reportFunctionMemberAccess]
        .lowered.compile()
        .memory_analysis()
        .peak_memory_in_bytes
    )
