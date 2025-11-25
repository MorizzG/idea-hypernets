import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import wandb
from grain import ReadOptions
from omegaconf import DictConfig
from tqdm import tqdm
from wandb.apis.public import Run

from hyper_lap.embedder import InputEmbedder
from hyper_lap.models import FilmUnet, HyperNet, Unet
from hyper_lap.serialisation import load_pytree
from hyper_lap.training.metrics import calc_metrics
from hyper_lap.training.trainer import transpose
from hyper_lap.training.utils import get_datasets, load_model_artifact, print_config


def load_unet(config: DictConfig, weights_path: Path) -> Unet:
    unet = Unet(**config.unet, key=jr.PRNGKey(config.seed))

    unet, _ = load_pytree(weights_path, (unet, None))

    return unet


def load_hypernet(
    config: DictConfig, weights_path: Path, num_datasets: int
) -> tuple[HyperNet, InputEmbedder]:
    key = jr.PRNGKey(config.seed)

    unet_key, hypernet_key, embedder_key = jr.split(key, 3)

    unet = Unet(**config.unet, key=unet_key)

    hypernet = HyperNet(unet, **config.hypernet, res=False, key=hypernet_key)

    embedder = InputEmbedder(num_datasets=num_datasets, **config.embedder, key=embedder_key)

    hypernet, embedder = load_pytree(weights_path, (hypernet, embedder))

    return hypernet, embedder


def load_film_unet(
    config: DictConfig, weights_path: Path, num_datasets: int
) -> tuple[FilmUnet, InputEmbedder]:
    key = jr.PRNGKey(config.seed)

    unet_key, embedder_key = jr.split(key)

    film_unet = FilmUnet(**config.film_unet, key=unet_key)

    embedder = InputEmbedder(num_datasets=num_datasets, **config.embedder, key=embedder_key)

    film_unet, embedder = load_pytree(weights_path, (film_unet, embedder))

    return film_unet, embedder


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("artifact_name", type=str)
    parser.add_argument("run_name", type=str)

    parser.add_argument("--num-batches", type=int, default=20)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--datasets", type=str, default=None)

    args = parser.parse_args()

    run_name: str = args.run_name
    num_batches: int = args.num_batches
    seed: int = args.seed

    if args.datasets is not None:
        datasets = set(args.datasets.split(","))
    else:
        datasets = None

    api = wandb.Api()

    run: Run = api.run(run_name)

    artifacts = [artifact for artifact in run.logged_artifacts() if artifact.type == "model"]

    if len(artifacts) == 0:
        raise RuntimeError('Cannot find "model" artifact for Run')
    elif len(artifacts) > 1:
        raise RuntimeError('Multiple "model" artifacts associated with Run')

    artifact = artifacts[0]

    config, weights_path = load_model_artifact(artifact.qualified_name)

    print_config(config)

    trainsets, valsets, oodsets = get_datasets(
        config.dataset,
        config.trainsets.split(","),
        config.oodsets.split(","),
        batch_size=config.batch_size,
    )

    num_datasets = len(trainsets) + len(oodsets)

    if "unet" in run.tags:
        net = load_unet(config, weights_path)
        embedder = None
    elif "hypernet" in run.tags:
        net, embedder = load_hypernet(config, weights_path, num_datasets)
    elif "film_unet" in run.tags:
        net, embedder = load_film_unet(config, weights_path, num_datasets)
    else:
        raise RuntimeError("Don't know how to load model")

    for valset in valsets:
        valset = valset.seed(seed).shuffle()[:num_batches]

        dataset_name = valset[0]["name"]  # pyright: ignore

        if datasets is not None and dataset_name not in datasets:
            continue

        all_metrics = []

        for batch in tqdm(
            valset.to_iter_dataset(ReadOptions(num_threads=2, prefetch_buffer_size=2)),
            desc=f"{dataset_name:<20}",
            total=num_batches,
            leave=True,
        ):
            batch: dict = jt.map(lambda x: jnp.asarray(x) if eqx.is_array(x) else x, batch)

            for i in range(config.batch_size // 32):
                images = batch["image"][32 * i : 32 * (i + 1)]
                labels = batch["label"][32 * i : 32 * (i + 1)]

                if embedder is not None:
                    example_image = batch["example_image"]
                    example_label = batch["example_label"]
                    dataset_idx = batch["dataset_idx"]

                    input_emb = eqx.filter_jit(embedder)(example_image, example_label, dataset_idx)
                else:
                    input_emb = None

                logits = jax.jit(jax.vmap(net, in_axes=(0, None)))(images, input_emb)

                dataset_metrics = calc_metrics(logits, labels)

                all_metrics.append(dataset_metrics)

        metrics = transpose(all_metrics)

        tqdm.write(f"Dataset: {dataset_name}:")
        tqdm.write(f"    Dice score: {np.mean(metrics['dice']):.5f}")
        tqdm.write(f"    IoU score : {np.mean(metrics['iou']):.5f}")
        tqdm.write(f"    Hausdorff : {np.mean(metrics['hausdorff']):.5f}")
        tqdm.write("")


if __name__ == "__main__":
    main()
