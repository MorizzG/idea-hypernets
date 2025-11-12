import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import matplotlib.pyplot as plt
import wandb
from omegaconf import DictConfig
from wandb.apis.public import Run

from hyper_lap.embedder import InputEmbedder
from hyper_lap.models import FilmUnet, HyperNet, Unet
from hyper_lap.serialisation import load_pytree
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

    parser.add_argument("num-images", type=int, default=10)

    args = parser.parse_args()

    # artifact_name = args.artifact_name
    run_name = args.run_name
    num_images: int = getattr(args, "num-images")

    api = wandb.Api()

    run: Run = api.run(run_name)

    image_folder = Path(f"images/{run.name}")

    image_folder.mkdir(parents=True, exist_ok=True)

    # artifact = api.artifact(artifact_name)

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
        batch_size=num_images,
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
        batch: dict = jt.map(lambda x: jnp.asarray(x) if eqx.is_array(x) else x, valset[0])

        dataset_name = batch.pop("name")

        images = batch["image"][:num_images]
        labels = batch["label"][:num_images]

        if embedder is not None:
            example_image = batch["example_image"]
            example_label = batch["example_label"]
            dataset_idx = batch["dataset_idx"]

            input_emb = jax.jit(embedder)(example_image, example_label, dataset_idx)
        else:
            input_emb = None

        logits = jax.jit(jax.vmap(net, in_axes=(0, None)))(images, input_emb)

        preds = jnp.argmax(logits, axis=1)

        for i in range(num_images):
            fig, axs = plt.subplots(ncols=3, figsize=(12, 6))

            axs[0].imshow(images[i].mean(axis=0), cmap="gray")
            axs[1].imshow(labels[i], cmap="gray")
            axs[2].imshow(preds[i], cmap="gray")

            for ax in axs:
                ax.axis("off")

            fig.savefig(image_folder / f"{dataset_name}_{i:03}.pdf")
            fig.savefig(image_folder / f"{dataset_name}_{i:03}.png")


if __name__ == "__main__":
    main()
