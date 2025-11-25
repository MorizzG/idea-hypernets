import argparse
import math
import shutil
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import matplotlib.pyplot as plt
from tqdm import trange
from wandb.apis.public import Run

import wandb
from hyper_lap.training.utils import get_datasets, load_model, print_config


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("artifact_name", type=str)
    parser.add_argument("run_name", type=str)

    parser.add_argument("num-images", type=int, default=10)

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--datasets", type=str, default=None)

    args = parser.parse_args()

    run_name: str = args.run_name
    num_images: int = getattr(args, "num-images")
    seed: int | None = args.seed

    if not num_images > 0:
        raise ValueError("num_images must be greater than 0")

    if args.datasets is not None:
        datasets = set(args.datasets.split(","))
    else:
        datasets = None

    api = wandb.Api()

    run: Run = api.run(run_name)

    image_folder = Path(f"images/{run.name}")

    if image_folder.exists():
        shutil.rmtree(image_folder)

    image_folder.mkdir(parents=True, exist_ok=False)

    config, net, embedder = load_model(run_name)

    print_config(config)

    print()
    print(f"image folder: {image_folder}")
    print()

    _trainsets, valsets, oodsets = get_datasets(
        config.dataset,
        config.trainsets.split(","),
        config.oodsets.split(","),
        batch_size=config.batch_size,
    )

    for dataset in valsets + oodsets:
        batch_size: int = dataset[0]["image"].shape[0]  # pyright: ignore

        if seed is not None:
            dataset = dataset.seed(seed).shuffle()

        dataset_name: str = dataset[0]["name"]  # pyright: ignore

        if datasets is not None and dataset_name not in datasets:
            continue

        subfolder = image_folder / dataset_name

        subfolder.mkdir(exist_ok=True)

        pbar = trange(num_images, desc=f"{dataset_name:<20}", leave=True)

        pbar_it = iter(pbar)

        try:
            for batch_num in range(math.ceil(num_images / batch_size)):
                batch: dict = jt.map(
                    lambda x: jnp.asarray(x) if eqx.is_array(x) else x, dataset[batch_num]
                )

                assert batch_size == batch["image"].shape[0]

                if embedder is not None:
                    example_image = batch["example_image"]
                    example_label = batch["example_label"]
                    dataset_idx = batch["dataset_idx"]

                    input_emb = eqx.filter_jit(embedder)(example_image, example_label, dataset_idx)
                else:
                    input_emb = None

                for k in range(batch_size // 32):
                    images = batch["image"][k * 32 : (k + 1) * 32]
                    labels = batch["label"][k * 32 : (k + 1) * 32]

                    logits = jax.jit(jax.vmap(net, in_axes=(0, None)))(images, input_emb)

                    preds = jnp.argmax(logits, axis=1)

                    for i in range(32):
                        image_idx = next(pbar_it)

                        fig, axs = plt.subplots(ncols=3, figsize=(12, 6))

                        axs[0].imshow(images[i].mean(axis=0), cmap="gray")
                        axs[1].imshow(labels[i], cmap="gray")
                        axs[2].imshow(preds[i], cmap="gray")

                        for ax in axs:
                            ax.axis("off")

                        fig.savefig(subfolder / f"{image_idx:03}.png")

                        plt.close(fig)
        except StopIteration:
            break


if __name__ == "__main__":
    main()
