from pathlib import Path

import jax.random as jr
import wandb
from omegaconf import OmegaConf

from hyper_lap.embedder import InputEmbedder
from hyper_lap.models import FilmUnet
from hyper_lap.training.trainer import Trainer
from hyper_lap.training.utils import (
    get_datasets,
    parse_args,
    print_config,
)


def main():
    global film_unet

    args, config = parse_args("film_unet")

    print_config(OmegaConf.to_object(config))

    model_name = f"filmunet-{config.dataset}-{config.embedder.kind}"

    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            name=args.run_name or model_name,
            config=OmegaConf.to_object(config),  # type: ignore
            tags=[config.dataset, config.embedder.kind, "film_unet"],
        )

    key = jr.PRNGKey(config.seed)

    unet_key, embedder_key = jr.split(key)

    film_unet = FilmUnet(**config.film_unet, key=unet_key)

    trainsets, valsets, oodsets = get_datasets(
        config.dataset,
        config.trainsets.split(","),
        config.oodsets.split(","),
        batch_size=config.batch_size,
    )

    embedder = InputEmbedder(
        num_datasets=len(trainsets) + len(oodsets), **config.embedder, key=embedder_key
    )

    trainer = Trainer(
        film_unet,
        embedder,
        trainsets,
        valsets,
        oodsets,
        loss_fn=config.loss_fn,
        model_name=model_name,
        optim_config=config.optim,
        num_workers=args.num_workers,
        batches_per_epoch=args.batches_per_epoch,
    )

    film_unet, embedder = trainer.run(
        film_unet, embedder, config.epochs, OmegaConf.to_object(config)
    )

    print()
    print()

    trainer.make_plots(film_unet, embedder, image_folder=Path(f"./images/{model_name}"))

    if not args.no_umap:
        trainer.make_umap(embedder, image_folder=Path(f"./images/{model_name}"))


if __name__ == "__main__":
    film_unet: FilmUnet

    main()
