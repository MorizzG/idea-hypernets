from pathlib import Path

import jax.random as jr
import wandb
from omegaconf import OmegaConf

from hyper_lap.embedder import InputEmbedder
from hyper_lap.models import AdaptorHyperNet, AdaptorUnet
from hyper_lap.training.trainer import Trainer
from hyper_lap.training.utils import (
    get_datasets,
    parse_args,
    print_config,
)


def main():
    global hypernet

    args, config = parse_args("hypernet")

    print_config(OmegaConf.to_object(config))

    model_name = f"hypernet-{config.dataset}-{config.embedder.kind}"
    OmegaConf.to_yaml
    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            name=args.run_name or model_name,
            config=OmegaConf.to_object(config),  # type: ignore
            tags=[config.dataset, config.embedder.kind, "adaptor_hypernet"],
        )

    key = jr.PRNGKey(config.seed)

    unet_key, hypernet_key, embedder_key = jr.split(key, 3)

    unet = AdaptorUnet(**config.unet, key=unet_key)

    hypernet = AdaptorHyperNet(unet, **config.hypernet, key=hypernet_key)

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
        hypernet,
        embedder,
        trainsets,
        valsets,
        oodsets,
        model_name=model_name,
        optim_config=config.optim,
        num_workers=args.num_workers,
    )

    hypernet, embedder = trainer.run(hypernet, embedder, config.epochs, OmegaConf.to_object(config))

    print()
    print()

    trainer.make_plots(hypernet, embedder, image_folder=Path(f"./images/{model_name}"))

    if not args.no_umap:
        trainer.make_umap(embedder, image_folder=Path(f"./images/{model_name}"))


if __name__ == "__main__":
    hypernet: AdaptorHyperNet

    main()
