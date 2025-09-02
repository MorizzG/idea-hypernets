from pathlib import Path

import jax.random as jr
import wandb
from omegaconf import OmegaConf

from hyper_lap.embedder import InputEmbedder
from hyper_lap.models import HyperNet, Unet
from hyper_lap.training.trainer import Trainer
from hyper_lap.training.utils import (
    make_base_config,
    make_dataloaders,
    parse_args,
    print_config,
)


def main():
    global hypernet

    base_config = make_base_config("hypernet")

    args, arg_config = parse_args()

    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            # sync_tensorboard=True,
        )

    config = OmegaConf.merge(base_config, arg_config)

    if missing_keys := OmegaConf.missing_keys(config):
        raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

    key = jr.PRNGKey(config.seed)

    unet_key, hypernet_key, embedder_key = jr.split(key, 3)

    unet = Unet(**config.unet, key=unet_key)

    hypernet = HyperNet(unet, **config.hypernet, res=False, key=hypernet_key)

    print_config(OmegaConf.to_object(config))

    model_name = f"hypernet-{config.dataset}-{config.embedder.kind}"

    if wandb.run is not None:
        wandb.run.name = args.run_name or model_name
        wandb.run.config.update(OmegaConf.to_object(config))  # type: ignore
        wandb.run.tags = [config.dataset, config.embedder.kind, "hypernet"]

    trainsets, valsets, testset = make_dataloaders(
        config.dataset,
        config.trainsets.split(","),
        config.testset,
        batch_size=config.batch_size,
    )

    embedder = InputEmbedder(num_datasets=len(trainsets), **config.embedder, key=embedder_key)

    trainer = Trainer(
        hypernet,
        embedder,
        trainsets,
        valsets,
        model_name=model_name,
        optim_config=config.optim,
        grad_accu=config.grad_accu,
        num_workers=args.num_workers,
    )

    hypernet, embedder = trainer.run(hypernet, embedder, config.epochs, OmegaConf.to_object(config))

    print()
    print()

    trainer.make_plots(hypernet, embedder, testset, image_folder=Path(f"./images/{model_name}"))

    if not args.no_umap:
        umap_datasets = trainsets

        if testset is not None:
            umap_datasets.append(testset)

        trainer.make_umap(embedder, umap_datasets, image_folder=Path(f"./images/{model_name}"))


if __name__ == "__main__":
    hypernet: HyperNet

    main()
