from pathlib import Path

import jax.random as jr
import wandb
from omegaconf import OmegaConf

from hyper_lap.models import Unet
from hyper_lap.training.trainer import Trainer
from hyper_lap.training.utils import (
    get_datasets,
    parse_args,
    print_config,
)


def main():
    global unet

    args, config = parse_args("unet")

    print_config(OmegaConf.to_object(config))

    model_name = f"unet-{config.dataset}"

    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            name=args.run_name or model_name,
            config=OmegaConf.to_object(config),  # type: ignore
            tags=[config.dataset, "unet"],
        )

    unet = Unet(**config.unet, key=jr.PRNGKey(config.seed))

    trainsets, valsets, oodsets = get_datasets(
        config.dataset,
        config.trainsets.split(","),
        config.oodsets.split(","),
        batch_size=config.batch_size,
    )

    trainer = Trainer(
        unet,
        None,
        trainsets,
        valsets,
        oodsets,
        model_name=model_name,
        optim_config=config.optim,
        num_workers=args.num_workers,
        batches_per_epoch=args.batches_per_epoch,
    )

    unet, _ = trainer.run(unet, None, config.epochs, OmegaConf.to_object(config))

    print()
    print()

    trainer.make_plots(unet, None, image_folder=Path(f"./images/{model_name}"))


if __name__ == "__main__":
    unet: Unet

    main()
