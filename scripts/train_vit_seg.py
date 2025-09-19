from pathlib import Path

import jax.random as jr
import wandb
from omegaconf import OmegaConf

from hyper_lap.models import VitSegmentator
from hyper_lap.training.trainer import Trainer
from hyper_lap.training.utils import (
    get_datasets,
    parse_args,
    print_config,
)


def main():
    global vit_seg

    args, config = parse_args("vit_seg")

    print_config(OmegaConf.to_object(config))

    model_name = f"vit_seg-{config.dataset}"

    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            name=args.run_name or model_name,
            config=OmegaConf.to_object(config),  # pyright: ignore
            tags=[config.dataset, "vit_seg"],
        )

    key = jr.PRNGKey(config.seed)

    vit_seg = VitSegmentator(**config.vit_seg, key=key)

    trainsets, valsets, oodsets = get_datasets(
        config.dataset,
        config.trainsets.split(","),
        config.oodsets.split(","),
        batch_size=config.batch_size,
    )

    trainer: Trainer = Trainer(
        vit_seg,
        None,
        trainsets,
        valsets,
        oodsets,
        loss_fn=config.loss_fn,
        model_name=model_name,
        optim_config=config.optim,
        num_workers=args.num_workers,
        batches_per_epoch=args.batches_per_epoch,
    )

    vit_seg, _ = trainer.run(vit_seg, None, config.epochs, OmegaConf.to_object(config))

    print()
    print()

    trainer.make_plots(vit_seg, None, image_folder=Path(f"./images/{model_name}"))


if __name__ == "__main__":
    vit_seg: VitSegmentator

    main()
