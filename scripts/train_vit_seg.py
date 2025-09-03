from pathlib import Path

import jax.random as jr
import wandb
from omegaconf import OmegaConf

from hyper_lap.models import VitSegmentator
from hyper_lap.training.trainer import Trainer
from hyper_lap.training.utils import (
    make_base_config,
    make_dataloaders,
    parse_args,
    print_config,
)


def main():
    global vit_seg

    base_config = make_base_config("vit_seg")

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

    model_key, embedder_key = jr.split(key)

    vit_seg = VitSegmentator(**config.vit_seg, key=model_key)

    print_config(OmegaConf.to_object(config))

    model_name = f"vit_seg-{config.dataset}"

    if wandb.run is not None:
        wandb.run.name = args.run_name or model_name
        wandb.run.config.update(OmegaConf.to_object(config))  # type: ignore
        wandb.run.tags = [config.dataset, "attention"]

    trainsets, valsets, testset = make_dataloaders(
        config.dataset,
        config.trainsets.split(","),
        config.testset,
        batch_size=config.batch_size,
    )

    trainer: Trainer = Trainer(
        vit_seg,
        None,
        trainsets,
        valsets,
        model_name=model_name,
        optim_config=config.optim,
        num_workers=args.num_workers,
    )

    vit_seg, _ = trainer.run(vit_seg, None, config.epochs, OmegaConf.to_object(config))

    print()
    print()

    trainer.make_plots(vit_seg, None, testset, image_folder=Path(f"./images/{model_name}"))


if __name__ == "__main__":
    vit_seg: VitSegmentator

    main()
