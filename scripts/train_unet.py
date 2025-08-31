from pathlib import Path

import jax.random as jr
import wandb
from omegaconf import OmegaConf

from hyper_lap.models import Unet
from hyper_lap.serialisation import load_pytree
from hyper_lap.training.trainer import Trainer
from hyper_lap.training.utils import (
    load_model_artifact,
    make_base_config,
    make_dataloaders,
    parse_args,
    print_config,
)


def main():
    global unet

    base_config = make_base_config("unet")

    args, arg_config = parse_args()

    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            # sync_tensorboard=True,
        )

    first_epoch = 0

    match args.command:
        case "train":
            config = OmegaConf.merge(base_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            unet = Unet(**config.unet, key=jr.PRNGKey(config.seed))

        case "resume":
            assert args.artifact is not None

            loaded_config, weights_path = load_model_artifact(args.artifact)

            first_epoch = loaded_config["epochs"]

            config = OmegaConf.merge(loaded_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            unet = Unet(**config.unet, key=jr.PRNGKey(config.seed))

            unet = load_pytree(weights_path, unet)

        case cmd:
            raise RuntimeError(f"Unrecognised command {cmd}")

    print_config(OmegaConf.to_object(config))

    model_name = f"unet-{config.dataset}"

    if wandb.run is not None:
        wandb.run.name = args.run_name or model_name
        wandb.run.config.update(OmegaConf.to_object(config))  # type: ignore
        wandb.run.tags = [config.dataset, "unet"]

    trainsets, valsets, testset = make_dataloaders(
        config.dataset,
        config.trainsets.split(","),
        config.testset,
        batch_size=config.batch_size,
    )

    trainer = Trainer(
        unet,
        None,
        trainsets,
        valsets,
        model_name=model_name,
        optim_config=config.optim,
        first_epoch=first_epoch,
        grad_accu=config.grad_accu,
        num_workers=args.num_workers,
    )

    unet, _ = trainer.run(unet, None, config.epochs, OmegaConf.to_object(config))

    print()
    print()

    trainer.make_plots(unet, None, testset, image_folder=Path(f"./images/{model_name}"))


if __name__ == "__main__":
    unet: Unet

    main()
