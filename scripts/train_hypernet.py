from pathlib import Path

import jax.random as jr
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from hyper_lap.datasets import Dataset
from hyper_lap.embedder import InputEmbedder
from hyper_lap.models import HyperNet, Unet
from hyper_lap.serialisation import save_with_config_safetensors
from hyper_lap.serialisation.safetensors import load_pytree
from hyper_lap.training.trainer import Trainer
from hyper_lap.training.utils import (
    load_model_artifact,
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

    first_epoch = 0

    match args.command:
        case "train":
            config = OmegaConf.merge(base_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            key = jr.PRNGKey(config.seed)

            unet_key, hypernet_key, embedder_key = jr.split(key, 3)

            unet = Unet(**config.unet, key=unet_key)

            hypernet = HyperNet(unet, **config.hypernet, res=False, key=hypernet_key)

        case "resume":
            assert args.artifact is not None

            loaded_config, weights_path = load_model_artifact(args.artifact)

            first_epoch = loaded_config["epochs"]

            config = OmegaConf.merge(loaded_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            key = jr.PRNGKey(config.seed)
            unet_key, hypernet_key, embedder_key = jr.split(key, 3)

            unet = Unet(**config.unet, key=unet_key)

            hypernet = HyperNet(unet, **config.hypernet, res=False, key=hypernet_key)

            hypernet = load_pytree(weights_path, hypernet)

        case cmd:
            raise RuntimeError(f"Unrecognised command {cmd}")

    print_config(OmegaConf.to_object(config))

    model_name = f"hypernet-{config.dataset}-{config.embedder.kind}"

    if wandb.run is not None:
        wandb.run.name = args.run_name or model_name
        wandb.run.config.update(OmegaConf.to_object(config))  # type: ignore
        wandb.run.tags = [config.dataset, config.embedder.kind, "hypernet"]

    train_loader, val_loader, test_loader = make_dataloaders(
        config.dataset,
        config.trainsets.split(","),
        config.testset,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
    )

    embedder = InputEmbedder(
        num_datasets=len(train_loader.datasets), **config.embedder, key=embedder_key
    )

    trainer: Trainer[HyperNet] = Trainer(
        hypernet,
        embedder,
        train_loader,
        val_loader,
        optim_config=config.optim,
        first_epoch=first_epoch,
        grad_accu=config.grad_accu,
    )

    for _ in trange(config.epochs):
        if wandb.run is not None:
            wandb.run.log(
                {
                    "epoch": trainer.epoch,
                    "learning_rate": trainer.learning_rate,
                }
            )
        else:
            tqdm.write(f"lr: {trainer.learning_rate:.2e}")

        hypernet, embedder = trainer.train(hypernet, embedder)

        trainer.validate(hypernet, embedder)

    model_path = Path(f"./models/{model_name}.safetensors")

    model_path.parent.mkdir(exist_ok=True)

    save_with_config_safetensors(model_path, OmegaConf.to_object(config), hypernet)

    if wandb.run is not None:
        model_artifact = wandb.Artifact(model_name, type="model")

        model_artifact.add_file(str(model_path.with_suffix(".json")))
        model_artifact.add_file(str(model_path.with_suffix(".safetensors")))

        wandb.run.log_artifact(model_artifact)

    print()
    print()

    trainer.make_plots(hypernet, embedder, test_loader, image_folder=Path(f"./images/{model_name}"))

    if not args.no_umap:
        umap_datasets = [dataset for dataset in train_loader.datasets]

        if test_loader is not None:
            assert isinstance(test_loader.dataset, Dataset)

            umap_datasets.append(test_loader.dataset)

        trainer.make_umap(embedder, umap_datasets, image_folder=Path(f"./images/{model_name}"))


if __name__ == "__main__":
    hypernet: HyperNet

    main()
