from pathlib import Path

import jax.random as jr
import wandb
from omegaconf import OmegaConf
from tqdm import trange

from hyper_lap.embedder import InputEmbedder
from hyper_lap.models import FilmUnet
from hyper_lap.serialisation import load_pytree, save_with_config_safetensors
from hyper_lap.training.trainer import Trainer
from hyper_lap.training.utils import (
    load_model_artifact,
    make_base_config,
    make_dataloaders,
    parse_args,
    print_config,
)


def main():
    global film_unet

    base_config = make_base_config("film_unet")

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

            unet_key, embedder_key = jr.split(key)

            film_unet = FilmUnet(**config.film_unet, key=unet_key)

        case "resume":
            assert args.artifact is not None

            loaded_config, weights_path = load_model_artifact(args.artifact)

            first_epoch = loaded_config["epochs"]

            config = OmegaConf.merge(loaded_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            key = jr.PRNGKey(config.seed)

            unet_key, embedder_key = jr.split(key)

            film_unet = FilmUnet(**config.film_unet, key=unet_key)

            film_unet = load_pytree(weights_path, film_unet)

        case cmd:
            raise RuntimeError(f"Unrecognised command {cmd}")

    print_config(OmegaConf.to_object(config))

    model_name = f"filmunet-{config.dataset}-{config.embedder.kind}"

    if wandb.run is not None:
        wandb.run.name = args.run_name or model_name
        wandb.run.config.update(OmegaConf.to_object(config))  # type: ignore
        wandb.run.tags = [config.dataset, config.embedder.kind, "film"]

    trainsets, valsets, testset = make_dataloaders(
        config.dataset,
        config.trainsets.split(","),
        config.testset,
        batch_size=config.batch_size,
    )

    embedder = InputEmbedder(num_datasets=len(trainsets), **config.embedder, key=embedder_key)

    trainer = Trainer(
        film_unet,
        embedder,
        trainsets,
        valsets,
        optim_config=config.optim,
        first_epoch=first_epoch,
        grad_accu=config.grad_accu,
        num_workers=args.num_workers,
    )

    for _ in trange(config.epochs):
        if wandb.run is not None:
            wandb.run.log(
                {
                    "epoch": trainer.epoch,
                    "learning_rate": trainer.learning_rate,
                }
            )

        film_unet, embedder = trainer.train(film_unet, embedder)

        trainer.validate(film_unet, embedder)

    model_path = Path(f"./models/{model_name}.safetensors")

    model_path.parent.mkdir(exist_ok=True)

    save_with_config_safetensors(model_path, OmegaConf.to_object(config), film_unet)

    if wandb.run is not None:
        model_artifact = wandb.Artifact(model_name, type="model")

        model_artifact.add_file(str(model_path.with_suffix(".json")))
        model_artifact.add_file(str(model_path.with_suffix(".safetensors")))

        wandb.run.log_artifact(model_artifact)

    print()
    print()

    trainer.make_plots(film_unet, embedder, testset, image_folder=Path(f"./images/{model_name}"))

    if not args.no_umap:
        umap_datasets = trainsets

        if testset is not None:
            umap_datasets.append(testset)

        trainer.make_umap(embedder, umap_datasets, image_folder=Path(f"./images/{model_name}"))


if __name__ == "__main__":
    film_unet: FilmUnet

    main()
