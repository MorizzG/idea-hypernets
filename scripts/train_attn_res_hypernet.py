from pathlib import Path

import jax.random as jr
import wandb
from omegaconf import OmegaConf

from hyper_lap.embedder import InputEmbedder
from hyper_lap.models import AttnHyperNet, Unet
from hyper_lap.serialisation import load_pytree
from hyper_lap.training.trainer import Trainer
from hyper_lap.training.utils import (
    get_datasets,
    load_model_artifact,
    parse_args,
    print_config,
)


def main():
    global hypernet

    args, config = parse_args("attn_res_hypernet")

    print_config(OmegaConf.to_object(config))

    model_name = f"attn_res_hypernet-{config.dataset}-{config.embedder.kind}"

    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            name=args.run_name or model_name,
            config=OmegaConf.to_object(config),  # type: ignore
            tags=[config.dataset, config.embedder.kind, "attn_res_hypernet"],
        )

    unet_config, unet_weights_path = load_model_artifact(config.unet_artifact)

    print(f"Loading U-Net weights from {unet_weights_path}")

    unet = Unet(**unet_config["unet"], key=jr.PRNGKey(unet_config["seed"]))  # type: ignore

    unet = load_pytree(unet_weights_path, unet)

    key = jr.PRNGKey(config.seed)

    model_key, embedder_key = jr.split(key)

    hypernet = AttnHyperNet(unet, **config.hypernet, res=True, key=model_key)

    first_epoch = unet_config["epochs"] + 1

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
        first_epoch=first_epoch,
        num_workers=args.num_workers,
        batches_per_epoch=args.batches_per_epoch,
    )

    print("Validation before training:")
    print()

    trainer.validate(hypernet, embedder)

    hypernet, embedder = trainer.run(hypernet, embedder, config.epochs, OmegaConf.to_object(config))

    print()
    print()

    trainer.make_plots(hypernet, embedder, image_folder=Path(f"./images/{model_name}"))

    if not args.no_umap:
        trainer.make_umap(embedder, image_folder=Path(f"./images/{model_name}"))


if __name__ == "__main__":
    hypernet: AttnHyperNet

    main()
