from pathlib import Path

import jax.random as jr
import wandb
from omegaconf import OmegaConf

from hyper_lap.embedder import InputEmbedder
from hyper_lap.models import AttnHyperNet, Unet
from hyper_lap.training.trainer import Trainer
from hyper_lap.training.utils import (
    get_datasets,
    make_base_config,
    parse_args,
    print_config,
)


def main():
    global hypernet

    base_config = make_base_config("attn_hypernet")

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

    hypernet = AttnHyperNet(unet, **config.hypernet, res=False, key=hypernet_key)

    print_config(OmegaConf.to_object(config))

    model_name = f"attn_hypernet-{config.dataset}-{config.embedder.kind}"

    if wandb.run is not None:
        wandb.run.name = args.run_name or model_name
        wandb.run.config.update(OmegaConf.to_object(config))  # type: ignore
        wandb.run.tags = [config.dataset, config.embedder.kind, "attn_res_hypernet"]

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
    hypernet: AttnHyperNet

    main()
