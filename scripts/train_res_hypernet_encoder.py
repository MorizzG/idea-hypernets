from pathlib import Path

import equinox as eqx
import jax.random as jr
import jax.tree as jt
import wandb
from omegaconf import OmegaConf

from hyper_lap.embedder import InputEmbedder
from hyper_lap.models import HyperNet, Unet
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

    args, config = parse_args("res_hypernet")

    print_config(OmegaConf.to_object(config))

    model_name = f"res_hypernet-{config.dataset}-{config.embedder.kind}-encoder"

    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            name=args.run_name or model_name,
            config=OmegaConf.to_object(config),  # pyright: ignore
            tags=[config.dataset, config.embedder.kind, "res_hypernet", "encoder-only"],
        )

    unet_config, unet_weights_path = load_model_artifact(config.unet_artifact)

    print(f"Loading U-Net weights from {unet_weights_path}")

    unet = Unet(**unet_config["unet"], key=jr.PRNGKey(unet_config["seed"]))

    unet = load_pytree(unet_weights_path, unet, strip_prefix="0")

    filter_spec = jt.map(lambda _: False, unet)
    filter_spec = eqx.tree_at(
        lambda filter_spec: filter_spec.unet.down,
        filter_spec,
        jt.map(lambda x: eqx.is_array(x), unet.unet.down),
    )
    filter_spec = eqx.tree_at(
        lambda filter_spec: filter_spec.recomb,
        filter_spec,
        jt.map(lambda x: eqx.is_array(x), unet.recomb),
    )

    key = jr.PRNGKey(config.seed)

    model_key, embedder_key = jr.split(key)

    hypernet = HyperNet(
        unet,
        res=True,
        filter_spec=filter_spec,
        **config.hypernet,
        key=model_key,
    )

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
        loss_fn=config.loss_fn,
        model_name=model_name,
        optim_config=config.optim,
        first_epoch=first_epoch,
        num_workers=args.num_workers,
        batches_per_epoch=args.batches_per_epoch,
    )

    print("Validation before training:")
    print()

    trainer.validate(hypernet, embedder, seed=0)

    hypernet, embedder = trainer.run(hypernet, embedder, config.epochs, OmegaConf.to_object(config))

    print()
    print()

    trainer.make_plots(hypernet, embedder, image_folder=Path(f"./images/{model_name}"))

    if not args.no_umap:
        trainer.make_umap(embedder, image_folder=Path(f"./images/{model_name}"))


if __name__ == "__main__":
    hypernet: HyperNet

    main()
