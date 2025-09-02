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
    load_model_artifact,
    make_base_config,
    make_dataloaders,
    parse_args,
    print_config,
)


def main():
    global hypernet

    base_config = make_base_config("res_hypernet")

    args, arg_config = parse_args()

    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            # sync_tensorboard=True,
        )

    config = OmegaConf.merge(base_config, arg_config)

    if missing_keys := OmegaConf.missing_keys(config):
        raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

    unet_config, unet_weights_path = load_model_artifact(config.unet_artifact)

    print(f"Loading U-Net weights from {unet_weights_path}")

    unet = Unet(**unet_config["unet"], key=jr.PRNGKey(unet_config["seed"]))  # type: ignore

    unet = load_pytree(unet_weights_path, unet)

    filter_spec = jt.map(lambda _: False, unet)
    filter_spec = eqx.tree_at(
        lambda filter_spec: filter_spec.unet.up,
        filter_spec,
        jt.map(lambda x: eqx.is_array(x), unet.unet.up),
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

    print_config(OmegaConf.to_object(config))

    model_name = f"res_hypernet-{config.dataset}-{config.embedder.kind}-decoder"

    if wandb.run is not None:
        wandb.run.name = args.run_name or model_name
        wandb.run.config.update(OmegaConf.to_object(config))  # type: ignore
        wandb.run.tags = [config.dataset, config.embedder.kind, "res_hypernet", "decoder-only"]

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
        first_epoch=first_epoch,
        grad_accu=config.grad_accu,
        num_workers=args.num_workers,
    )

    print("Validation before training:")
    print()

    trainer.validate(hypernet, embedder)

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
