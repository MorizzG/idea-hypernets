from jaxtyping import Array
from typing import Any

from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
import optax
import wandb
from omegaconf import MISSING, OmegaConf
from optax import OptState
from tqdm import tqdm, trange

from hyper_lap.datasets import Dataset
from hyper_lap.hyper import HyperNet, InputEmbedder
from hyper_lap.models import Unet
from hyper_lap.serialisation import save_with_config_safetensors
from hyper_lap.serialisation.safetensors import load_pytree
from hyper_lap.training.loss import loss_fn
from hyper_lap.training.trainer import Trainer
from hyper_lap.training.utils import (
    load_model_artifact,
    make_dataloaders,
    make_lr_schedule,
    parse_args,
    print_config,
)


@eqx.filter_jit
def training_step(
    hypernet: HyperNet,
    embedder: InputEmbedder | None,
    batch: dict[str, Array],
    opt: optax.GradientTransformation,
    opt_state: OptState,
) -> tuple[HyperNet, InputEmbedder, OptState, dict[str, Any]]:
    assert embedder is not None

    images = batch["image"]
    labels = batch["label"]
    dataset_idx = batch["dataset_idx"]

    def grad_fn(hypernet_embedder: tuple[HyperNet, InputEmbedder]) -> Array:
        hypernet, input_embedder = hypernet_embedder

        input_emb = input_embedder(images[0], labels[0], dataset_idx)

        logits = jax.vmap(hypernet, in_axes=(0, None))(images, input_emb)

        loss = jax.vmap(loss_fn)(logits, labels).mean()

        return loss

    loss, grads = eqx.filter_value_and_grad(grad_fn)((hypernet, embedder))

    aux = {"loss": loss}

    updates, opt_state = opt.update(grads, opt_state, (hypernet, embedder))  # type: ignore

    (hypernet, embedder) = eqx.apply_updates((hypernet, embedder), updates)

    return hypernet, embedder, opt_state, aux  # type: ignore


def main():
    global hypernet

    base_config = OmegaConf.create(
        {
            "seed": 42,
            "dataset": MISSING,
            "trainsets": MISSING,
            "testset": MISSING,
            "degenerate": False,
            "epochs": MISSING,
            "lr": MISSING,
            "batch_size": MISSING,
            "optim": {
                "lr": MISSING,
                "scheduler": MISSING,
                "epochs": "${epochs}",
            },
            "unet": {
                "base_channels": 8,
                "channel_mults": [1, 2, 4],
                "in_channels": 3,
                "out_channels": 2,
                "use_weight_standardized_conv": False,
            },
            "hypernet": {
                "block_size": 8,
                "input_emb_size": "${embedder.emb_size}",
                "pos_emb_size": 1024,
                "kernel_size": 3,
                "generator_kind": "basic",
            },
            "embedder": {
                "kind": "clip",
                "emb_size": 1024,
            },
        }
    )

    OmegaConf.set_readonly(base_config, True)
    OmegaConf.set_struct(base_config, True)

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

    lr_schedule = make_lr_schedule(len(train_loader), **config.optimizer)

    embedder = InputEmbedder(
        num_datasets=len(train_loader.datasets), **config.embedder, key=embedder_key
    )

    trainer: Trainer[HyperNet] = Trainer(
        hypernet,
        embedder,
        training_step,
        train_loader,
        val_loader,
        lr=lr_schedule,
        epoch=first_epoch,
    )

    for _ in trange(config.epochs):
        if wandb.run is not None:
            wandb.run.log(
                {
                    "epoch": trainer.epoch,
                    "learning_rate": trainer.learning_rate,
                }
            )

        hypernet, embedder, aux = trainer.train(hypernet, embedder)

        if wandb.run is not None:
            wandb.run.log(
                {
                    "epoch": trainer.epoch,
                    "loss/train/mean": np.mean(aux["loss"]),
                    "loss/train/std": np.std(aux["loss"]),
                }
            )
        else:
            tqdm.write(f"Loss: {np.mean(aux['loss']):.3}")
            tqdm.write("")

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
    main()
