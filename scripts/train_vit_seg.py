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
from hyper_lap.embedder import InputEmbedder
from hyper_lap.models import VitSegmentator
from hyper_lap.serialisation.safetensors import load_pytree, save_with_config_safetensors
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
@eqx.debug.assert_max_traces(max_traces=3)
def training_step(
    vit_seg: VitSegmentator,
    _embedder: InputEmbedder | None,
    batch: dict[str, Array],
    opt: optax.GradientTransformation,
    opt_state: OptState,
) -> tuple[VitSegmentator, InputEmbedder | None, OptState, dict[str, Any]]:
    images = batch["image"]
    labels = batch["label"]
    dataset_idx = batch["dataset_idx"]

    def grad_fn(vit_seg: VitSegmentator) -> Array:
        logits = jax.vmap(vit_seg)(images)

        loss = jax.vmap(loss_fn)(logits, labels).mean()

        return loss

    loss, grads = eqx.filter_value_and_grad(grad_fn)(vit_seg)

    aux = {"loss": loss}

    updates, opt_state = opt.update(grads, opt_state, vit_seg)  # type: ignore

    vit_seg = eqx.apply_updates(vit_seg, updates)

    return vit_seg, _embedder, opt_state, aux


def main():
    global vit_seg

    base_config = OmegaConf.create(
        {
            "seed": 42,
            "dataset": MISSING,
            "trainsets": MISSING,
            "testset": MISSING,
            "degenerate": False,
            "epochs": MISSING,
            "batch_size": MISSING,
            "optim": {
                "lr": MISSING,
                "scheduler": MISSING,
                "epochs": "${epochs}",
            },
            "vit_seg": {
                "image_size": 336,
                "patch_size": 16,
                "d_model": 512,
                "depth": 6,
                "num_heads": 8,
                "dim_head": None,
                "in_channels": 3,
                "out_channels": 2,
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

            model_key, embedder_key = jr.split(key)

            vit_seg = VitSegmentator(**config.vit_seg, key=model_key)

        case "resume":
            assert args.artifact is not None

            loaded_config, weights_path = load_model_artifact(args.artifact)

            first_epoch = loaded_config["epochs"]

            config = OmegaConf.merge(loaded_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            key = jr.PRNGKey(config.seed)

            model_key, embedder_key = jr.split(key)

            vit_seg = VitSegmentator(**config.vit_seg, key=model_key)

            vit_seg = load_pytree(weights_path, vit_seg)

        case cmd:
            raise RuntimeError(f"Unrecognised command {cmd}")

    print_config(OmegaConf.to_object(config))

    model_name = f"vit_seg-{config.dataset}"

    if wandb.run is not None:
        wandb.run.name = args.run_name or model_name
        wandb.run.config.update(OmegaConf.to_object(config))  # type: ignore
        wandb.run.tags = [config.dataset, "attention"]

    train_loader, val_loader, test_loader = make_dataloaders(
        config.dataset,
        config.trainsets.split(","),
        config.testset,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
    )

    lr_schedule = make_lr_schedule(len(train_loader), **config.optim)

    trainer: Trainer = Trainer(
        vit_seg,
        None,
        training_step,
        train_loader,
        val_loader,
        lr=lr_schedule,
        epoch=first_epoch,
    )

    for _ in trange(config.epochs):
        tqdm.write(f"Learning Rate: {trainer.learning_rate:.1e}")

        if wandb.run is not None:
            wandb.run.log(
                {
                    "epoch": trainer.epoch,
                    "learning_rate": trainer.learning_rate,
                }
            )

        vit_seg, _, aux = trainer.train(vit_seg, None)

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

        trainer.validate(vit_seg, None)

    model_path = Path(f"./models/{model_name}.safetensors")

    model_path.parent.mkdir(exist_ok=True)

    save_with_config_safetensors(model_path, OmegaConf.to_object(config), vit_seg)

    if wandb.run is not None:
        model_artifact = wandb.Artifact(model_name, type="model")

        model_artifact.add_file(str(model_path.with_suffix(".json")))
        model_artifact.add_file(str(model_path.with_suffix(".safetensors")))

        wandb.run.log_artifact(model_artifact)

    print()
    print()

    trainer.make_plots(vit_seg, None, test_loader, image_folder=Path(f"./images/{model_name}"))


if __name__ == "__main__":
    main()
