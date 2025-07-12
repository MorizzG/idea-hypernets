from jaxtyping import Array
from typing import Any

from functools import partial
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
from hyper_lap.hyper import ResHyperNet
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
    hypernet: ResHyperNet,
    batch: dict[str, Array],
    opt: optax.GradientTransformation,
    opt_state: OptState,
    *,
    lamda: float,
) -> tuple[ResHyperNet, OptState, dict[str, Any]]:
    images = batch["image"]
    labels = batch["label"]

    def grad_fn(hypernet: ResHyperNet) -> tuple[Array, dict[str, Any]]:
        model, aux = hypernet(images[0], labels[0], with_aux=True)

        logits = jax.vmap(model)(images)

        ce_loss = jax.vmap(loss_fn)(logits, labels).mean()

        aux["ce_loss"] = ce_loss

        reg_loss = aux["reg_loss"]

        # jax.debug.print("loss={loss}, reg_loss={reg_loss}", loss=loss, reg_loss=reg_loss)

        loss = ce_loss + lamda * reg_loss

        aux["loss"] = loss

        return loss, aux

    (loss, aux), grads = eqx.filter_value_and_grad(grad_fn, has_aux=True)(hypernet)

    updates, opt_state = opt.update(grads, opt_state, hypernet)  # type: ignore

    hypernet = eqx.apply_updates(hypernet, updates)

    return hypernet, opt_state, aux


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
            "embedder": MISSING,
            "unet_artifact": MISSING,
            "lamda": MISSING,
            "hypernet": {
                "block_size": 8,
                "input_emb_size": 3 * 1024,
                "pos_emb_size": 3 * 1024,
                "kernel_size": 3,
                "embedder_kind": "${embedder}",
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

    match args.command:
        case "train":
            config = OmegaConf.merge(base_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            unet_config, path = load_model_artifact(config.unet_artifact)

            unet = Unet(**unet_config["unet"], key=jr.PRNGKey(unet_config["seed"]))  # type: ignore

            unet = load_pytree(path, unet)

            hypernet = ResHyperNet(unet, **config["hypernet"], key=jr.PRNGKey(config["seed"]))  # type: ignore

            first_epoch = unet_config["epochs"]

        case "resume":
            assert args.artifact is not None

            loaded_config, weights_path = load_model_artifact(args.artifact)

            # TODO: remove this once runs are redone again
            if "emb_size" in loaded_config["hypernet"]:
                emb_size = loaded_config["hypernet"].pop("emb_size", None)

                loaded_config["hypernet"]["input_emb_size"] = emb_size
                loaded_config["hypernet"]["pos_emb_size"] = emb_size

            config = OmegaConf.merge(loaded_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            unet_config, path = load_model_artifact(config.unet_artifact)

            unet = Unet(**unet_config["unet"], key=jr.PRNGKey(unet_config["seed"]))  # type: ignore

            unet = load_pytree(path, unet)

            hypernet = ResHyperNet(unet, **config["hypernet"], key=jr.PRNGKey(config["seed"]))  # type: ignore

            hypernet = load_pytree(weights_path, hypernet)

            first_epoch = unet_config["epochs"] + loaded_config["epochs"]

        case cmd:
            raise RuntimeError(f"Unrecognised command {cmd}")

    print_config(OmegaConf.to_object(config))

    model_name = f"reshypernet-{config.dataset}-{config.embedder}"

    if wandb.run is not None:
        wandb.run.name = args.run_name or model_name
        wandb.run.config.update(OmegaConf.to_object(config))  # type: ignore
        wandb.run.tags = [config.dataset, config.embedder, "res_hypernet"]

    train_loader, val_loader, test_loader = make_dataloaders(
        config.dataset,
        config.trainsets.split(","),
        config.testset,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
    )

    lr_schedule = make_lr_schedule(config.lr, config.epochs, len(train_loader))

    trainer: Trainer[ResHyperNet] = Trainer(
        hypernet,
        partial(training_step, lamda=config.lamda),
        train_loader,
        val_loader,
        lr=lr_schedule,
        epoch=first_epoch,
    )

    print("Validation before training:")
    print()

    trainer.validate(hypernet)

    for _ in trange(config.epochs):
        if "lr_schedule" in vars():
            tqdm.write(f"learning rate: {trainer.learning_rate:.1e}")

            if wandb.run is not None:
                wandb.run.log(
                    {
                        "epoch": trainer.epoch,
                        "learning_rate": trainer.learning_rate,
                    }
                )

        hypernet, aux = trainer.train(hypernet)

        if wandb.run is not None:
            wandb.run.log(
                {
                    "epoch": trainer.epoch,
                    "loss/train/mean": np.mean(aux["loss"]),
                    "loss/train/std": np.std(aux["loss"]),
                    "reg_loss/train/mean": np.mean(aux["reg_loss"]),
                    "reg_loss/train/std": np.std(aux["reg_loss"]),
                }
            )
        else:
            tqdm.write(f"Loss:      {np.mean(aux['loss']):.3}")
            tqdm.write(f"Reg. Loss: {np.mean(aux['reg_loss']):.3}")
            tqdm.write("")

        trainer.validate(hypernet)

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

    trainer.make_plots(hypernet, test_loader, image_folder=Path(f"./images/{model_name}"))

    umap_datasets = [dataset for dataset in train_loader.datasets]

    if test_loader is not None:
        assert isinstance(test_loader.dataset, Dataset)

        umap_datasets.append(test_loader.dataset)

    trainer.make_umap(
        hypernet.input_embedder, umap_datasets, image_folder=Path(f"./images/{model_name}")
    )


if __name__ == "__main__":
    main()
