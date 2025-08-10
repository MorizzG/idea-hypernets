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
    input_embedder: InputEmbedder | None,
    batch: dict[str, Array],
    opt: optax.GradientTransformation,
    opt_state: OptState,
    *,
    lamda: float,
) -> tuple[HyperNet, InputEmbedder, OptState, dict[str, Any]]:
    assert input_embedder is not None

    images = batch["image"]
    labels = batch["label"]
    dataset_idx = batch["dataset_idx"]

    def grad_fn(
        hypernet_embedder: tuple[HyperNet, InputEmbedder],
    ) -> tuple[Array, dict[str, Any]]:
        hypernet, input_embedder = hypernet_embedder

        input_embedding = input_embedder(images[0], labels[0], dataset_idx)

        model, aux = hypernet.generate(input_embedding, with_aux=True)

        logits = jax.vmap(model)(images)

        ce_loss = jax.vmap(loss_fn)(logits, labels).mean()

        aux["ce_loss"] = ce_loss

        reg_loss = aux["reg_loss"]

        # jax.debug.print("loss={loss}, reg_loss={reg_loss}", loss=loss, reg_loss=reg_loss)

        loss = ce_loss + lamda * reg_loss

        aux["loss"] = loss

        return loss, aux

    (loss, aux), grads = eqx.filter_value_and_grad(grad_fn, has_aux=True)(
        (hypernet, input_embedder)
    )

    updates, opt_state = opt.update(grads, opt_state, (hypernet, input_embedder))  # type: ignore

    (hypernet, input_embedder) = eqx.apply_updates((hypernet, input_embedder), updates)

    return hypernet, input_embedder, opt_state, aux  # type: ignore


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
            "unet_artifact": "morizzg/idea-laplacian-hypernet/unet-medidec:v73",
            "lamda": 0.0,
            "hypernet": {
                "block_size": 8,
                "input_emb_size": "${embedder.emb_size}",
                "pos_emb_size": 1024,
                "kernel_size": 3,
                "generator_kind": "basic",
                "generator_kw_args": {
                    "h_size": 1024,
                },
            },
            "embedder": {
                "kind": "clip",
                "emb_size": 3 * 1024,
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

            unet_config, unet_weights_path = load_model_artifact(config.unet_artifact)

            print(f"Loading U-Net weights from {unet_weights_path}")

            unet = Unet(**unet_config["unet"], key=jr.PRNGKey(unet_config["seed"]))  # type: ignore

            unet = load_pytree(unet_weights_path, unet)

            hypernet = HyperNet(
                unet,
                res=True,
                **config.hypernet,
                key=jr.PRNGKey(config.seed),
            )

            first_epoch = unet_config["epochs"]

        case "resume":
            assert args.artifact is not None

            loaded_config, weights_path = load_model_artifact(args.artifact)

            config = OmegaConf.merge(loaded_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            unet_config, unet_weights_path = load_model_artifact(config.unet_artifact)

            unet = Unet(**unet_config["unet"], key=jr.PRNGKey(unet_config["seed"]))  # type: ignore

            unet = load_pytree(unet_weights_path, unet)

            hypernet = ResHyperNet(unet, **config["hypernet"], key=jr.PRNGKey(config["seed"]))  # type: ignore

            hypernet = load_pytree(weights_path, hypernet)

            first_epoch = unet_config["epochs"] + loaded_config["epochs"]

        case cmd:
            raise RuntimeError(f"Unrecognised command {cmd}")

    print_config(OmegaConf.to_object(config))

    model_name = f"reshypernet-{config.dataset}-{config.embedder.kind}"

    if wandb.run is not None:
        wandb.run.name = args.run_name or model_name
        wandb.run.config.update(OmegaConf.to_object(config))  # type: ignore
        wandb.run.tags = [config.dataset, config.embedder.kind, "res_hypernet"]

    train_loader, val_loader, test_loader = make_dataloaders(
        config.dataset,
        config.trainsets.split(","),
        config.testset,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
    )

    lr_schedule = make_lr_schedule(config.lr, config.epochs, len(train_loader))

    input_embedder = InputEmbedder(
        num_datasets=len(train_loader.datasets),
        **config.embedder,
        key=jr.PRNGKey(config.seed),
    )

    trainer: Trainer[HyperNet] = Trainer(
        hypernet,
        input_embedder,
        partial(training_step, lamda=config.lamda),
        train_loader,
        val_loader,
        lr=lr_schedule,
        epoch=first_epoch,
    )

    print("Validation before training:")
    print()

    trainer.validate(hypernet, input_embedder)

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

        hypernet, input_embedder, aux = trainer.train(hypernet, input_embedder)

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

        trainer.validate(hypernet, input_embedder)

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

    trainer.make_plots(
        hypernet, input_embedder, test_loader, image_folder=Path(f"./images/{model_name}")
    )

    if not args.no_umap:
        umap_datasets = [dataset for dataset in train_loader.datasets]

        if test_loader is not None:
            assert isinstance(test_loader.dataset, Dataset)

            umap_datasets.append(test_loader.dataset)

        trainer.make_umap(
            input_embedder, umap_datasets, image_folder=Path(f"./images/{model_name}")
        )


if __name__ == "__main__":
    main()
