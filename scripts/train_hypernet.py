from jaxtyping import Array

from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import optax
import wandb
from omegaconf import MISSING, OmegaConf
from optax import OptState
from tqdm import tqdm, trange

from hyper_lap.hyper import HyperNet
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
    batch: dict[str, Array],
    opt: optax.GradientTransformation,
    opt_state: OptState,
) -> tuple[Array, HyperNet, OptState]:
    images = batch["image"]
    labels = batch["label"]

    def grad_fn(hypernet: HyperNet) -> Array:
        model = hypernet(images[0], labels[0])

        logits = jax.vmap(model)(images)

        loss = jax.vmap(loss_fn)(logits, labels).mean()

        return loss

    loss, grads = eqx.filter_value_and_grad(grad_fn)(hypernet)

    updates, opt_state = opt.update(grads, opt_state, hypernet)  # type: ignore

    hypernet = eqx.apply_updates(hypernet, updates)

    return loss, hypernet, opt_state


def main():
    global model_name

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
            "unet": {
                "base_channels": 8,
                "channel_mults": [1, 2, 4],
                "in_channels": 3,
                "out_channels": 2,
                "use_res": False,
                "use_weight_standardized_conv": False,
            },
            "hypernet": {
                "block_size": 8,
                "emb_size": 3 * 1024,
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

    first_epoch = 0

    match args.command:
        case "train":
            config = OmegaConf.merge(base_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            key = jr.PRNGKey(config.seed)
            unet_key, hypernet_key = jr.split(key)

            unet = Unet(**config.unet, key=unet_key)
            hypernet = HyperNet(unet, **config.hypernet, key=hypernet_key)

        case "resume":
            assert args.artifact is not None

            loaded_config, weights_path = load_model_artifact(args.artifact)

            first_epoch = loaded_config["epochs"]

            config = OmegaConf.merge(loaded_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            key = jr.PRNGKey(config["seed"])  # type: ignore
            unet_key, hypernet_key = jr.split(key)

            unet = Unet(**config["unet"], key=unet_key)  # type: ignore
            hypernet = HyperNet(unet, **config["hypernet"], key=hypernet_key)  # type: ignore

            hypernet = load_pytree(weights_path, hypernet)

        case cmd:
            raise RuntimeError(f"Unrecognised command {cmd}")

    print_config(OmegaConf.to_object(config))

    if wandb.run is not None:
        wandb.run.config.update(OmegaConf.to_object(config))  # type: ignore
        wandb.run.tags = [config.dataset, config.embedder, "hypernet"]

    model_name = f"{Path(__file__).stem}_{config.dataset}_{config.embedder}"

    train_loader, val_loader, test_loader = make_dataloaders(
        config.dataset,
        config.trainsets.split(","),
        config.testset,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
    )

    lr_schedule = make_lr_schedule(config.lr, config.epochs, len(train_loader))

    trainer: Trainer[HyperNet] = Trainer(
        hypernet, training_step, train_loader, val_loader, lr=lr_schedule
    )

    for _ in trange(first_epoch, first_epoch + config.epochs):
        tqdm.write(f"Learning Rate: {trainer.learning_rate:.1e}")

        if wandb.run is not None:
            wandb.run.log(
                {
                    "epoch": trainer.epoch,
                    "learning_rate": trainer.learning_rate,
                }
            )

        hypernet = trainer.train(hypernet)

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
    umap_datasets.append(test_loader.dataset)  # type: ignore

    trainer.make_umap(
        hypernet.input_embedder, umap_datasets, image_folder=Path(f"./images/{model_name}")
    )


if __name__ == "__main__":
    main()
