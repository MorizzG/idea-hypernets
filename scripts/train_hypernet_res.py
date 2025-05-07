from jaxtyping import Array

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import optax
import wandb
from matplotlib import pyplot as plt
from omegaconf import MISSING, OmegaConf
from optax import OptState
from tqdm import tqdm, trange
from umap import UMAP

from hyper_lap.datasets import Dataset, MultiDataLoader
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
) -> tuple[Array, ResHyperNet, OptState]:
    images = batch["image"]
    labels = batch["label"]

    def grad_fn(hypernet: ResHyperNet) -> Array:
        model = hypernet(images[0], labels[0])

        logits = jax.vmap(model)(images)

        loss = jax.vmap(loss_fn)(logits, labels).mean()

        return loss

    loss, grads = eqx.filter_value_and_grad(grad_fn)(hypernet)

    updates, opt_state = opt.update(grads, opt_state, hypernet)  # type: ignore

    hypernet = eqx.apply_updates(hypernet, updates)

    return loss, hypernet, opt_state


def make_umap(hypernet: ResHyperNet, datasets: list[Dataset]):
    image_folder = Path(f"./images/{model_name}")

    assert image_folder.exists()

    embedder = hypernet.input_embedder

    embedder = eqx.filter_jit(eqx.filter_vmap(embedder))

    multi_dataloader = MultiDataLoader(
        *datasets,
        num_samples=100,
        dataloader_args=dict(batch_size=100, num_workers=8),
    )

    samples = {
        dataset.name: jt.map(jnp.asarray, next(iter(dataloader)))
        for dataset, dataloader in zip(multi_dataloader.datasets, multi_dataloader.dataloaders)
    }

    embs = {name: embedder(X["image"], X["label"]) for name, X in samples.items()}

    umap = UMAP()
    umap.fit(jnp.concat([embs for embs in embs.values()]))

    projs: dict[str, Array] = {name: umap.transform(embs) for name, embs in embs.items()}  # type: ignore

    fig, ax = plt.subplots()

    for name, proj in projs.items():
        ax.scatter(proj[:, 0], proj[:, 1], 4.0, label=name)

    pos = ax.get_position()
    ax.set_position((pos.x0, pos.y0, pos.width * 0.75, pos.height))

    fig.legend(loc="outside center right")

    fig.savefig(image_folder / "umap.pdf")

    if wandb.run is not None:
        image = wandb.Image(fig, mode="RGBA", caption="UMAP")

        wandb.run.log({"images/umap": image})


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

    unet_config, path = load_model_artifact(
        "morizzg/idea-laplacian-hypernet/unet_all_medidec:latest"
    )

    unet = Unet(**unet_config["unet"], key=jr.PRNGKey(unet_config["seed"]))  # type: ignore

    unet = load_pytree(path, unet)

    first_epoch = unet_config["epochs"]

    match args.command:
        case "train":
            config = OmegaConf.merge(base_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            hypernet = ResHyperNet(unet, **config["hypernet"], key=jr.PRNGKey(config["seed"]))  # type: ignore

        case "resume":
            assert args.artifact is not None

            loaded_config, weights_path = load_model_artifact(args.artifact)

            config = OmegaConf.merge(loaded_config, arg_config)

            first_epoch += config.epochs

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            hypernet = ResHyperNet(unet, **config["hypernet"], key=jr.PRNGKey(config["seed"]))  # type: ignore

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

    opt = optax.adamw(lr_schedule)
    # opt = optax.adamw(config.lr)

    opt_state = opt.init(eqx.filter(hypernet, eqx.is_array_like))

    trainer: Trainer[ResHyperNet] = Trainer(train_loader, val_loader, opt, opt_state, training_step)

    print("Validation before training:")
    print()

    trainer.validate(hypernet)

    for epoch in trange(first_epoch, first_epoch + config.epochs):
        tqdm.write(f"Epoch {trainer.epoch:02}\n")

        # if isinstance(trainer.opt_state[-1], optax.ScaleByScheduleState):
        #     tqdm.write(f"learning rate: {lr_schedule(trainer.opt_state[-1].count):.1e}")

        #     if wandb.run is not None:
        #         wandb.run.log(
        #             {
        #                 "epoch": epoch,
        #                 "learning_rate": lr_schedule(opt_state[2].count),  # type: ignore
        #             }
        #         )

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

    # make_umap(hypernet, trainsets + [testset])


if __name__ == "__main__":
    main()
