from jaxtyping import Array, Float, Integer

import warnings
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import optax
from matplotlib import pyplot as plt
from optax import OptState
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from umap import UMAP

import wandb
from hyper_lap.datasets import Dataset, DegenerateDataset, MultiDataLoader
from hyper_lap.hyper import HyperNet, HyperNetConfig
from hyper_lap.metrics import dice_score
from hyper_lap.models import UnetConfig
from hyper_lap.serialisation import save_with_config_safetensors
from hyper_lap.training.utils import (
    Config,
    load_amos_datasets,
    load_medidec_datasets,
    make_hypernet,
    parse_args,
    to_PIL,
)

warnings.simplefilter("ignore")


_key = jr.key(0)


def consume():
    global _key
    _key, _consume = jr.split(_key)
    return _consume


@jax.jit
def loss_fn(logits: Float[Array, "c h w"], labels: Integer[Array, "h w"]) -> Array:
    # C H W -> H W C
    logits = jnp.moveaxis(logits, 0, -1)

    neg_log_prob = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    # sum over spatial dims
    neg_log_likelihood = neg_log_prob.sum()

    return neg_log_likelihood


@eqx.filter_jit
def training_step(
    hypernet: HyperNet,
    opt: optax.GradientTransformation,
    batch: dict[str, Array],
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


@eqx.filter_jit
def calc_dice_score(hypernet: HyperNet, batch: dict[str, Array]):
    gen_image = batch["image"][0]
    gen_label = batch["label"][0]

    model = eqx.filter_jit(hypernet)(gen_image, gen_label)

    images = batch["image"]
    labels = batch["label"]

    labels = (labels == 1).astype(jnp.int32)

    logits = eqx.filter_jit(jax.vmap(model))(images)

    preds = jnp.argmax(logits, axis=1)

    dices = jax.jit(jax.vmap(dice_score))(preds, labels)

    return jnp.mean(dices)


def train(
    hypernet: HyperNet,
    train_loader: MultiDataLoader,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    *,
    pbar: tqdm,
    epoch: int,
) -> tuple[HyperNet, optax.OptState]:
    pbar.write("Training:\n")

    losses = []

    for batch_tensor in tqdm(train_loader, leave=False):
        batch: dict[str, Array] = jt.map(jnp.asarray, batch_tensor)

        loss, hypernet, opt_state = training_step(hypernet, opt, batch, opt_state)

        losses.append(loss.item())

    loss_mean = jnp.mean(jnp.array(losses)).item()
    loss_std = jnp.std(jnp.array(losses)).item()

    pbar.write(f"Loss: {loss_mean:.3}")

    if wandb.run is not None:
        wandb.run.log(
            {
                "epoch": epoch,
                "loss/train/mean": loss_mean,
                "loss/train/std": loss_std,
            }
        )

    return hypernet, opt_state


def validate(hypernet: HyperNet, train_loader: MultiDataLoader, *, pbar: tqdm, epoch: int):
    pbar.write("Validation:\n")

    for dataloader in train_loader.dataloaders:
        name: str = dataloader.dataset.name  # type: ignore

        batch = jt.map(jnp.asarray, next(iter(dataloader)))

        dice = calc_dice_score(hypernet, batch)

        pbar.write(f"Dice score {name: <15}: {dice:.3}")

        if wandb.run is not None:
            wandb.run.log(
                {
                    "epoch": epoch,
                    f"dice/{name}": dice,
                }
            )

    pbar.write("")


def make_plots(hypernet, train_loader: MultiDataLoader, test_loader: DataLoader):
    for dataset, dataloader in zip(train_loader.datasets, train_loader.dataloaders):
        print(f"Dataset {dataset.name}")
        print()

        batch = jt.map(jnp.asarray, next(iter(dataloader)))

        gen_image = jnp.asarray(batch["image"][0])
        gen_label = jnp.asarray(batch["label"][0])

        model = eqx.filter_jit(hypernet)(gen_image, gen_label)

        image = jnp.asarray(batch["image"][1])
        label = jnp.asarray(batch["label"][1])

        logits = eqx.filter_jit(model)(image)
        pred = jnp.argmax(logits, axis=0)

        fig, axs = plt.subplots(ncols=3)

        axs[0].imshow(image.mean(axis=0), cmap="gray")
        axs[1].imshow(label, cmap="gray")
        axs[2].imshow(pred, cmap="gray")

        fig.savefig(f"images/{model_name}_{dataset.name}.pdf")

        dice_score = calc_dice_score(hypernet, batch)

        print(f"Dice score: {dice_score:.3f}")
        # print(f"{logits.mean():.3} +/- {logits.std():.3}")

        print()
        print()

        if wandb.run is not None:
            class_labels = {0: "background", 1: "foreground"}

            image = wandb.Image(
                to_PIL(image),
                caption="Input image",
                masks={
                    "ground_truth": {"mask_data": np.asarray(label), "class_labels": class_labels},
                    "prediction": {"mask_data": np.asarray(pred), "class_labels": class_labels},
                },
            )

            wandb.run.log({f"images/train/{dataset.name}": image})

    assert isinstance(test_loader.dataset, Dataset)

    print()
    print()
    print(f"Test: {test_loader.dataset.name}")
    print()

    batch = jt.map(jnp.asarray, next(iter(test_loader)))

    gen_image = jnp.asarray(batch["image"][0])
    gen_label = jnp.asarray(batch["label"][0])

    dice = calc_dice_score(hypernet, batch)

    print((f"Dice score: {dice:.3f}"))

    model = eqx.filter_jit(hypernet)(gen_image, gen_label)

    image = jnp.asarray(batch["image"][1])
    label = jnp.asarray(batch["label"][1])

    logits = eqx.filter_jit(model)(image)
    pred = jnp.argmax(logits, axis=0)

    fig, axs = plt.subplots(ncols=3)

    axs[0].imshow(image.mean(axis=0), cmap="gray")
    axs[1].imshow(label, cmap="gray")
    axs[2].imshow(pred, cmap="gray")

    fig.savefig(f"images/{model_name}_{test_loader.dataset.name}_test.pdf")

    if wandb.run is not None:
        class_labels = {0: "background", 1: "foreground"}

        image = wandb.Image(
            to_PIL(image),
            caption="Input image",
            masks={
                "ground_truth": {"mask_data": np.asarray(label), "class_labels": class_labels},
                "prediction": {"mask_data": np.asarray(pred), "class_labels": class_labels},
            },
        )

        wandb.run.log({f"images/test/{test_loader.dataset.name}": image})


def make_umap(hypernet: HyperNet, datasets: list[Dataset]):
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

    fig.savefig(f"./images/{model_name}_umap.pdf")

    if wandb.run is not None:
        fig.canvas.draw()

        image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")  # type: ignore
        # NOTE: reversed converts (W, H) from get_width_height to (H, W)
        # convert from (H * W * 4,) to (H, W, 4)
        image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 4)

        image = wandb.Image(fig, mode="RGBA", caption="UMAP")

        wandb.run.log({"images/umap": image})


def main():
    global model_name

    args = parse_args()

    config = Config(
        seed=42,
        dataset=args.dataset,
        embedder=args.embedder,
        epochs=args.epochs,
        learning_rate=1e-7,
        unet=UnetConfig(
            base_channels=16,
            channel_mults=[1, 2, 4],
            in_channels=3,
            out_channels=2,
            use_res=False,
            use_weight_standardized_conv=True,
        ),
        hypernet=HyperNetConfig(
            block_size=8, emb_size=3 * 1024, kernel_size=3, embedder_kind=args.embedder
        ),
    )

    model_name = f"hypernet_all_{config.dataset}_{config.embedder}"

    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            config=asdict(config),
            tags=[config.dataset, config.embedder],
            # sync_tensorboard=True,
        )

    if args.dataset == "amos":
        datasets = load_amos_datasets(normalised=True)

        testset = datasets.pop("liver")

        trainsets = list(datasets.values())
    elif args.dataset == "medidec":
        datasets = load_medidec_datasets(normalised=True)

        # only use CT datasets
        datasets = {
            name: dataset
            for name, dataset in datasets.items()
            if dataset.metadata.modality["0"] == "CT"
        }

        testset = datasets.pop("HepaticVessel")

        trainsets = list(datasets.values())
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")

    if args.degenerate:
        print("Using degenerate dataset")

        datasets = [DegenerateDataset(dataset) for dataset in datasets]

        for dataset in datasets:
            for X in dataset:
                assert jnp.all(X["image"] == dataset[0]["image"])
                assert jnp.all(X["label"] == dataset[0]["label"])

    train_loader = MultiDataLoader(
        *trainsets,
        num_samples=100 * args.batch_size,
        dataloader_args=dict(batch_size=args.batch_size, num_workers=args.num_workers),
    )

    # use 2 * batch_size for test loader since we need no grad here
    test_loader = DataLoader(testset, batch_size=2 * args.batch_size, num_workers=8)

    hypernet = make_hypernet(config)

    opt = optax.adamw(config.learning_rate)

    opt_state = opt.init(eqx.filter(hypernet, eqx.is_array_like))

    for epoch in (pbar := trange(config.epochs)):
        pbar.write(f"Epoch {epoch:02}\n")

        hypernet, opt_state = train(hypernet, train_loader, opt, opt_state, pbar=pbar, epoch=epoch)

        validate(hypernet, train_loader, pbar=pbar, epoch=epoch)

    model_path = Path(f"models/{model_name}")

    save_with_config_safetensors(model_path, config, hypernet)

    if wandb.run is not None:
        model_artifact = wandb.Artifact("model-" + wandb.run.name, type="model")

        model_artifact.add_file(str(model_path.with_suffix(".json")))
        model_artifact.add_file(str(model_path.with_suffix(".safetensors")))

        wandb.run.log_artifact(model_artifact)

    print()
    print()

    make_plots(hypernet, train_loader, test_loader)
    make_umap(hypernet, trainsets + [testset])


if __name__ == "__main__":
    main()
