from jaxtyping import Array, Float, Integer

import shutil
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import optax
import wandb
from matplotlib import pyplot as plt
from omegaconf import MISSING, OmegaConf
from optax import OptState
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from umap import UMAP

from hyper_lap.datasets import Dataset, DegenerateDataset, MultiDataLoader
from hyper_lap.hyper import HyperNet
from hyper_lap.metrics import dice_score, hausdorff_distance, jaccard_index
from hyper_lap.serialisation import save_with_config_safetensors
from hyper_lap.serialisation.safetensors import load_pytree
from hyper_lap.training.utils import (
    load_amos_datasets,
    load_medidec_datasets,
    load_model_artifact,
    make_hypernet,
    parse_args,
    print_config,
    to_PIL,
)

_key = jr.key(0)


def consume():
    global _key
    _key, _consume = jr.split(_key)
    return _consume


def calc_metrics(hypernet: HyperNet, batch: dict[str, Array]) -> dict[str, Array]:
    images = batch["image"]
    labels = batch["label"]

    model = eqx.filter_jit(hypernet)(images[0], labels[0])

    logits = eqx.filter_jit(jax.vmap(model))(images)

    preds = jnp.argmax(logits, axis=1)

    preds = preds != 0
    labels = labels != 0

    dice = jax.jit(jax.vmap(dice_score))(preds, labels).mean()

    iou = jax.jit(jax.vmap(jaccard_index))(preds, labels).mean()

    hd = np.array([hausdorff_distance(preds[i], labels[i]) for i in range(preds.shape[0])]).mean()

    return {"dice": dice, "iou": iou, "hausdorff": hd}


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


def validate(hypernet: HyperNet, val_loader: MultiDataLoader, *, pbar: tqdm, epoch: int):
    pbar.write("Validation:")
    pbar.write("")

    all_metrics: dict[str, dict[str, Array]] = {}

    for dataloader in val_loader.dataloaders:
        dataset_name: str = dataloader.dataset.name  # type: ignore

        batch = jt.map(jnp.asarray, next(iter(dataloader)))

        metrics = calc_metrics(hypernet, batch)

        pbar.write(f"Dataset: {dataset_name}:")
        pbar.write(f"    Dice score: {metrics['dice']:.3f}")
        pbar.write(f"    IoU score : {metrics['iou']:.3f}")
        pbar.write(f"    Hausdorff : {metrics['hausdorff']:.3f}")
        pbar.write("")

        all_metrics[dataset_name] = metrics

    if wandb.run is not None:
        wandb.run.log(
            {
                "epoch": epoch,
            }
            | {
                f"{metric}/{dataset_name}": val.item()
                for dataset_name, metrics in all_metrics.items()
                for metric, val in metrics.items()
            }
        )

    pbar.write("")


def make_plots(hypernet: HyperNet, val_loader: MultiDataLoader, test_loader: DataLoader):
    image_folder = Path(f"./images/{model_name}")

    if image_folder.exists():
        shutil.rmtree(image_folder)

    image_folder.mkdir(parents=True)

    for dataset, dataloader in zip(val_loader.datasets, val_loader.dataloaders):
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

        fig.savefig(image_folder / f"{dataset.name}.pdf")

        metrics = calc_metrics(hypernet, batch)

        print(f"Dataset {dataset.name}:")
        print(f"    Dice score: {metrics['dice']:.3f}")
        print(f"    IoU score : {metrics['iou']:.3f}")
        print(f"    Hausdorff : {metrics['hausdorff']:.3f}")

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

    metrics = calc_metrics(hypernet, batch)

    print(f"Dice score: {metrics['dice']:.3f}")
    print(f"IoU score : {metrics['iou']:.3f}")
    print(f"Hausdorff : {metrics['hausdorff']:.3f}")

    model = eqx.filter_jit(hypernet)(gen_image, gen_label)

    image = jnp.asarray(batch["image"][1])
    label = jnp.asarray(batch["label"][1])

    logits = eqx.filter_jit(model)(image)
    pred = jnp.argmax(logits, axis=0)

    fig, axs = plt.subplots(ncols=3)

    axs[0].imshow(image.mean(axis=0), cmap="gray")
    axs[1].imshow(label, cmap="gray")
    axs[2].imshow(pred, cmap="gray")

    fig.savefig(image_folder / f"{test_loader.dataset.name}_test.pdf")

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

    match args.command:
        case "train":
            config = OmegaConf.merge(base_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            hypernet = make_hypernet(OmegaConf.to_object(config))  # type: ignore

        case "resume":
            assert args.artifact is not None

            loaded_config, weights_path = load_model_artifact(args.artifact)

            config = OmegaConf.merge(loaded_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            hypernet = make_hypernet(OmegaConf.to_object(config))  # type: ignore

            hypernet = load_pytree(weights_path, hypernet)

        case cmd:
            raise RuntimeError(f"Unrecognised command {cmd}")

    print_config(OmegaConf.to_object(config))

    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            config=OmegaConf.to_object(config),  # type: ignore
            tags=[config.dataset, config.embedder, "hypernet"],
            # sync_tensorboard=True,
        )

    model_name = f"{Path(__file__).stem}_{config.dataset}_{config.embedder}"

    match config.dataset:
        case "amos":
            trainsets = load_amos_datasets("train")
            valsets = load_amos_datasets("validation")

            trainset_names = {"spleen", "pancreas"}
            testset_name = "liver"

            testset = trainsets.pop(testset_name)
            valsets.pop(testset_name)

            trainsets = {
                name: dataset for name, dataset in trainsets.items() if name in trainset_names
            }
            valsets = {name: dataset for name, dataset in valsets.items() if name in trainset_names}

            trainsets = list(trainsets.values())
            valsets = list(valsets.values())
        case "medidec":
            trainsets = load_medidec_datasets("train")
            valsets = load_medidec_datasets("validation")

            trainset_names = {"Liver", "Pancreas", "Lung"}
            testset_name = "Spleen"

            testset = trainsets.pop(testset_name)
            _ = valsets.pop(testset_name)

            trainsets = {
                name: dataset for name, dataset in trainsets.items() if name in trainset_names
            }
            valsets = {name: dataset for name, dataset in valsets.items() if name in trainset_names}

            trainsets = list(trainsets.values())
            valsets = list(valsets.values())
        case _:
            raise RuntimeError(f"Invalid dataset {config.dataset}")

    print(f"Trainsets: {', '.join([trainset.name for trainset in trainsets])}")
    print(f"Testset:   {testset.name}")

    if config.degenerate:
        print("Using degenerate dataset")

        trainsets = [DegenerateDataset(dataset) for dataset in trainsets]

        for dataset in trainsets:
            for X in dataset:
                # assert jnp.all(X["image"] == dataset[0]["image"])
                # assert jnp.all(X["label"] == dataset[0]["label"])

                assert eqx.tree_equal(X, dataset[0])

    train_loader = MultiDataLoader(
        *trainsets,
        num_samples=100 * config.batch_size,
        dataloader_args=dict(batch_size=config.batch_size, num_workers=args.num_workers),
    )

    val_loader = MultiDataLoader(
        *valsets,
        num_samples=2 * config.batch_size,
        dataloader_args=dict(batch_size=2 * config.batch_size, num_workers=args.num_workers),
    )

    # use 2 * batch_size for test loader since we need no grad here
    test_loader = DataLoader(testset, batch_size=2 * config.batch_size, num_workers=8)

    opt = optax.adamw(config.lr)

    opt_state = opt.init(eqx.filter(hypernet, eqx.is_array_like))

    for epoch in (pbar := trange(config.epochs)):
        pbar.write(f"Epoch {epoch:02}\n")

        hypernet, opt_state = train(hypernet, train_loader, opt, opt_state, pbar=pbar, epoch=epoch)

        validate(hypernet, val_loader, pbar=pbar, epoch=epoch)

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

    make_plots(hypernet, val_loader, test_loader)
    make_umap(hypernet, trainsets + [testset])


if __name__ == "__main__":
    main()
