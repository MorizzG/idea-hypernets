from jaxtyping import Array, Float, Integer

import shutil
import warnings
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
from hyper_lap.metrics import dice_score, hausdorff_distance, jaccard_index
from hyper_lap.models import FilmUnet
from hyper_lap.serialisation.safetensors import load_pytree, save_with_config_safetensors
from hyper_lap.training.utils import (
    load_amos_datasets,
    load_medidec_datasets,
    load_model_artifact,
    parse_args,
    print_config,
    to_PIL,
)

warnings.simplefilter("ignore")


def calc_metrics(film_unet: FilmUnet, batch: dict[str, Array]) -> dict[str, Array]:
    images = batch["image"]
    labels = batch["label"]

    cond_emb = eqx.filter_jit(film_unet.embedder)(images[0], labels[0])

    logits = eqx.filter_jit(jax.vmap(film_unet, in_axes=(0, None)))(images, cond_emb)

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
    film_unet: FilmUnet,
    opt: optax.GradientTransformation,
    batch: dict[str, Array],
    opt_state: OptState,
) -> tuple[Array, FilmUnet, OptState]:
    images = batch["image"]
    labels = batch["label"]

    def grad_fn(film_unet: FilmUnet) -> Array:
        cond_emb = film_unet.embedder(images[0], labels[0])

        logits = jax.vmap(film_unet, in_axes=(0, None))(images, cond_emb)

        loss = jax.vmap(loss_fn)(logits, labels).mean()

        return loss

    loss, grads = eqx.filter_value_and_grad(grad_fn)(film_unet)

    updates, opt_state = opt.update(grads, opt_state, film_unet)  # type: ignore

    film_unet = eqx.apply_updates(film_unet, updates)

    return loss, film_unet, opt_state


def train(
    film_unet: FilmUnet,
    train_loader: MultiDataLoader,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    *,
    pbar: tqdm,
    epoch: int,
) -> tuple[FilmUnet, optax.OptState]:
    pbar.write("Training:\n")

    losses = []

    for batch_tensor in tqdm(train_loader, leave=False):
        batch: dict[str, Array] = jt.map(jnp.asarray, batch_tensor)

        loss, film_unet, opt_state = training_step(film_unet, opt, batch, opt_state)

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

    return film_unet, opt_state


def validate(film_unet: FilmUnet, val_loader: MultiDataLoader, *, pbar: tqdm, epoch: int):
    pbar.write("Validation:\n")

    for dataloader in val_loader.dataloaders:
        dataset_name: str = dataloader.dataset.name  # type: ignore

        batch = jt.map(jnp.asarray, next(iter(dataloader)))

        metrics = calc_metrics(film_unet, batch)

        pbar.write(f"Dataset: {dataset_name}:")
        pbar.write(f"    Dice score: {metrics['dice']:.3f}")
        pbar.write(f"    IoU score : {metrics['iou']:.3f}")
        pbar.write(f"    Hausdorff : {metrics['hausdorff']:.3f}")
        pbar.write("")

        if wandb.run is not None:
            wandb.run.log(
                {
                    f"{metric_name}/{dataset_name}": metric_value.item()
                    for metric_name, metric_value in metrics.items()
                },
                commit=False,
            )

    if wandb.run is not None:
        wandb.run.log({"epoch": epoch})

    pbar.write("")


def make_plots(film_unet: FilmUnet, val_loader: MultiDataLoader, test_loader: DataLoader):
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

        cond_emb = eqx.filter_jit(film_unet.embedder)(gen_image, gen_label)

        image = jnp.asarray(batch["image"][1])
        label = jnp.asarray(batch["label"][1])

        logits = eqx.filter_jit(film_unet)(image, cond_emb)
        pred = jnp.argmax(logits, axis=0)

        fig, axs = plt.subplots(ncols=3)

        axs[0].imshow(image.mean(axis=0), cmap="gray")
        axs[1].imshow(label, cmap="gray")
        axs[2].imshow(pred, cmap="gray")

        fig.savefig(image_folder / f"{dataset.name}.pdf")

        metrics = calc_metrics(film_unet, batch)

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

    metrics = calc_metrics(film_unet, batch)

    print(f"Dice score: {metrics['dice']:.3f}")
    print(f"IoU score : {metrics['iou']:.3f}")
    print(f"Hausdorff : {metrics['hausdorff']:.3f}")

    cond_emb = eqx.filter_jit(film_unet.embedder)(gen_image, gen_label)

    image = jnp.asarray(batch["image"][1])
    label = jnp.asarray(batch["label"][1])

    logits = eqx.filter_jit(film_unet)(image, cond_emb)
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


def make_umap(film_unet: FilmUnet, datasets: list[Dataset]):
    image_folder = Path(f"./images/{model_name}")

    assert image_folder.exists()

    embedder = film_unet.embedder

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
            "film_unet": {
                "base_channels": 16,
                "channel_mults": [1, 2, 4],
                "in_channels": 3,
                "out_channels": 2,
                "emb_size": 3 * 1024,
                "emb_kind": "${embedder}",
                "use_res": False,
                "use_weight_standardized_conv": False,
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

            film_unet = FilmUnet(**config.film_unet, key=jr.PRNGKey(config.seed))

        case "resume":
            assert args.artifact is not None

            loaded_config, weights_path = load_model_artifact(args.artifact)

            config = OmegaConf.merge(loaded_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            film_unet = FilmUnet(**config.film_unet, key=jr.PRNGKey(config.seed))

            film_unet = load_pytree(weights_path, film_unet)

        case cmd:
            raise RuntimeError(f"Unrecognised command {cmd}")

    print_config(OmegaConf.to_object(config))
    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            config=OmegaConf.to_object(config),  # type: ignore
            tags=[config.dataset, config.embedder, "film"],
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

    opt_state = opt.init(eqx.filter(film_unet, eqx.is_array_like))

    for epoch in (pbar := trange(config.epochs)):
        pbar.write(f"Epoch {epoch:02}\n")

        film_unet, opt_state = train(
            film_unet, train_loader, opt, opt_state, pbar=pbar, epoch=epoch
        )

        validate(film_unet, val_loader, pbar=pbar, epoch=epoch)

    model_path = Path(f"./models/{model_name}.safetensors")

    model_path.parent.mkdir(exist_ok=True)

    save_with_config_safetensors(model_path, OmegaConf.to_object(config), film_unet)

    if wandb.run is not None:
        model_artifact = wandb.Artifact(model_name, type="model")

        model_artifact.add_file(str(model_path.with_suffix(".json")))
        model_artifact.add_file(str(model_path.with_suffix(".safetensors")))

        wandb.run.log_artifact(model_artifact)

    print()
    print()

    make_plots(film_unet, val_loader, test_loader)
    make_umap(film_unet, trainsets + [testset])


if __name__ == "__main__":
    main()
