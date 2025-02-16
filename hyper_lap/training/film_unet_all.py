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
from matplotlib import pyplot as plt
from optax import OptState
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from umap import UMAP

import wandb
from hyper_lap.datasets import Dataset, DegenerateDataset, MultiDataLoader
from hyper_lap.metrics import dice_score, hausdorff_distance, jaccard_index
from hyper_lap.models import FilmUnet
from hyper_lap.serialisation.safetensors import save_pytree
from hyper_lap.training.utils import load_amos_datasets, load_medidec_datasets, parse_args, to_PIL

warnings.simplefilter("ignore")


_key = jr.key(0)


def consume():
    global _key
    _key, _consume = jr.split(_key)
    return _consume


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


def validate(film_unet: FilmUnet, train_loader: MultiDataLoader, *, pbar: tqdm, epoch: int):
    pbar.write("Validation:\n")

    all_metrics: dict[str, dict[str, Array]] = {}

    for dataloader in train_loader.dataloaders:
        dataset_name: str = dataloader.dataset.name  # type: ignore

        batch = jt.map(jnp.asarray, next(iter(dataloader)))

        metrics = calc_metrics(film_unet, batch)

        pbar.write(f"Dataset: {dataset_name}")
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


def make_plots(film_unet: FilmUnet, train_loader: MultiDataLoader, test_loader: DataLoader):
    image_folder = Path(f"./images/{model_name}")

    if image_folder.exists():
        shutil.rmtree(image_folder)

    image_folder.mkdir(parents=True)

    for dataset, dataloader in zip(train_loader.datasets, train_loader.dataloaders):
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

        print(f"Dice score: {metrics['dice']:.3f}")
        print(f"IoU       : {metrics['iou']:.3f}")
        print(f"Hausdorff : {metrics['hausdorff']:.3f}")
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

    metrics = calc_metrics(film_unet, batch)

    print(f"Dice score: {metrics['dice']:.3f}")
    print(f"IoU       : {metrics['iou']:.3f}")
    print(f"Hausdorff : {metrics['hausdorff']:.3f}")
    print()

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

    fig.savefig(image_folder / f"/{model_name}_umap.pdf")

    if wandb.run is not None:
        # fig.canvas.draw()

        # image_flat = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")  # type: ignore
        # # NOTE: reversed converts (W, H) from get_width_height to (H, W)
        # # convert from (H * W * 4,) to (H, W, 4)
        # image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 4)

        image = wandb.Image(fig, mode="RGBA", caption="UMAP")

        wandb.run.log({"images/umap": image})


def main():
    global model_name

    args = parse_args()

    config = {
        "seed": 42,
        "dataset": args.dataset,
        "embedder": args.embedder,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "film_unet": {
            "base_channels": 16,
            "channel_mults": [1, 2, 4],
            "in_channels": 3,
            "out_channels": 2,
            "emb_size": 3 * 1024,
            "emb_kind": "clip",
            "use_res": False,
            "use_weight_standardized_conv": False,
        },
    }

    model_name = f"film_unet_all_{config['dataset']}_{config['embedder']}"

    if args.wandb:
        wandb.init(
            project="idea-laplacian-hypernet",
            config=config,
            tags=[config["dataset"], config["embedder"]],
            # sync_tensorboard=True,
        )

    if args.dataset == "amos":
        datasets = load_amos_datasets(normalised=True)

        testset = datasets.pop("liver")

        datasets = {
            name: dataset
            for name, dataset in datasets.items()
            if name in {"spleen", "pancreas"}  #
        }

        trainsets = list(datasets.values())
    elif args.dataset == "medidec":
        datasets = load_medidec_datasets(normalised=True)

        # only use CT datasets
        datasets = {
            name: dataset
            for name, dataset in datasets.items()
            if dataset.metadata.modality["0"] == "CT"
        }

        # CT datasets: Liver, Lung, Pancreas, HepaticVessel, Spleen, Colon

        testset = datasets.pop("Spleen")

        datasets = {
            name: dataset
            for name, dataset in datasets.items()
            if name in {"Liver", "Pancreas", "Lung"}
        }

        trainsets = list(datasets.values())
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")

    print(f"Trainsets: {', '.join([trainset.name for trainset in trainsets])}")
    print(f"Testset:   {testset.name}")

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
        dataloader_args=dict(batch_size=config["batch_size"], num_workers=args.num_workers),
    )

    # use 2 * batch_size for test loader since we need no grad here
    test_loader = DataLoader(testset, batch_size=2 * args.batch_size, num_workers=8)

    film_unet = FilmUnet(**config["film_unet"], key=jr.PRNGKey(config["seed"]))

    opt = optax.adamw(config["learning_rate"])

    opt_state = opt.init(eqx.filter(film_unet, eqx.is_array_like))

    print()
    print()

    for epoch in (pbar := trange(config["epochs"])):
        pbar.write(f"Epoch {epoch:02}\n")

        film_unet, opt_state = train(
            film_unet, train_loader, opt, opt_state, pbar=pbar, epoch=epoch
        )

        validate(film_unet, train_loader, pbar=pbar, epoch=epoch)

    model_path = Path(f"./models/{model_name}")

    model_path.parent.mkdir(exist_ok=True)

    save_pytree(model_path, film_unet)

    if wandb.run is not None:
        model_artifact = wandb.Artifact("model-" + wandb.run.name, type="model")

        model_artifact.add_file(str(model_path.with_suffix(".json")))
        model_artifact.add_file(str(model_path.with_suffix(".safetensors")))

        wandb.run.log_artifact(model_artifact)

    print()
    print()

    make_plots(film_unet, train_loader, test_loader)
    # make_umap(film_unet, trainsets + [testset])


if __name__ == "__main__":
    main()
