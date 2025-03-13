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
from omegaconf import MISSING, OmegaConf
from optax import OptState
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import wandb
from hyper_lap.datasets import Dataset, DegenerateDataset
from hyper_lap.metrics import dice_score, hausdorff_distance, jaccard_index
from hyper_lap.models import Unet
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


_key = jr.key(0)


def consume():
    global _key
    _key, _consume = jr.split(_key)
    return _consume


def calc_metrics(unet: Unet, batch: dict[str, Array]) -> dict[str, Array]:
    images = batch["image"]
    labels = batch["label"]

    logits = eqx.filter_jit(jax.vmap(unet))(images)

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
    unet: Unet,
    opt: optax.GradientTransformation,
    batch: dict[str, Array],
    opt_state: OptState,
) -> tuple[Array, Unet, OptState]:
    images = batch["image"]
    labels = batch["label"]

    def grad_fn(film_unet: Unet) -> Array:
        logits = jax.vmap(film_unet)(images)

        loss = jax.vmap(loss_fn)(logits, labels).mean()

        return loss

    loss, grads = eqx.filter_value_and_grad(grad_fn)(unet)

    updates, opt_state = opt.update(grads, opt_state, unet)  # type: ignore

    unet = eqx.apply_updates(unet, updates)

    return loss, unet, opt_state


def train(
    unet: Unet,
    train_loader: DataLoader,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    *,
    pbar: tqdm,
    epoch: int,
) -> tuple[Unet, optax.OptState]:
    pbar.write("Training:\n")

    losses = []

    for batch_tensor in tqdm(train_loader, leave=False):
        batch: dict[str, Array] = jt.map(jnp.asarray, batch_tensor)

        loss, unet, opt_state = training_step(unet, opt, batch, opt_state)

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

    return unet, opt_state


def validate(unet: Unet, val_loader: DataLoader, *, pbar: tqdm, epoch: int):
    pbar.write("Validation:\n")

    dataset_name: str = val_loader.dataset.name  # type: ignore

    batch = jt.map(jnp.asarray, next(iter(val_loader)))

    metrics = calc_metrics(unet, batch)

    pbar.write(f"Dataset: {dataset_name}:")
    pbar.write(f"    Dice score: {metrics['dice']:.3f}")
    pbar.write(f"    IoU score : {metrics['iou']:.3f}")
    pbar.write(f"    Hausdorff : {metrics['hausdorff']:.3f}")
    pbar.write("")

    if wandb.run is not None:
        wandb.run.log({"epoch": epoch, **metrics})

    pbar.write("")


def make_plots(unet: Unet, val_loader: DataLoader):
    image_folder = Path(f"./images/{model_name}")

    dataset = val_loader.dataset

    assert isinstance(dataset, Dataset)

    if image_folder.exists():
        shutil.rmtree(image_folder)

    image_folder.mkdir(parents=True)

    print(f"Dataset {dataset.name}")
    print()

    batch = jt.map(jnp.asarray, next(iter(val_loader)))

    image = jnp.asarray(batch["image"][1])
    label = jnp.asarray(batch["label"][1])

    logits = eqx.filter_jit(unet)(image)
    pred = jnp.argmax(logits, axis=0)

    fig, axs = plt.subplots(ncols=3)

    axs[0].imshow(image.mean(axis=0), cmap="gray")
    axs[1].imshow(label, cmap="gray")
    axs[2].imshow(pred, cmap="gray")

    fig.savefig(image_folder / f"{dataset.name}.pdf")

    metrics = calc_metrics(unet, batch)

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
                "base_channels": 32,
                "channel_mults": [1, 2, 4],
                "in_channels": 3,
                "out_channels": 2,
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

            unet = Unet(**config.unet, key=jr.PRNGKey(config.seed))

        case "resume":
            assert args.artifact is not None

            loaded_config, weights_path = load_model_artifact(args.artifact)

            config = OmegaConf.merge(loaded_config, arg_config)

            if missing_keys := OmegaConf.missing_keys(config):
                raise RuntimeError(f"Missing mandatory config options: {' '.join(missing_keys)}")

            unet = Unet(**config.unet, key=jr.PRNGKey(config.seed))

            unet = load_pytree(weights_path, unet)

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

    model_name = f"unet_{config.dataset}_{config.embedder}"

    match config.dataset:
        case "amos":
            trainsets = load_amos_datasets("train")
            valsets = load_amos_datasets("validation")

            trainset = trainsets["spleen"]
            valset = valsets["spleen"]
        case "medidec":
            trainsets = load_medidec_datasets("train")
            valsets = load_medidec_datasets("validation")

            trainset = trainsets["Liver"]
            valset = valsets["Liver"]
        case _:
            raise RuntimeError(f"Invalid dataset {config.dataset}")

    print(f"Trainset: {trainset.name}")

    if config.degenerate:
        print("Using degenerate dataset")

        trainset = DegenerateDataset(trainset)
        valset = trainset

    train_loader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        valset,
        batch_size=2 * config.batch_size,
        num_workers=args.num_workers,
    )

    opt = optax.adamw(config.lr)

    opt_state = opt.init(eqx.filter(unet, eqx.is_array_like))

    for epoch in (pbar := trange(config.epochs)):
        pbar.write(f"Epoch {epoch:02}\n")

        film_unet, opt_state = train(unet, train_loader, opt, opt_state, pbar=pbar, epoch=epoch)

        validate(film_unet, val_loader, pbar=pbar, epoch=epoch)

    model_path = Path(f"./models/{model_name}.safetensors")

    model_path.parent.mkdir(exist_ok=True)

    save_with_config_safetensors(model_path, OmegaConf.to_object(config), unet)

    if wandb.run is not None:
        model_artifact = wandb.Artifact("model-" + wandb.run.name, type="model")

        model_artifact.add_file(str(model_path.with_suffix(".json")))
        model_artifact.add_file(str(model_path.with_suffix(".safetensors")))

        wandb.run.log_artifact(model_artifact)

    print()
    print()

    make_plots(unet, train_loader)


if __name__ == "__main__":
    main()
