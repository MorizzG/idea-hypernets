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
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange

from hyper_lap.datasets import Dataset, DegenerateDataset, PreloadedDataset
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

warnings.simplefilter("ignore")


_key = jr.key(0)


def consume():
    global _key
    _key, _consume = jr.split(_key)
    return _consume


def calc_metrics(hypernet: HyperNet, batch: dict[str, Array]) -> dict[str, Array]:
    images = batch["image"]
    labels = batch["label"]

    gen_image = images[0]
    gen_label = labels[0]

    model = eqx.filter_jit(hypernet)(gen_image, gen_label)

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
    train_loader: DataLoader,
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


def validate(hypernet: HyperNet, train_loader: DataLoader, *, pbar: tqdm, epoch: int):
    pbar.write("Validation:")
    pbar.write("")

    batch = jt.map(jnp.asarray, next(iter(train_loader)))

    metrics = calc_metrics(hypernet, batch)

    pbar.write(f"Dice score: {metrics['dice']:.3}")
    pbar.write(f"IoU score : {metrics['iou']:.3}")
    pbar.write(f"Hausdorff : {metrics['hausdorff']:.3}")
    pbar.write("")

    if wandb.run is not None:
        wandb.run.log(
            {
                "epoch": epoch,
            }
            | metrics
        )
    pbar.write("")


def make_plots(hypernet, train_loader: DataLoader):
    image_folder = Path(f"./images/{model_name}")

    if image_folder.exists():
        shutil.rmtree(image_folder)

    image_folder.mkdir(parents=True)

    batch = jt.map(jnp.asarray, next(iter(train_loader)))

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

    assert isinstance(train_loader.dataset, Dataset)

    fig.savefig(image_folder / f"{train_loader.dataset.name}.pdf")

    metrics = calc_metrics(hypernet, batch)

    print(f"Dice score: {metrics['dice']:.3}")
    print(f"IoU score : {metrics['iou']:.3}")
    print(f"Hausdorff : {metrics['hausdorff']:.3}")

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

        wandb.run.log({"images/train": image})


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
                "base_channels": 16,
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
            dataset = load_amos_datasets(normalised=True)["liver"]
        case "medidec":
            dataset = load_medidec_datasets(normalised=True)["Liver"]
        case _:
            raise RuntimeError(f"Invalid dataset {config.dataset}")

    print(f"Trainset: {dataset.name}")

    if config.degenerate:
        print("Using degenerate dataset")

        dataset = DegenerateDataset(dataset)

        for X in dataset:
            # assert jnp.all(X["image"] == dataset[0]["image"])
            # assert jnp.all(X["label"] == dataset[0]["label"])

            assert eqx.tree_equal(X, dataset[0])
    else:
        dataset = PreloadedDataset(dataset)

    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=RandomSampler(dataset, num_samples=100 * config.batch_size),
        num_workers=args.num_workers,
    )

    opt = optax.adamw(config.lr)

    opt_state = opt.init(eqx.filter(hypernet, eqx.is_array_like))

    for epoch in (pbar := trange(config.epochs)):
        pbar.write(f"Epoch {epoch:02}\n")

        hypernet, opt_state = train(hypernet, train_loader, opt, opt_state, pbar=pbar, epoch=epoch)

        validate(hypernet, train_loader, pbar=pbar, epoch=epoch)

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

    make_plots(hypernet, train_loader)


if __name__ == "__main__":
    main()
