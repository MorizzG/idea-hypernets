from jaxtyping import Array, Float, Integer

import warnings
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import optax
from matplotlib import pyplot as plt
from optax import OptState
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange

from hyper_lap.datasets import Dataset, DegenerateDataset
from hyper_lap.datasets.preloaded import PreloadedDataset
from hyper_lap.hyper import HyperNet
from hyper_lap.metrics import dice_score
from hyper_lap.serialisation import save_with_config_safetensors
from hyper_lap.serialisation.safetensors import load_pytree
from hyper_lap.training.utils import (
    ResumeArgs,
    TrainArgs,
    load_amos_datasets,
    load_medidec_datasets,
    load_model_artifact,
    make_hypernet,
    parse_args,
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
    gen_image: Array,
    gen_label: Array,
) -> tuple[Array, HyperNet, OptState]:
    images = batch["image"]
    labels = batch["label"]

    images = images[:, 0:1]
    labels = (labels != 0).astype(jnp.int32)

    dynamic_hypernet, static_hypernet = eqx.partition(hypernet, eqx.is_array)

    def grad_fn(dynamic_hypernet: HyperNet) -> Array:
        hypernet = eqx.combine(dynamic_hypernet, static_hypernet)

        model = hypernet(gen_image, gen_label)

        logits = jax.vmap(model)(images)

        loss = jax.vmap(loss_fn)(logits, labels).sum()

        return loss

    loss, grads = eqx.filter_value_and_grad(grad_fn)(hypernet)

    updates, opt_state = opt.update(grads, opt_state, dynamic_hypernet)

    dynamic_hypernet = eqx.apply_updates(dynamic_hypernet, updates)

    hypernet = eqx.combine(dynamic_hypernet, static_hypernet)

    return loss, hypernet, opt_state


@eqx.filter_jit
def calc_dice_score(hypernet: HyperNet, batch: dict[str, Array]):
    gen_image = batch["image"][0][0:1]
    gen_label = batch["label"][0]

    model = eqx.filter_jit(hypernet)(gen_image, gen_label)

    images = batch["image"]
    labels = batch["label"]

    images = images[:, 0:1]
    labels = (labels == 1).astype(jnp.int32)

    logits = eqx.filter_jit(jax.vmap(model))(images)

    preds = jnp.argmax(logits, axis=1)

    dices = jax.jit(jax.vmap(dice_score))(preds, labels)

    return jnp.mean(dices)


def train(
    hypernet: HyperNet,
    train_loader: DataLoader,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    pbar: tqdm,
) -> tuple[HyperNet, optax.OptState]:
    pbar.write("Training:\n")

    losses = []

    for batch_tensor in tqdm(train_loader, leave=False):
        batch: dict[str, Array] = jt.map(jnp.asarray, batch_tensor)

        gen_image = batch["image"][0][0:1]
        gen_label = batch["label"][0]

        loss, hypernet, opt_state = training_step(
            hypernet, opt, batch, opt_state, gen_image, gen_label
        )

        losses.append(loss.item())

    mean_loss = jnp.mean(jnp.array(losses))

    pbar.write(f"Loss: {mean_loss:.3}")

    return hypernet, opt_state


def validate(hypernet: HyperNet, train_loader: DataLoader, pbar: tqdm):
    pbar.write("Validation:\n")

    batch = jt.map(jnp.asarray, next(iter(train_loader)))

    dice = calc_dice_score(hypernet, batch)

    pbar.write(f"Dice score: {dice:.3}")
    pbar.write("")


def make_plots(hypernet, train_loader: DataLoader):
    dataset = train_loader.dataset
    assert isinstance(dataset, Dataset)

    batch = jt.map(jnp.asarray, next(iter(train_loader)))

    gen_image = jnp.asarray(batch["image"][0][0:1])
    gen_label = jnp.asarray(batch["label"][0])

    model = eqx.filter_jit(hypernet)(gen_image, gen_label)

    image = jnp.asarray(batch["image"][1][0:1])
    label = jnp.asarray(batch["label"][1])

    logits = eqx.filter_jit(model)(image)
    pred = jnp.argmax(logits, axis=0)

    fig, axs = plt.subplots(ncols=3)

    axs[0].imshow(image[0], cmap="gray")
    axs[1].imshow(label, cmap="gray")
    axs[2].imshow(pred, cmap="gray")

    fig.savefig(f"images/{model_name}.pdf")

    dice_score = calc_dice_score(hypernet, batch)

    print(f"Dice score: {dice_score:.3f}")
    print(f"{logits.mean():.3} +/- {logits.std():.3}")

    print()
    print()


def main():
    global model_name

    args = parse_args()

    match args:
        case TrainArgs():
            config = {
                "seed": 42,
                "dataset": args.dataset,
                "unet": {
                    "base_channels": 8,
                    "channel_mults": [1, 2, 4],
                    "in_channels": 1,
                    "out_channels": 2,
                    "use_res": False,
                    "use_weight_standardized_conv": False,
                },
                "hypernet": {
                    "block_size": 8,
                    "emb_size": 512,
                    "kernel_size": 3,
                    "embedder_kind": args.embedder,
                },
            }

            hypernet = make_hypernet(config)
        case ResumeArgs(artifact=artifact_name):
            config, weights_path = load_model_artifact(artifact_name)

            hypernet = make_hypernet(config)

            load_pytree(weights_path, hypernet)
        case _:
            assert False

    del args

    model_name = Path(__file__).stem + "_" + config["embedder"]

    if config["dataset"] == "amos":
        dataset = load_amos_datasets(normalised=True)["liver"]
    elif config["dataset"] == "medidec":
        dataset = load_medidec_datasets(normalised=True)["Liver"]
    else:
        raise ValueError(f"Invalid dataset {config['dataset']}")

    if config["degenerate"]:
        print("Using degenerate dataset")

        dataset = DegenerateDataset(dataset)

        for X in dataset:
            assert jnp.all(X["image"] == dataset[0]["image"])
            assert jnp.all(X["label"] == dataset[0]["label"])
    else:
        dataset = PreloadedDataset(dataset)

    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=RandomSampler(dataset, num_samples=100 * config["batch_size"]),
        num_workers=config["num_workers"],
    )

    opt = optax.adamw(config["lr"])

    opt_state = opt.init(eqx.filter(hypernet, eqx.is_array_like))

    for epoch in (pbar := trange(config["epochs"])):
        pbar.write(f"Epoch {epoch:02}\n")

        hypernet, opt_state = train(hypernet, train_loader, opt, opt_state, pbar)

        validate(hypernet, train_loader, pbar)

    save_with_config_safetensors(f"models/{model_name}", config, hypernet)

    print()
    print()

    make_plots(hypernet, train_loader)


if __name__ == "__main__":
    main()
