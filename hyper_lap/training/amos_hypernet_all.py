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
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from hyper_lap.datasets import AmosSliced, DegenerateDataset, MultiDataLoader
from hyper_lap.hyper.hypernet import HyperNet
from hyper_lap.metrics import dice_score
from hyper_lap.models import Unet
from hyper_lap.training.utils import load_amos_datasets, make_hypernet, parse_args, save_hypernet

warnings.simplefilter("ignore")


_key = jr.key(0)


def consume():
    global _key
    _key, _consume = jr.split(_key)
    return _consume


model_name = Path(__file__).stem


args = parse_args()


datasets = load_amos_datasets()

if args.degenerate:
    print("Using degenerate dataset")

    datasets = [DegenerateDataset(dataset) for dataset in datasets]

    for dataset in datasets:
        for X in dataset:
            assert jnp.all(X["image"] == dataset[0]["image"])
            assert jnp.all(X["label"] == dataset[0]["label"])

train_sets = datasets[:-1]
test_dataset = datasets[-1]

train_loader = MultiDataLoader(
    *train_sets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
)
test_loader = DataLoader(test_dataset, batch_size=128, num_workers=8)


hyper_params = {
    "seed": 42,
    "unet": {
        "base_channels": 8,
        "channel_mults": [1, 2, 4],
        "in_channels": 1,
        "out_channels": 2,
        "use_weight_standardized_conv": True,
    },
    "hypernet": {"block_size": 8, "emb_size": 512, "embedder_kind": args.embedder},
}

model_template, hypernet = make_hypernet(hyper_params)


opt = optax.adamw(1e-5)


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
    batch: dict[str, Array],
    opt_state: OptState,
    gen_image: Array,
    gen_label: Array,
) -> tuple[Array, HyperNet, OptState]:
    images = batch["image"]
    labels = batch["label"]

    images = images[:, 0:1]
    labels = (labels == 1).astype(jnp.int32)

    dynamic_hypernet, static_hypernet = eqx.partition(hypernet, eqx.is_array)

    def grad_fn(dynamic_hypernet: HyperNet) -> Array:
        hypernet = eqx.combine(dynamic_hypernet, static_hypernet)

        model = hypernet(model_template, gen_image, gen_label)

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

    model = eqx.filter_jit(hypernet)(model_template, gen_image, gen_label)

    images = batch["image"]
    labels = batch["label"]

    images = images[:, 0:1]
    labels = (labels == 1).astype(jnp.int32)

    logits = eqx.filter_jit(jax.vmap(model))(images)

    preds = jnp.argmax(logits, axis=1)

    dices = jax.jit(jax.vmap(dice_score))(preds, labels)

    return jnp.mean(dices)


def validate(hypernet: HyperNet, datasets: list[AmosSliced] | list[DegenerateDataset], pbar: tqdm):
    pbar.write("Validation:\n")

    for dataloader in train_loader.dataloaders:
        dataset: AmosSliced | DegenerateDataset = dataloader.dataset  # type: ignore

        assert isinstance(dataset, (AmosSliced, DegenerateDataset))

        if isinstance(dataset, AmosSliced):
            name = dataset.metadata.name
        elif isinstance(dataset, DegenerateDataset):
            assert isinstance(dataset.orig_dataset, AmosSliced)

            name = dataset.orig_dataset.metadata.name
        else:
            assert False

        pbar.write(f"Dataset: {name}")

        batch = jt.map(jnp.array, next(iter(dataloader)))

        dice = calc_dice_score(hypernet, batch)

        pbar.write(f"Dice score: {dice:.3}")
        pbar.write("")


def main():
    global hypernet

    opt_state = opt.init(eqx.filter(hypernet, eqx.is_array_like))

    for epoch in (pbar := trange(args.epochs)):
        pbar.write(f"Epoch {epoch:02}\n")

        pbar.write("Training:\n")

        losses = []

        for batch_tensor in tqdm(train_loader, leave=False):
            batch: dict[str, Array] = jt.map(jnp.asarray, batch_tensor)

            gen_image = batch["image"][0][0:1]
            gen_label = batch["label"][0]

            loss, hypernet, opt_state = training_step(
                hypernet, batch, opt_state, gen_image, gen_label
            )

            losses.append(loss.item())

        mean_loss = jnp.mean(jnp.array(losses))

        pbar.write(f"Loss: {mean_loss:.3}")

        for dataloader in train_loader.dataloaders:
            batch = jt.map(jnp.asarray, next(iter(dataloader)))

            dice = calc_dice_score(hypernet, batch)

            pbar.write(f"Dice score: {dice:.3}")
        pbar.write("")

        # validate(hypernet, train_sets, pbar)

    save_hypernet(f"models/{model_name}.eqx", hyper_params, hypernet)

    for i, dataloader in enumerate(train_loader.dataloaders):
        print(f"Dataset {i}:")

        batch = jt.map(jnp.asarray, next(iter(dataloader)))

        gen_image = jnp.asarray(batch["image"][0][0:1])
        gen_label = jnp.asarray(batch["label"][0])

        model = eqx.filter_jit(hypernet)(model_template, gen_image, gen_label)

        image = jnp.asarray(batch["image"][1][0:1])
        label = jnp.asarray(batch["label"][1])

        logits = eqx.filter_jit(model)(image)
        pred = jnp.argmax(logits, axis=0)

        fig, axs = plt.subplots(ncols=3)

        axs[0].imshow(image[0], cmap="gray")
        axs[1].imshow(label, cmap="gray")
        axs[2].imshow(pred, cmap="gray")

        fig.savefig(f"images/{model_name}_{i:02}.png")

        dice_score = calc_dice_score(hypernet, batch)

        print(f"Dice score: {dice_score}")
        print(f"{logits.mean():.3} +/- {logits.std():.3}")

        print()

    print()
    print()
    print("Test:")
    print()

    batch = jt.map(jnp.asarray, next(iter(test_loader)))

    gen_image = jnp.asarray(batch["image"][0][0:1])
    gen_label = jnp.asarray(batch["label"][0])

    dice = calc_dice_score(hypernet, batch)

    print((f"Dice score: {dice:.3}"))

    model = eqx.filter_jit(hypernet)(model_template, gen_image, gen_label)

    image = jnp.asarray(test_dataset[1]["image"][0:1])
    label = jnp.asarray(test_dataset[1]["label"])

    logits = eqx.filter_jit(model)(image)
    pred = jnp.argmax(logits, axis=0)

    fig, axs = plt.subplots(ncols=3)

    axs[0].imshow(image[0], cmap="gray")
    axs[1].imshow(label, cmap="gray")
    axs[2].imshow(pred, cmap="gray")

    fig.savefig("images/figure.png")


if __name__ == "__main__":
    main()
