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
from umap import UMAP

from hyper_lap.datasets import Dataset, DegenerateDataset, MultiDataLoader
from hyper_lap.hyper import HyperNet, HyperNetConfig
from hyper_lap.metrics import dice_score
from hyper_lap.models import UnetConfig
from hyper_lap.serialisation import save_hypernet_safetensors
from hyper_lap.training.utils import HyperParams, load_amos_datasets, make_hypernet, parse_args

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
    labels = (labels == 1).astype(jnp.int32)

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


def validate(hypernet: HyperNet, train_loader: MultiDataLoader, pbar: tqdm):
    pbar.write("Validation:\n")

    for dataloader in train_loader.dataloaders:
        dataset: Dataset = dataloader.dataset  # type: ignore

        assert isinstance(dataset, Dataset)

        name = dataset.metadata.name

        pbar.write(f"Dataset: {name}")

        batch = jt.map(jnp.array, next(iter(dataloader)))

        dice = calc_dice_score(hypernet, batch)

        pbar.write(f"Dice score: {dice:.3}")
        pbar.write("")


def make_plots(hypernet, train_loader: MultiDataLoader, test_loader: DataLoader):
    for dataset, dataloader in zip(train_loader.datasets, train_loader.dataloaders):
        print(f"Dataset {dataset.name}")
        print()

        batch = jt.map(jnp.asarray, next(iter(dataloader)))

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

        fig.savefig(f"images/{model_name}_{dataset.name}.pdf")

        dice_score = calc_dice_score(hypernet, batch)

        print(f"Dice score: {dice_score:.3f}")
        print(f"{logits.mean():.3} +/- {logits.std():.3}")

        print()
        print()

    print()
    print()
    print(f"Test: {test_loader.dataset.name}")  # type: ignore
    print()

    batch = jt.map(jnp.asarray, next(iter(test_loader)))

    gen_image = jnp.asarray(batch["image"][0][0:1])
    gen_label = jnp.asarray(batch["label"][0])

    dice = calc_dice_score(hypernet, batch)

    print((f"Dice score: {dice:.3f}"))

    model = eqx.filter_jit(hypernet)(gen_image, gen_label)

    image = jnp.asarray(batch["image"][1][0:1])
    label = jnp.asarray(batch["label"][1])

    logits = eqx.filter_jit(model)(image)
    pred = jnp.argmax(logits, axis=0)

    fig, axs = plt.subplots(ncols=3)

    axs[0].imshow(image[0], cmap="gray")
    axs[1].imshow(label, cmap="gray")
    axs[2].imshow(pred, cmap="gray")

    fig.savefig(f"images/{model_name}_test.pdf")


def make_umap(hypernet: HyperNet, datasets: list[Dataset]):
    embedder = hypernet.input_embedder

    embedder = eqx.filter_jit(eqx.filter_vmap(embedder))

    multi_dataloader = MultiDataLoader(
        *datasets,
        num_samples=100,
        dataloader_args=dict(batch_size=100, num_workers=8),
    )

    samples = {
        dataset.metadata.name: jt.map(jnp.asarray, next(iter(dataloader)))
        for dataset, dataloader in zip(multi_dataloader.datasets, multi_dataloader.dataloaders)
    }

    embs = {name: embedder(X["image"], X["label"]) for name, X in samples.items()}

    umap = UMAP()
    umap.fit(jnp.concat([embs for embs in embs.values()]))

    projs: dict[str, Array] = {name: umap.transform(embs) for name, embs in embs.items()}  # type: ignore

    fig, ax = plt.subplots()

    for name, proj in projs.items():
        ax.scatter(proj[:, 0], proj[:, 1], 4.0, label=name.split(" ")[1])

    pos = ax.get_position()
    ax.set_position((pos.x0, pos.y0, pos.width * 0.75, pos.height))

    fig.legend(loc="outside center right")

    fig.savefig(f"./images/{model_name}_umap.pdf")


def main():
    global hypernet
    global model_name

    args = parse_args()

    model_name = Path(__file__).stem + "_" + args.embedder

    datasets = load_amos_datasets(normalised=True)

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
        *train_sets,
        num_samples=100 * args.batch_size,
        dataloader_args=dict(batch_size=args.batch_size, num_workers=args.num_workers),
    )
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=8)

    hyper_params = HyperParams(
        seed=42,
        unet=UnetConfig(
            base_channels=8,
            channel_mults=[1, 2, 4],
            in_channels=1,
            out_channels=2,
            use_res=False,
            use_weight_standardized_conv=False,
        ),
        hypernet=HyperNetConfig(
            block_size=8, emb_size=512, kernel_size=3, embedder_kind=args.embedder
        ),
    )

    hypernet = make_hypernet(hyper_params)

    opt = optax.adamw(1e-6)

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
                hypernet, opt, batch, opt_state, gen_image, gen_label
            )

            losses.append(loss.item())

        mean_loss = jnp.mean(jnp.array(losses))

        pbar.write(f"Loss: {mean_loss:.3}")

        for dataloader in train_loader.dataloaders:
            name: str = dataloader.dataset.name  # type: ignore

            batch = jt.map(jnp.asarray, next(iter(dataloader)))

            dice = calc_dice_score(hypernet, batch)

            pbar.write(f"Dice score {name: <15}: {dice:.3}")
        pbar.write("")

        # validate(hypernet, train_sets, pbar)

    save_hypernet_safetensors(f"models/{model_name}", hyper_params, hypernet)

    print()
    print()

    make_plots(hypernet, train_loader, test_loader)
    make_umap(hypernet, train_sets + [test_dataset])


if __name__ == "__main__":
    main()
