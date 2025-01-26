from jaxtyping import Array, Float, Integer

import warnings
from argparse import ArgumentParser
from dataclasses import dataclass
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

from hyper_lap.datasets import DegenerateDataset, MediDecSliced
from hyper_lap.datasets.multi_dataloader import MultiDataLoader
from hyper_lap.hyper.hypernet import HyperNet
from hyper_lap.metrics import dice_score
from hyper_lap.models import Unet

warnings.simplefilter("ignore")


@dataclass
class Dataset:
    name: str

    dataset: MediDecSliced | DegenerateDataset

    gen_image: Array
    gen_label: Array

    # dataloader: DataLoader


DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
NUM_WORKERS = 8


_key = jr.key(0)


def consume():
    global _key
    _key, _consume = jr.split(_key)
    return _consume


parser = ArgumentParser()

parser.add_argument("--degenerate", action="store_true", help="Use degenerate dataset")
parser.add_argument(
    "--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs to train for"
)
parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
parser.add_argument(
    "--num-workers", type=int, default=NUM_WORKERS, help="Number of dataloader worker threads"
)

args = parser.parse_args()

degenerate = args.degenerate

epochs = args.epochs
batch_size = args.batch_size
num_workers = args.num_workers


# num_workers = min(multiprocessing.cpu_count() // 2, 64)
# num_workers = 4

print(f"Using {num_workers} workers")


root_dir = Path("/vol/ideadata/eg94ifeh/idea-laplacian-hypernet/datasets/MediDecSliced")

if not root_dir.exists():
    root_dir = Path("/media/LinuxData/datasets/MediDecSliced")

if not root_dir.exists():
    raise RuntimeError("Could not determine root_dir")

assert root_dir.exists()

dataset_names = [
    "01_BrainTumour",
    "02_Heart",
    "03_Liver",
    # "04_Hippocampus",  # hippocampus is small and has weird shapes, so we exclude it
    "05_Prostate",
    "06_Lung",
    "07_Pancreas",
    "08_HepaticVessel",
    "09_Spleen",
    "10_Colon",
]


def load_dataset(dataset_name: str) -> Dataset:
    dataset = MediDecSliced(root_dir / dataset_name, split="train")

    if degenerate:
        dataset = DegenerateDataset(dataset)

    # dataset = PreloadedDataset(dataset)

    gen_image = jnp.asarray(dataset[0]["image"][0:1])
    gen_label = jnp.asarray(dataset[0]["label"])

    # shape_x, shape_y = dataset[0]["image"].shape[-2:]

    # if shape_x * shape_y >= 512 * 512:
    #     batch_size = 16
    # elif shape_x * shape_y >= 128 * 128:
    #     batch_size = 32
    # else:
    #     batch_size = 64

    # batch_size = BATCH_SIZE

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return Dataset(
        name=dataset_name,
        dataset=dataset,
        gen_image=gen_image,
        gen_label=gen_label,
    )


# datasets: list[Dataset] = []

# for dataset_name in dataset_names[:-1]:
#     datasets.append(load_dataset(dataset_name))

datasets = [load_dataset(dataset_name) for dataset_name in dataset_names[:-1]]

test_dataset = load_dataset(dataset_names[-1])

test_loader = DataLoader(test_dataset.dataset, batch_size=128, num_workers=32)

multi_dataloader = MultiDataLoader(
    *[dataset.dataset for dataset in datasets],
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)

model_template = Unet(8, [1, 2, 4], in_channels=1, out_channels=2, key=consume())
hypernet = HyperNet(model_template, 8, emb_size=64, key=consume())


opt = optax.adamw(1e-4)


@jax.jit
def loss_fn(logits: Float[Array, "c h w"], labels: Integer[Array, "h w"]) -> Array:
    # C H W -> H W C
    logits = jnp.moveaxis(logits, 0, -1)

    # b c h w
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
def calc_dice_score(
    hypernet: HyperNet, batch: dict[str, Array], gen_image: Array, gen_label: Array
):
    model = eqx.filter_jit(hypernet)(model_template, gen_image, gen_label)

    images = batch["image"]
    labels = batch["label"]

    images = images[:, 0:1]
    labels = (labels == 1).astype(jnp.int32)

    logits = eqx.filter_jit(jax.vmap(model))(images)

    preds = jnp.argmax(logits, axis=1)

    dices = jax.jit(jax.vmap(dice_score))(preds, labels)

    return jnp.mean(dices)


def validate(hypernet: HyperNet, datasets: list[Dataset], pbar: tqdm):
    pbar.write("Validation:\n")

    for dataloader in multi_dataloader.dataloaders:
        dataset: MediDecSliced | DegenerateDataset = dataloader.dataset  # type: ignore

        assert isinstance(dataset, (MediDecSliced, DegenerateDataset))

        if isinstance(dataset, MediDecSliced):
            name = dataset.metadata.name
        elif isinstance(dataset, DegenerateDataset):
            assert isinstance(dataset.orig_dataset, MediDecSliced)

            name = dataset.orig_dataset.metadata.name
        else:
            assert False

        pbar.write(f"Dataset: {name}")

        batch = jt.map(jnp.array, next(iter(dataloader)))

        gen_image = batch["image"][0][0:1]
        gen_label = batch["label"][0]

        dice = calc_dice_score(hypernet, batch, gen_image, gen_label)

        pbar.write(f"Dice score: {dice:.3}")
        pbar.write("")


def main():
    global hypernet

    opt_state = opt.init(eqx.filter(hypernet, eqx.is_array_like))

    for epoch in (pbar := trange(epochs)):
        pbar.write(f"Epoch {epoch:02}\n")

        pbar.write("Training:\n")

        losses = []

        for dataset_idx, batch_tensor in tqdm(multi_dataloader, leave=False):  # type: ignore
            batch: dict[str, Array] = jt.map(jnp.array, batch_tensor)

            gen_image = datasets[dataset_idx].gen_image
            gen_label = datasets[dataset_idx].gen_label

            loss, hypernet, opt_state = training_step(
                hypernet, batch, opt_state, gen_image, gen_label
            )

            losses.append(loss.item())

        mean_loss = jnp.mean(jnp.array(losses))

        pbar.write(f"Loss: {mean_loss:.3}")

        validate(hypernet, datasets, pbar)

        # batch = jt.map(jnp.asarray, next(iter(train_loader)))

        # dice = calc_dice_score(hypernet, batch)

        # pbar.write(f"Dice score: {dice:.3}")
        # pbar.write("")

    print()
    print()
    print("Test:")
    print()

    gen_image = test_dataset.gen_image
    gen_label = test_dataset.gen_label

    batch = jt.map(jnp.asarray, next(iter(test_loader)))

    dice = calc_dice_score(hypernet, batch, gen_image, gen_label)

    print((f"Dice score: {dice:.3}"))

    model = eqx.filter_jit(hypernet)(model_template, gen_image, gen_label)

    image = jnp.asarray(test_dataset.dataset[0]["image"][0:1])
    label = jnp.asarray(test_dataset.dataset[0]["label"])

    logits = eqx.filter_jit(model)(image)
    pred = jnp.argmax(logits, axis=0)

    fig, axs = plt.subplots(ncols=3)

    axs[0].imshow(image[0], cmap="gray")
    axs[1].imshow(label, cmap="gray")
    axs[2].imshow(pred, cmap="gray")

    fig.savefig("images/figure.png")


if __name__ == "__main__":
    main()
