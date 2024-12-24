from jaxtyping import Array, Float, Integer

import multiprocessing
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

    dataloader: DataLoader


BATCH_SIZE = 32
EPOCHS = 50


num_workers = min(multiprocessing.cpu_count() // 2, 64)
# num_workers = 4
print(f"Using {num_workers} workers")


_key = jr.key(0)


def consume():
    global _key
    _key, _consume = jr.split(_key)
    return _consume


parser = ArgumentParser()
parser.add_argument("--degenerate", action="store_true", help="Use degenerate dataset")

args = parser.parse_args()

degenerate = args.degenerate


# root_dir = "/vol/ideadata/eg94ifeh/idea-laplacian-hypernet/datasets/MediDecSliced"
root_dir = Path("/media/LinuxData/datasets/MediDecSliced")

dataset_names = [
    "01_BrainTumour",
    "02_Heart",
    "03_Liver",
    # "04_Hippocampus",
    # "05_Prostate",
    "06_Lung",
    "07_Pancreas",
    "08_HepaticVessel",
    "09_Spleen",
    "10_Colon",
]


def load_dataset(dataset_name: str) -> Dataset:
    dataset = MediDecSliced(root_dir / dataset_name)

    if degenerate:
        dataset = DegenerateDataset(dataset)

    # dataset = PreloadedDataset(dataset)

    gen_image = jnp.asarray(dataset[0]["image"][0:1])
    gen_label = jnp.asarray(dataset[0]["label"])

    shape_x, shape_y = dataset[0]["image"].shape[-2:]

    if shape_x * shape_y >= 512 * 512:
        batch_size = 16
    elif shape_x * shape_y >= 128 * 128:
        batch_size = 32
    else:
        batch_size = 64

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return Dataset(
        name=dataset_name,
        dataset=dataset,
        gen_image=gen_image,
        gen_label=gen_label,
        dataloader=dataloader,
    )


datasets: list[Dataset] = []

for dataset_name in dataset_names[:-1]:
    datasets.append(load_dataset(dataset_name))


test_dataset = load_dataset(dataset_names[-1])

model_template = Unet(8, [1, 2, 4], in_channels=1, out_channels=2, key=consume())
hypernet = HyperNet(model_template, 8, emb_size=64, key=consume())


opt = optax.adamw(1e-4)

opt_state = opt.init(eqx.filter(hypernet, eqx.is_array_like))


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
def calc_dice_score(hypernet: HyperNet, batch: dict[str, Array]):
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

    for dataset in datasets:
        pbar.write(f"Dataset: {dataset.name}")

        train_loader = dataset.dataloader

        batch = jt.map(jnp.array, next(iter(train_loader)))

        dice = calc_dice_score(hypernet, batch)

        pbar.write(f"Dice score: {dice:.3}")
        pbar.write("")


for epoch in (pbar := trange(EPOCHS)):
    pbar.write(f"Epoch {epoch:02}\n")

    pbar.write("Training:\n")

    dataset_idx = jr.randint(consume(), (), 0, len(datasets))

    dataset = datasets[dataset_idx]

    pbar.write(f"Dataset: {dataset.name}")
    pbar.write("")

    train_loader = dataset.dataloader
    gen_image = dataset.gen_image
    gen_label = dataset.gen_label

    losses = []

    for batch_tensor in tqdm(train_loader, leave=False):
        batch: dict[str, Array] = jt.map(jnp.array, batch_tensor)

        loss, hypernet, opt_state = training_step(hypernet, batch, opt_state, gen_image, gen_label)

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

batch = jt.map(jnp.asarray, next(iter(test_dataset.dataloader)))

dice = calc_dice_score(hypernet, batch)

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

fig.show()
plt.show()
