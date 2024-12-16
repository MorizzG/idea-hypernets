from jaxtyping import Array, Float, Integer

import multiprocessing
import warnings
from argparse import ArgumentParser

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

from hyper_lap.datasets import DegenerateDataset, LidcIdri
from hyper_lap.hyper.hypernet import HyperNet
from hyper_lap.metrics import dice_score
from hyper_lap.models import Unet

warnings.simplefilter("ignore")

BATCH_SIZE = 64
EPOCHS = 50

_key = jr.key(0)


def consume():
    global _key
    _key, _consume = jr.split(_key)
    return _consume


parser = ArgumentParser()
parser.add_argument("--degenerate", action="store_true", help="Use degenerate dataset")

args = parser.parse_args()

degenerate = args.degenerate


dataset = LidcIdri("/media/LinuxData/datasets/LIDC-IDRI-slices")

if degenerate:
    dataset = DegenerateDataset(dataset)

gen_image = jnp.asarray(dataset[0]["image"])
gen_label = jnp.asarray(dataset[0]["masks"][0])

num_workers = min(multiprocessing.cpu_count() // 2, 64)
print(f"Using {num_workers} workers")

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)


model_template = Unet(8, [1, 2, 4], in_channels=1, out_channels=2, key=consume())
hypernet = HyperNet(model_template, 8, emb_size=256, key=consume())


opt = optax.adamw(5e-4)

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
    images: Array,
    labels: Array,
    opt_state: OptState,
    image: Array,
    label: Array,
) -> tuple[Array, HyperNet, OptState]:
    dynamic_hypernet, static_hypernet = eqx.partition(hypernet, eqx.is_array)

    def grad_fn(dynamic_hypernet: HyperNet) -> Array:
        hypernet = eqx.combine(dynamic_hypernet, static_hypernet)

        model = hypernet(model_template, image, label)

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
    labels = batch["masks"][0]

    images = images[:, 0:1]
    labels = (labels == 1).astype(jnp.int32)

    logits = eqx.filter_jit(jax.vmap(model))(images)

    preds = jnp.argmax(logits, axis=1)

    dices = jax.jit(jax.vmap(dice_score))(preds, labels)

    return jnp.mean(dices)


for epoch in (pbar := trange(EPOCHS)):
    losses = []

    for batch_tensor in tqdm(train_loader, leave=False):
        batch: dict[str, Array] = jt.map(jnp.asarray, batch_tensor)

        images = batch["image"]
        labels = batch["masks"][0]

        # image = image[:, 0:1]
        labels = (labels == 1).astype(jnp.int32)

        loss, hypernet, opt_state = training_step(
            hypernet, images, labels, opt_state, gen_image, gen_label
        )

        losses.append(loss.item())

        # inner_pbar.update(BATCH_SIZE)

    mean_loss = jnp.mean(jnp.array(losses))

    pbar.write(f"Loss: {mean_loss:.3}")

    batch = jt.map(jnp.asarray, next(iter(train_loader)))

    dice = calc_dice_score(hypernet, batch)

    pbar.write(f"Dice score: {dice:.3}")
    pbar.write("")


model = hypernet(model_template, gen_image, gen_label)

image = jnp.asarray(dataset[0]["image"])
label = jnp.asarray(dataset[0]["masks"][0])

logits = eqx.filter_jit(model)(image)
pred = jnp.argmax(logits, axis=0)

fig, axs = plt.subplots(ncols=3)

axs[0].imshow(image[0], cmap="gray")
axs[1].imshow(label, cmap="gray")
axs[2].imshow(pred, cmap="gray")

fig.show()
plt.show()
