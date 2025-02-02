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

from hyper_lap.datasets import DegenerateDataset, PreloadedDataset
from hyper_lap.metrics import dice_score
from hyper_lap.models import Unet
from hyper_lap.training.utils import load_amos_datasets, parse_args

warnings.simplefilter("ignore")


_key = jr.key(0)


def consume():
    global _key
    _key, _consume = jr.split(_key)
    return _consume


model_name = Path(__file__).stem


args = parse_args()


dataset = load_amos_datasets()[0]


if args.degenerate:
    print("Using degenerate dataset")

    dataset = DegenerateDataset(dataset)

    for X in dataset:
        assert jnp.all(X["image"] == dataset[0]["image"])
        assert jnp.all(X["label"] == dataset[0]["label"])
else:
    dataset = PreloadedDataset(dataset)


num_workers = args.num_workers

print(f"Using {num_workers} workers")


train_loader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers
)


model = Unet(
    8, [1, 2, 4], in_channels=1, out_channels=2, use_weight_standardized_conv=True, key=consume()
)

if args.degenerate:
    opt = optax.adamw(1e-4)
else:
    opt = optax.adamw(1e-3)

opt_state = opt.init(eqx.filter(model, eqx.is_array))


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
    model: Unet, images: Array, labels: Array, opt_state: OptState
) -> tuple[Array, Unet, OptState]:
    dynamic_model, static_model = eqx.partition(model, eqx.is_array)

    def grad_fn(dynamic_model: Unet) -> Array:
        model = eqx.combine(dynamic_model, static_model)

        logits = jax.vmap(model)(images)

        loss = jax.vmap(loss_fn)(logits, labels).sum()

        return loss

    loss, grads = eqx.filter_value_and_grad(grad_fn)(dynamic_model)

    updates, opt_state = opt.update(grads, opt_state, dynamic_model)

    dynamic_model = eqx.apply_updates(dynamic_model, updates)

    model = eqx.combine(dynamic_model, static_model)

    return loss, model, opt_state


@eqx.filter_jit
def calc_dice_score(model: Unet, batch: dict[str, Array]):
    images = batch["image"]
    labels = batch["label"]

    images = images[:, 0:1]
    labels = (labels == 1).astype(jnp.int32)

    logits = eqx.filter_jit(jax.vmap(model))(images)

    preds = jnp.argmax(logits, axis=1)

    dices = jax.jit(jax.vmap(dice_score))(preds, labels)

    return jnp.mean(dices)


for epoch in (pbar := trange(args.epochs)):
    pbar.write(f"Epoch {epoch:02}\n")

    losses = []

    for batch_tensor in tqdm(train_loader, leave=False):
        batch: dict[str, Array] = jt.map(jnp.asarray, batch_tensor)

        images = batch["image"]
        labels = batch["label"]

        images = images[:, 0:1]
        labels = (labels == 1).astype(jnp.int32)

        loss, model, opt_state = training_step(model, images, labels, opt_state)

        losses.append(loss.item())

        # inner_pbar.update(BATCH_SIZE)

    mean_loss = jnp.mean(jnp.array(losses))

    pbar.write(f"Loss: {mean_loss:.3}")

    batch = jt.map(jnp.asarray, next(iter(train_loader)))

    dice = calc_dice_score(model, batch)

    pbar.write(f"Dice score: {dice:.3}")
    pbar.write("")


image = jnp.asarray(dataset[0]["image"][0:1])
label = jnp.asarray(dataset[0]["label"])

logits = eqx.filter_jit(model)(image)
pred = jnp.argmax(logits, axis=0)

fig, axs = plt.subplots(ncols=3)

axs[0].imshow(image[0], cmap="gray")
axs[1].imshow(label, cmap="gray")
axs[2].imshow(pred, cmap="gray")

fig.savefig("images/figure.png")
