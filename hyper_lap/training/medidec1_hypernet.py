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
from hyper_lap.hyper.hypernet import HyperNet
from hyper_lap.metrics import dice_score
from hyper_lap.training.utils import load_medidec_datasets, make_hypernet, parse_args, save_hypernet

warnings.simplefilter("ignore")


_key = jr.key(0)


def consume():
    global _key
    _key, _consume = jr.split(_key)
    return _consume


args = parse_args()


dataset = load_medidec_datasets()[0]

dataset = PreloadedDataset(dataset)

if args.degenerate:
    print("Using degenerate dataset")

    dataset = DegenerateDataset(dataset)

    for X in dataset:
        assert jnp.all(X["image"] == dataset[0]["image"])
        assert jnp.all(X["label"] == dataset[0]["label"])


num_workers = args.num_workers

print(f"Using {num_workers} workers")


train_loader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers
)


gen_image = jnp.asarray(dataset[0]["image"][0:1])
gen_label = jnp.asarray(dataset[0]["label"])


hyper_params = {
    "seed": 42,
    "unet": {
        "base_channels": 8,
        "channel_mults": [1, 2, 4],
        "in_channels": 1,
        "out_channels": 2,
        "weight_standardized_conv": True,
    },
    "hypernet": {"block_size": 8, "emb_size": 512, "embedder_kind": "clip"},
}

# unet_key, hypernet_key = jr.split(jr.PRNGKey(hyper_params["seed"]))

# model_template = Unet(**hyper_params["unet"], key=unet_key)
# hypernet = HyperNet(model_template, **hyper_params, key=hypernet_key)
model_template, hypernet = make_hypernet(hyper_params)


opt = optax.adamw(1e-5)
opt_state = opt.init(eqx.filter(hypernet, eqx.is_array_like))


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
    model = eqx.filter_jit(hypernet)(model_template, gen_image, gen_label)

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

        loss, hypernet, opt_state = training_step(hypernet, batch, opt_state, gen_image, gen_label)

        losses.append(loss.item())

    mean_loss = jnp.mean(jnp.array(losses))

    pbar.write(f"Loss: {mean_loss:.3}")

    batch = jt.map(jnp.asarray, next(iter(train_loader)))

    dice = calc_dice_score(hypernet, batch)

    pbar.write(f"Dice score: {dice:.3}")
    pbar.write("")

model = eqx.filter_jit(hypernet)(model_template, gen_image, gen_label)

image = jnp.asarray(dataset[0]["image"][0:1])
label = jnp.asarray(dataset[0]["label"])

logits = eqx.filter_jit(model)(image)
pred = jnp.argmax(logits, axis=0)

fig, axs = plt.subplots(ncols=3)

axs[0].imshow(image[0], cmap="gray")
axs[1].imshow(label, cmap="gray")
axs[2].imshow(pred, cmap="gray")


model_name = Path(__file__).stem

fig.savefig(f"images/{model_name}.png")

print(f"{logits.mean():.3} +/- {logits.std():.3}")

save_hypernet(f"models/{model_name}.eqx", hyper_params, hypernet)
