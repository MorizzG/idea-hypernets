import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array
import optax
from optax import OptState
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from hyper_lap.datasets import MediDec, SlicedDataset
from hyper_lap.models import Unet

warnings.simplefilter("ignore")

BATCH_SIZE = 32
EPOCHS = 10

_key = jr.key(0)


def consume():
    global _key
    _key, _consume = jr.split(_key)
    return _consume


dataset = MediDec("/media/LinuxData/datasets/MediDec/Task01_BrainTumour")

sliced_dataset = SlicedDataset(dataset)

train_loader = DataLoader(sliced_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = Unet(8, [1, 2, 4, 6, 6], in_channels=1, out_channels=2, key=consume())

opt = optax.adamw(1e-4)

opt_state = opt.init(eqx.filter(model, eqx.is_array))


@jax.jit
def loss_fn(logits: Array, labels: Array) -> Array:
    # b c h w
    neg_log_prob = optax.softmax_cross_entropy_with_integer_labels(
        jnp.moveaxis(logits, 0, -1), labels
    )

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


for epoch in (pbar := trange(EPOCHS)):
    losses = []

    inner_pbar = tqdm(total=len(sliced_dataset), leave=False)
    for batch in train_loader:
        # for batch in tqdm(train_loader, leave=False):
        batch: dict[str, Array] = jt.map(jnp.asarray, batch)

        image = batch["image"]
        label = batch["label"]

        image = image[:, 0:1]
        label = (label == 1).astype(jnp.int32)

        loss, model, opt_state = training_step(model, image, label, opt_state)

        losses.append(loss.item())

        inner_pbar.update(BATCH_SIZE)

    mean_loss = jnp.mean(jnp.array(losses))

    pbar.write(f"Loss: {mean_loss:.3}")
