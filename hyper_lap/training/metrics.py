from jaxtyping import Array

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hyper_lap.metrics import dice_score, hausdorff_distance, jaccard_index
from hyper_lap.models import Unet


def calc_metrics(unet: Unet, batch: dict[str, Array]) -> dict[str, Array]:
    images = batch["image"]
    labels = batch["label"]

    logits = eqx.filter_jit(eqx.filter_vmap(unet))(images)

    preds = jnp.argmax(logits, axis=1)

    preds = preds != 0
    labels = labels != 0

    dice = jax.jit(jax.vmap(dice_score))(preds, labels).mean()

    iou = jax.jit(jax.vmap(jaccard_index))(preds, labels).mean()

    hd = np.array([hausdorff_distance(preds[i], labels[i]) for i in range(preds.shape[0])]).mean()

    return {"dice": dice, "iou": iou, "hausdorff": hd}
