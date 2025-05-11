from jaxtyping import Array, Float, Integer

import jax
import jax.numpy as jnp
import numpy as np

from hyper_lap.metrics import dice_score, hausdorff_distance, jaccard_index


def calc_metrics(logits: Float[Array, "c h w"], labels: Integer[Array, "h w"]) -> dict[str, Array]:
    preds = jnp.argmax(logits, axis=1)

    preds = preds != 0
    labels = labels != 0

    dice = jax.jit(jax.vmap(dice_score))(preds, labels).mean()

    iou = jax.jit(jax.vmap(jaccard_index))(preds, labels).mean()

    hd = np.array([hausdorff_distance(preds[i], labels[i]) for i in range(preds.shape[0])]).mean()

    return {"dice": dice, "iou": iou, "hausdorff": hd}
