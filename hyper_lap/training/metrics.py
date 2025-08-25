from jaxtyping import Array, Float, Integer

import jax
import jax.numpy as jnp

from hyper_lap.metrics import dice_score, hausdorff_distance, jaccard_index


def calc_metrics(
    logits: Float[Array, "b c h w"], labels: Integer[Array, "b h w"]
) -> dict[str, Array]:
    preds = jnp.argmax(logits, axis=1)

    preds = preds != 0
    labels = labels != 0

    @jax.jit
    def dice_fn(preds, labels):
        return jax.vmap(dice_score)(preds, labels).mean()

    @jax.jit
    def iou_fn(preds, labels):
        return jax.vmap(jaccard_index)(preds, labels).mean()

    dice = dice_fn(preds, labels)

    iou = iou_fn(preds, labels)

    hd = jnp.array([hausdorff_distance(preds[i], labels[i]) for i in range(preds.shape[0])]).mean()

    return {"dice": dice, "iou": iou, "hausdorff": hd}
