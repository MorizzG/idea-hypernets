from jaxtyping import Array, Bool, jaxtyped

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
import skimage.metrics
from beartype import beartype
from chex import (
    assert_equal_shape,
    assert_equal_shape_suffix,
    assert_rank,
    assert_shape,
    assert_type,
)


# @jaxtyped(typechecker=beartype)
def dice_score(pred: Bool[Array, "h w"], label: Bool[Array, "h w"]) -> Array:
    """
    Calculate Dice score (F1 score).
    """

    assert_rank([pred, label], 2)
    assert_equal_shape([pred, label])
    assert_type([pred, label], jnp.bool)

    eps = 1e-10

    true_pos = jnp.count_nonzero(pred & label)

    pred_pos = jnp.count_nonzero(pred)
    label_pos = jnp.count_nonzero(label)

    dice_score = 2 * true_pos / (pred_pos + label_pos + eps)

    return dice_score


@jaxtyped(typechecker=beartype)
def intersection_over_union(pred: Bool[Array, "h w"], label: Bool[Array, "h w"]) -> Array:
    """
    Calculate Jaccard index (Intersection over Union).
    """

    assert_rank([pred, label], 2)
    assert_equal_shape([pred, label])
    assert_type([pred, label], jnp.bool)

    eps = 1e-10

    intersection = jnp.count_nonzero(pred & label)

    union = jnp.count_nonzero(pred | label)

    jaccard_index = intersection / (union + eps)

    return jaccard_index


@jaxtyped(typechecker=beartype)
def generalised_energy_distance(preds: Bool[Array, "n h w"], labels: Bool[Array, "m h w"]) -> Array:
    """
    Calculate generalised energy distance (GED).
    """
    assert_rank([preds, labels], 3)
    assert_equal_shape_suffix([preds, labels], 2)
    assert_type([preds, labels], jnp.bool)

    # vmap first axis of each input separately, dist = 1 - IoU
    def dist_fn(x, y):
        ious = jax.vmap(jax.vmap(intersection_over_union, in_axes=(None, 0)), in_axes=(0, None))(
            x, y
        )
        dists = 1 - ious

        assert_shape([ious, dists], [x.shape[0], y.shape[0]])

        return dists.mean()

    pred_label_dist = dist_fn(preds, labels)
    pred_pred_dist = dist_fn(preds, preds)
    label_label_dist = dist_fn(labels, labels)

    ged = 2 * pred_label_dist - pred_pred_dist - label_label_dist

    return ged


@jaxtyped(typechecker=beartype)
def hausdorff_distance(pred: Bool[Array, "h w"], label: Bool[Array, "h w"]) -> float:
    """
    Calculate Hausdorff distance.
    """

    assert_rank([pred, label], 2)
    assert_equal_shape([pred, label])
    assert_type([pred, label], jnp.bool)

    assert not isinstance(pred, jax.core.Tracer), "Can't use hausdorff inside JIT"

    pred_np = np.asarray(pred)
    label_np = np.asarray(label)

    if np.count_nonzero(pred_np) == 0 or np.count_nonzero(label_np) == 0:
        return 1.0

    d = skimage.metrics.hausdorff_distance(pred_np, label_np)

    if d == float("inf"):
        print("found inf")
        return 1.0

    assert isinstance(d, float)

    h, w = pred_np.shape

    diag = np.sqrt(h**2 + w**2)

    return d / diag
