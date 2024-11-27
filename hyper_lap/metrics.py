import jax
import jax.numpy as jnp
from chex import assert_equal_shape, assert_equal_shape_suffix, assert_rank, assert_shape
from jaxtyping import Array, Float, Integer


def dice_score(pred: Float[Array, "h w"], label: Integer[Array, "h w"]) -> Array:
    """
    Calculate Dice score (F1 score).
    """

    assert_rank([pred, label], 2)
    assert_equal_shape([pred, label])

    eps = 1e-10

    pred = pred.astype(bool)
    label = label.astype(bool)

    true_pos = jnp.count_nonzero(pred & label)

    pred_pos = jnp.count_nonzero(pred)
    label_pos = jnp.count_nonzero(label)

    # if pred_pos + label_pos == 0:
    #     warnings.warn("both prediction and label are empty")

    dice_score = 2 * true_pos / (pred_pos + label_pos + eps)

    return dice_score


def jaccard_index(pred: Integer[Array, "h w"], label: Integer[Array, "h w"]) -> Array:
    """
    Calculate Jaccard index (Intersection over Union).
    """

    assert_rank([pred, label], 2)
    assert_equal_shape([pred, label])

    eps = 1e-10

    pred = pred.astype(bool)
    label = label.astype(bool)

    intersection = jnp.count_nonzero(pred & label)

    union = jnp.count_nonzero(pred | label)

    # if union == 0:
    #     warnings.warn("both prediction and label are empty")

    jaccard_index = (intersection + eps) / (union + eps)

    return jaccard_index


def generalised_energy_distance(
    preds: Integer[Array, "n h w"], labels: Integer[Array, "m h w"]
) -> Array:
    """
    Calculate generalised energy distance (GED).
    """
    assert_rank([preds, labels], 3)
    assert_equal_shape_suffix([preds, labels], 2)

    # vmap first axis of each input separately, dist = 1 - IoU
    def dist_fn(x, y):
        ious = jax.vmap(jax.vmap(jaccard_index, in_axes=(None, 0)), in_axes=(0, None))(x, y)
        dists = 1 - ious

        assert_shape([ious, dists], [x.shape[0], y.shape[0]])

        return dists.mean()

    pred_label_dist = dist_fn(preds, labels)
    pred_pred_dist = dist_fn(preds, preds)
    label_label_dist = dist_fn(labels, labels)

    ged = 2 * pred_label_dist - pred_pred_dist - label_label_dist

    return ged
