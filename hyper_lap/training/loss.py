from jaxtyping import Array, Float, Integer

import jax
import jax.numpy as jnp
import optax
from chex import assert_equal_shape, assert_equal_shape_suffix, assert_rank, assert_shape


def ce_loss_fn(logits: Float[Array, "c h w"], labels: Integer[Array, "h w"]) -> Array:
    # C H W -> H W C
    logits = jnp.moveaxis(logits, 0, -1)

    neg_log_prob = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    # sum over spatial dims
    neg_log_likelihood = neg_log_prob.sum()

    return neg_log_likelihood


def focal_loss_fn(logits: Float[Array, "c h w"], labels: Integer[Array, "h w"]) -> Array:
    assert_rank(logits, 3)
    assert_rank(labels, 2)

    assert_equal_shape_suffix([logits, labels], 2)

    # C H W -> H W C
    # logits = jnp.moveaxis(logits, 0, -1)

    # neg_log_prob = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    log_probs = jax.nn.log_softmax(logits, axis=0)[1, ...]

    assert_equal_shape([log_probs, labels])

    neg_log_prob = optax.sigmoid_focal_loss(logits=log_probs, labels=labels, alpha=0.25, gamma=2.0)

    # sum over spatial dims
    neg_log_likelihood = neg_log_prob.sum()

    return neg_log_likelihood


def dice_loss_fn(logits: Float[Array, "c h w"], labels: Integer[Array, "h w"]) -> Array:
    assert_rank(logits, 3)
    assert_rank(labels, 2)

    eps = 1.0

    probs = jax.nn.softmax(logits, axis=0)
    labels = jax.nn.one_hot(labels, 2, axis=0)

    assert_equal_shape([logits, probs, labels])

    intersection = (probs * labels).sum(axis=(1, 2))

    prob_sum = probs.sum(axis=(1, 2))
    label_sum = labels.sum(axis=(1, 2))

    dice = (2.0 * intersection + eps) / (prob_sum + label_sum + eps)

    dice_loss = 1.0 - dice

    assert_shape(dice_loss, (2,))

    return dice_loss.sum()


def hybrid_loss_fn(logits: Float[Array, "c h w"], labels: Integer[Array, "h w"]) -> Array:
    return 0.5 * ce_loss_fn(logits, labels) + 0.5 * dice_loss_fn(logits, labels)
