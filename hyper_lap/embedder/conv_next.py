from jaxtyping import Array, Float, Integer, PRNGKeyArray

import equinox as eqx
import jax.numpy as jnp
from chex import assert_shape

from hyper_lap.layers.convnext import ConvNeXt


class ConvNextEmbedder(eqx.Module):
    convnext: ConvNeXt

    def __init__(self, emb_size: int, key: PRNGKeyArray):
        super().__init__()

        self.convnext = ConvNeXt(emb_size, 96, in_channels=3, depths=[3, 3, 9, 3], key=key)

    def __call__(
        self,
        image: Float[Array, "3 h w"],
        label: Integer[Array, "h w"],
        _dataset_idx: Integer[Array, ""],
    ) -> Float[Array, " emb_size"]:
        c, h, w = image.shape

        assert c == 3
        assert_shape(label, [h, w])

        image = image.mean(axis=0)

        pos_masked_image = (label != 0) * image
        neg_masked_image = (label == 0) * image

        input = jnp.stack([image, pos_masked_image, neg_masked_image])

        assert_shape(input, (3, h, w))

        emb = self.convnext(input)

        return emb
