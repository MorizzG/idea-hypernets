from jaxtyping import Array, Float, Integer, PRNGKeyArray

import equinox as eqx
import jax.numpy as jnp
from chex import assert_shape

from hyper_lap.layers.vit import ViT


class ViTEmbedder(eqx.Module):
    vit: ViT

    def __init__(self, emb_size: int, key: PRNGKeyArray):
        super().__init__()

        self.vit = ViT(512, 3, 16, 6, 8, 64, emb_size, key=key)

    def __call__(
        self,
        image: Float[Array, "3 h w"],
        label: Integer[Array, "h w"],
        _dataset_idx: Integer[Array, ""],
    ) -> Float[Array, " emb_size"]:
        c, h, w = image.shape

        assert c == 3
        assert_shape(label, [h, w])

        label = jnp.expand_dims((label != 0).astype(image.dtype), 0)

        combined = jnp.concatenate([image, label], axis=0)

        assert_shape(combined, (4, h, w))

        emb = self.vit(combined)

        return emb
