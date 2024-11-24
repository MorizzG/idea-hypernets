from chex import assert_shape
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer, PRNGKeyArray

from hyper_lap.modules.vit import ViT


class InputEmbedder(eqx.Module):
    emb_size: int

    vit: ViT

    def __init__(self, emb_size: int, *, key: PRNGKeyArray):
        super().__init__()

        self.emb_size = emb_size

        self.vit = ViT(512, 3, 16, 6, 8, 64, emb_size, key=key)

    def __call__(
        self, image: Float[Array, "1 h w"], label: Integer[Array, "h w"]
    ) -> Float[Array, "k"]:
        c, h, w = image.shape

        assert c == 1
        assert_shape(label, (h, w))

        labels_onehot = jnp.zeros([2, h, w]).at[label].set(1)

        combined = jnp.concatenate([image, labels_onehot], axis=0)

        assert_shape(combined, (3, h, w))

        emb = self.vit(combined)

        return emb
