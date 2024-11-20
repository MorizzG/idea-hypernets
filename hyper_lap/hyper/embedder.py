from chex import assert_shape
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer


class InputEmbedder(eqx.Module):
    def __init__(self, emb_size: int):
        super().__init__()

        self.emb_size = emb_size

    def __call__(
        self, image: Float[Array, "c h w"], label: Integer[Array, "h w"]
    ) -> Float[Array, "k"]:
        c, h, w = image.shape
        assert_shape(label, (h, w))

        max_label = label.max()

        labels_onehot = jnp.zeros([max_label + 1, h, w]).at[label].set(1)
