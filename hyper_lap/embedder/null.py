from jaxtyping import Array, Float, Integer

import equinox as eqx
import jax.numpy as jnp


class ZeroEmbedder(eqx.Module):
    emb_size: int

    def __init__(self, emb_size: int):
        super().__init__()

        self.emb_size = emb_size

    def __call__(
        self,
        image: Float[Array, "3 h w"],
        label: Integer[Array, "h w"],
        _dataset_idx: Integer[Array, ""],
    ) -> Float[Array, " emb_size"]:
        return jnp.zeros(self.emb_size)
