import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


class Conv2dGenerator(eqx.Module):
    emb_size: int
    h_size: int

    in_channels: int
    out_channels: int
    kernel_size: int

    first: eqx.nn.Linear
    second: eqx.nn.Linear

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        emb_size: int,
        h_size: int | None = None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if h_size is None:
            h_size = emb_size

        self.emb_size = emb_size
        self.h_size = h_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        first_key, second_key = jr.split(key)

        # project from embeddings to hidden_size * in_channels
        self.first = eqx.nn.Linear(emb_size, h_size * in_channels, key=first_key)

        # project from hidden to kernels
        self.second = eqx.nn.Linear(h_size, out_channels * kernel_size**2, key=second_key)

    def __call__(self, input_emb: Array, pos_emb: Array) -> Float[Array, "c_out c_in k k"]:
        emb = jnp.concatenate([input_emb, pos_emb])

        a_s = self.first(emb)

        a_s = a_s.reshape(self.in_channels, self.h_size)

        # shape: block, block*kernel**2
        kernel = jax.vmap(self.second)(a_s)

        # c_in c_out k k
        kernel = kernel.reshape(
            self.in_channels, self.out_channels, self.kernel_size, self.kernel_size
        )

        # swap to c_out c_in k k
        kernel = kernel.swapaxes(0, 1)

        return kernel
