from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr

from hyper_lap.modules import SiLU


class Conv2dGenerator(eqx.Module):
    input_emb_size: int = eqx.field(static=True)
    pos_emb_size: int = eqx.field(static=True)
    h_size: int = eqx.field(static=True)

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    first: nn.Linear

    # middle: eqx.nn.Linear
    # middle: nn.Sequential
    middle: nn.MLP

    second: nn.Linear

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        input_emb_size: int,
        pos_emb_size: int,
        h_size: int | None = None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        total_emb_size = input_emb_size + pos_emb_size

        if h_size is None:
            h_size = input_emb_size + pos_emb_size

        self.input_emb_size = input_emb_size
        self.pos_emb_size = pos_emb_size
        self.h_size = h_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        first_key, middle_key, second_key = jr.split(key, 3)

        # project from embeddings to hidden_size * in_channels
        self.first = eqx.nn.Linear(total_emb_size, h_size * in_channels, key=first_key)

        # middle_keys = jr.split(middle_key, 5)

        # middle = [eqx.nn.Linear(h_size, h_size, key=middle_keys[0])]

        # for key in middle_keys[1:]:
        #     middle += [SiLU(), eqx.nn.Linear(h_size, h_size, key=key)]

        # self.middle = nn.Sequential(middle)

        self.middle = nn.MLP(
            in_size=h_size,
            out_size=h_size,
            width_size=h_size,
            depth=3,
            activation=jax.nn.swish,
            key=middle_key,
        )

        # project from hidden to kernels
        self.second = eqx.nn.Linear(h_size, out_channels * kernel_size**2, key=second_key)

    def __call__(
        self, input_emb: Float[Array, " input_emb_size"], pos_emb: Float[Array, " pos_emb_size"]
    ) -> Float[Array, "c_out c_in k k"]:
        assert input_emb.shape == (self.input_emb_size,)
        assert pos_emb.shape == (self.pos_emb_size,)

        emb = jnp.concatenate([input_emb, pos_emb])

        a_s = self.first(emb)

        a_s = a_s.reshape(self.in_channels, self.h_size)

        x = jax.nn.swish(a_s)

        x = jax.vmap(self.middle)(x)

        x = jax.nn.swish(x)

        # shape: in_channels, out_channels * kernel_size**2
        kernel = jax.vmap(self.second)(x)

        # c_in c_out k k
        kernel = kernel.reshape(
            self.in_channels, self.out_channels, self.kernel_size, self.kernel_size
        )

        # swap to c_out c_in k k
        kernel = kernel.swapaxes(0, 1)

        return kernel
