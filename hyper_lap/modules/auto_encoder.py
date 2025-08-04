from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jr


class AutoEncoder(eqx.Module):
    down1: nn.Linear
    down2: nn.Linear
    down3: nn.Linear

    up1: nn.Linear
    up2: nn.Linear
    up3: nn.Linear

    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        middle_sizes: tuple[int, int] | None = None,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if middle_sizes is None:
            middle_sizes = (in_size, in_size)

        middle1, middle2 = middle_sizes

        keys = jr.split(key, 6)

        self.down1 = nn.Linear(in_size, middle1, key=keys[0])
        self.down2 = nn.Linear(middle1, middle2, key=keys[1])
        self.down3 = nn.Linear(middle2, out_size, key=keys[2])

        self.up1 = nn.Linear(out_size, middle2, key=keys[3])
        self.up2 = nn.Linear(middle2, middle1, key=keys[4])
        self.up3 = nn.Linear(middle1, in_size, key=keys[5])

    def encode(self, x: Float[Array, "in_size"]) -> Float[Array, "out_size"]:
        x = self.down1(x)
        x = jax.nn.silu(x)
        x = self.down2(x)
        x = jax.nn.silu(x)
        x = self.down3(x)

        return x

    def decode(self, x: Float[Array, "out_size"]) -> Float[Array, "in_size"]:
        x = self.up1(x)
        x = jax.nn.silu(x)
        x = self.up2(x)
        x = jax.nn.silu(x)
        x = self.up3(x)

        return x

    def __call__(self, x: Float[Array, "in_size"]) -> Float[Array, "in_size"]:
        return self.decode(self.encode(x))
