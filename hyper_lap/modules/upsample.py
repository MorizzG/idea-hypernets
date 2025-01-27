from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional

import equinox as eqx
import equinox.nn as nn
import jax

from ._util import _channel_to_spatials2d


class ConvUpsample2d(eqx.Module):
    conv: nn.Conv2d

    def __init__(self, in_channels: int, out_channels: int, *, key: PRNGKeyArray):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 2**2 * out_channels, 1, use_bias=False, key=key)

    def __call__(
        self, x: Float[Array, "c h w"], *, key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "c h w"]:
        x = self.conv(x)

        x = _channel_to_spatials2d(x)

        return x


class ConvTransposedUpsample2d(eqx.Module):
    conv: nn.ConvTranspose2d

    def __init__(self, in_channels: int, out_channels: int, *, key: PRNGKeyArray):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, key=key)

    def __call__(
        self, x: Float[Array, "c h w"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "c 2*h 2*w"]:
        x = self.conv(x)

        return x


class BilinearUpsample2d(eqx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Array) -> Float[Array, "c h w"]:
        c, h, w = x.shape

        return jax.image.resize(x, [c, 2 * h, 2 * w], method="bilinear")
