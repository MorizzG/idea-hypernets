from typing import Optional, Sequence

import equinox as eqx
import equinox.nn as nn
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from hyper_lap.modules.unet import Block, UnetModule


class Unet(eqx.Module):
    init_conv: nn.Conv2d
    unet: UnetModule
    recomb: Block
    final_conv: nn.Conv2d

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        in_channels: int = 1,
        out_channels: int = 2,
        *,
        use_res: bool = False,
        key: PRNGKeyArray,
    ):
        super().__init__()

        init_key, unet_key, recomb_key, final_key = jr.split(key, 4)

        self.init_conv = nn.Conv2d(in_channels, base_channels, 1, key=init_key)

        self.unet = UnetModule(
            base_channels, channel_mults, key=unet_key, block_args={"use_res": use_res}
        )

        self.recomb = Block(base_channels, base_channels, key=recomb_key)

        self.final_conv = nn.Conv2d(base_channels, out_channels, 1, key=final_key)

    def __call__(
        self, x: Float[Array, "c_in h w d"], *, key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "c_out h w d"]:
        x = self.init_conv(x)
        x = self.unet(x)
        x = self.recomb(x)
        x = self.final_conv(x)

        return x
