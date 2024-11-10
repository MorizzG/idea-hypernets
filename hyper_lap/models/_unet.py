from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional, Sequence

import equinox.nn as nn
import jax.random as jr

from hyper_lap.modules.unet import Block, UnetModule


class Unet:
    init_conv: nn.Conv3d
    unet: UnetModule
    recomb: Block
    final_conv: nn.Conv3d

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        in_channels: int = 1,
        *,
        use_res: bool = False,
        key: PRNGKeyArray,
    ):
        super().__init__()

        init_key, unet_key, recomb_key, final_key = jr.split(key, 4)

        self.init_conv = nn.Conv3d(in_channels, base_channels, 1, key=init_key)

        self.unet = UnetModule(
            base_channels, channel_mults, key=unet_key, block_args={"use_res": use_res}
        )

        self.recomb = Block(base_channels, base_channels, key=recomb_key)

        self.final_conv = nn.Conv3d(base_channels, 2, 1, key=final_key)

    def __call__(
        self, x: Float[Array, "c_in h w d"], *, key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "c_out h w d"]:
        x = self.init_conv(x)
        x = self.unet(x)
        x = self.recomb(x)
        x = self.final_conv(x)

        return x
