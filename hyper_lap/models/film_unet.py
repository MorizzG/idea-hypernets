from collections.abc import Sequence
from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn
import jax.random as jr

from hyper_lap.layers.conv import ConvNormAct
from hyper_lap.layers.film_unet import FilmUnetModule
from hyper_lap.layers.unet import ResBlock


class FilmUnet(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    emb_size: int = eqx.field(static=True)

    init_conv: ConvNormAct
    unet: FilmUnetModule
    recomb: ResBlock
    final_conv: nn.Conv2d

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        in_channels: int,
        out_channels: int,
        *,
        emb_size: int,
        kernel_size: int = 5,
        use_weight_standardized_conv: bool = False,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        self.emb_size = emb_size

        init_key, unet_key, recomb_key, final_key = jr.split(key, 4)

        self.init_conv = ConvNormAct(
            in_channels,
            base_channels,
            kernel_size=1,
            use_weight_standardized_conv=use_weight_standardized_conv,
            key=init_key,
        )

        self.unet = FilmUnetModule(
            base_channels,
            channel_mults,
            emb_size,
            key=unet_key,
            block_args={
                "kernel_size": kernel_size,
                "use_weight_standardized_conv": use_weight_standardized_conv,
            },
        )

        self.recomb = ResBlock(
            base_channels,
            base_channels,
            kernel_size=kernel_size,
            use_weight_standardized_conv=use_weight_standardized_conv,
            key=recomb_key,
        )

        self.final_conv = nn.Conv2d(base_channels, out_channels, 1, use_bias=False, key=final_key)

    def __call__(
        self, x: Float[Array, "c_in h w"], input_emb: Array | None = None
    ) -> Float[Array, "c_out h w"]:
        if input_emb is None:
            raise ValueError("input_emb can't be None in FilmUnet")

        x = self.init_conv(x)
        x = self.unet(x, input_emb)
        x = self.recomb(x)
        x = self.final_conv(x)

        return x
