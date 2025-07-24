from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional, Sequence

import equinox as eqx
import equinox.nn as nn
import jax.random as jr

from hyper_lap.hyper.embedder import InputEmbedder
from hyper_lap.modules.film_unet import FilmUnetModule
from hyper_lap.modules.unet import Block, ConvNormAct


class FilmUnet(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    emb_size: int = eqx.field(static=True)

    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    embedder: InputEmbedder

    init_conv: ConvNormAct
    unet: FilmUnetModule
    recomb: Block
    final_conv: nn.Conv2d

    def __init__(
        self,
        *,
        base_channels: int,
        channel_mults: Sequence[int],
        in_channels: int,
        out_channels: int,
        emb_size: int,
        embedder_kind: InputEmbedder.EmbedderKind,
        use_res: bool,
        use_weight_standardized_conv: bool,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.emb_size = emb_size

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        key, emb_key = jr.split(key)

        self.embedder = InputEmbedder(emb_size, kind=embedder_kind, key=emb_key)

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
                "use_res": use_res,
                "use_weight_standardized_conv": use_weight_standardized_conv,
            },
        )

        self.recomb = Block(
            base_channels,
            base_channels,
            use_weight_standardized_conv=use_weight_standardized_conv,
            key=recomb_key,
        )

        self.final_conv = nn.Conv2d(base_channels, out_channels, 1, use_bias=False, key=final_key)

    def __call__(
        self, x: Float[Array, "c_in h w"], cond: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "c_out h w"]:
        x = self.init_conv(x)
        x = self.unet(x, cond)
        x = self.recomb(x)
        x = self.final_conv(x)

        return x
