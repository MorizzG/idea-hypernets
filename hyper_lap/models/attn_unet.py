from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional, Sequence

import equinox as eqx
import equinox.nn as nn
import jax.random as jr

from hyper_lap.hyper.embedder import InputEmbedder
from hyper_lap.modules.attn_unet import AttnUnetModule
from hyper_lap.modules.unet import ConvNormAct, ResBlock


class AttentionUnet(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    init_conv: ConvNormAct
    attn_unet: AttnUnetModule
    recomb: ResBlock
    final_conv: nn.Conv2d

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        in_channels: int,
        out_channels: int,
        *,
        use_weight_standardized_conv: bool = False,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        key, emb_key = jr.split(key)

        # self.embedder = InputEmbedder(emb_size, kind=embedder_kind, key=emb_key)

        init_key, unet_key, recomb_key, final_key = jr.split(key, 4)

        self.init_conv = ConvNormAct(
            in_channels,
            base_channels,
            kernel_size=1,
            use_weight_standardized_conv=use_weight_standardized_conv,
            key=init_key,
        )

        self.attn_unet = AttnUnetModule(
            base_channels,
            channel_mults,
            key=unet_key,
            block_args={
                "use_weight_standardized_conv": use_weight_standardized_conv,
            },
        )

        self.recomb = ResBlock(
            2 * base_channels,
            use_weight_standardized_conv=use_weight_standardized_conv,
            key=recomb_key,
        )

        self.final_conv = nn.Conv2d(
            2 * base_channels, out_channels, 1, use_bias=False, key=final_key
        )

    def __call__(
        self,
        x: Float[Array, "c_in h w"],
        context: Array | None = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "c_out h w"]:
        x = self.init_conv(x)
        x = self.attn_unet(x, context)
        x = self.recomb(x)
        x = self.final_conv(x)

        return x
