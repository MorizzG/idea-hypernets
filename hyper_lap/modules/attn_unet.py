from jaxtyping import Array, Float, PRNGKeyArray
from typing import Any, Optional, Sequence

import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr

from .attention import SpatialCrossAttention
from .conv import ConvNormAct
from .upsample import BilinearUpsample2d


class ResBlock(eqx.Module):
    res_conv: Optional[nn.Conv2d]

    layers: nn.Sequential

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 8,
        n_convs: int = 2,
        *,
        use_weight_standardized_conv: bool,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if n_convs < 1:
            raise ValueError("Must have at least one conv")

        keys = jr.split(key, n_convs)

        layers = [
            ConvNormAct(
                in_channels,
                out_channels,
                kernel_size,
                groups=groups,
                use_weight_standardized_conv=use_weight_standardized_conv,
                key=keys[0],
            )
        ]

        layers += [
            ConvNormAct(
                out_channels,
                out_channels,
                kernel_size,
                groups=groups,
                use_weight_standardized_conv=use_weight_standardized_conv,
                key=key,
            )
            for key in keys[1:]
        ]

        self.layers = nn.Sequential(layers)

    def __call__(
        self, x: Float[Array, "c h w d"], *, key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "c h w d"]:
        res = x

        for layer in self.layers:
            x = layer(x)

        return x


class AttnUnetDown(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    res_blocks: list[ResBlock]
    attn_blocks: list[SpatialCrossAttention]
    downs: list[nn.MaxPool2d]

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        *,
        key: PRNGKeyArray,
        block_args: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        if block_args is None:
            block_args = {}

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        channels = base_channels

        self.res_blocks = []
        self.attn_blocks = []
        self.downs = []

        for channel_mult in channel_mults[1:]:
            new_channels = channel_mult * base_channels

            key, res_key, attn_key = jr.split(key, 3)

            self.res_blocks.append(ResBlock(channels, new_channels, key=res_key, **block_args))

            channels = new_channels

            self.attn_blocks.append(SpatialCrossAttention(channels, 4, channels // 2, key=attn_key))

            self.downs.append(nn.MaxPool2d(2, 2))

    def __call__(
        self, x: Array, context: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, list[Array]]:
        skips: list[Array] = []

        for res_block, attn_block, down in zip(self.res_blocks, self.attn_blocks, self.downs):
            x = res_block(x)

            x = attn_block(x, context)

            skips.append(x)

            c, h, w = x.shape

            assert h % 2 == 0 and w % 2 == 0, (
                f"spatial dims of shape {x.shape} are not divisible by 2"
            )

            x = down(x)

        return x, skips


class AttnUnetUp(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    res_blocks: list[ResBlock]
    attn_blocks: list[SpatialCrossAttention]
    ups: list[BilinearUpsample2d]

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        *,
        key: PRNGKeyArray,
        block_args: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        if block_args is None:
            block_args = {}

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        self.res_blocks = []
        self.attn_blocks = []
        self.ups = []

        channels = base_channels * channel_mults[-1]

        for channel_mult in list(reversed(channel_mults))[1:]:
            new_channels = channel_mult * base_channels

            key, res_key, attn_key = jr.split(key, 3)

            # self.ups.append(ConvUpsample2d(channels, channels, key=up_key))
            self.ups.append(BilinearUpsample2d())

            self.res_blocks.append(ResBlock(2 * channels, new_channels, key=res_key, **block_args))

            self.attn_blocks.append(SpatialCrossAttention(channels, 4, channels // 2, key=attn_key))

            channels = new_channels

    def __call__(
        self, x: Array, context: Array, skips: list[Array], *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        skips = skips.copy()

        for up, res_block, attn_block in zip(self.ups, self.res_blocks, self.attn_blocks):
            x = up(x)

            skip = skips.pop()

            x = jnp.concat([skip, x], axis=0)

            x = res_block(x)

            x = attn_block(x, context)

        return x


class AttnUnetModule(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    down: AttnUnetDown

    middle_res1: ResBlock
    middle_attn: SpatialCrossAttention
    middle_res2: ResBlock

    up: AttnUnetUp

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        *,
        key: PRNGKeyArray,
        block_args: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        down_key, middle_key, up_key = jr.split(key, 3)

        self.down = AttnUnetDown(base_channels, channel_mults, key=down_key, block_args=block_args)

        middle_channels = base_channels * channel_mults[-1]

        res1_key, attn_key, res2_key = jr.split(middle_key, 3)

        self.middle_res1 = ResBlock(
            middle_channels, middle_channels, key=res1_key, **(block_args or {})
        )
        self.middle_attn = SpatialCrossAttention(
            middle_channels, 4, middle_channels // 2, key=attn_key
        )
        self.middle_res2 = ResBlock(
            middle_channels, middle_channels, key=res2_key, **(block_args or {})
        )

        self.up = AttnUnetUp(base_channels, channel_mults, key=up_key, block_args=block_args)

    def __call__(self, x: Array, context: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        c, h, w = x.shape

        down_factor = 2 ** len(self.channel_mults)

        assert h % down_factor == 0, (
            f"spatial dims must be divisible by {down_factor}, but shape is {x.shape}"
        )
        assert w % down_factor == 0, (
            f"spatial dims must be divisible by {down_factor}, but shape is {x.shape}"
        )

        x, skips = self.down(x, context)

        # x = self.middle(x)
        x = self.middle_res1(x)
        x = self.middle_attn(x, context)
        x = self.middle_res2(x)

        x = self.up(x, context, skips)

        return x
