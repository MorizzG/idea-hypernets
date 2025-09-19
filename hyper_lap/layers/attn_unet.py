from collections.abc import Sequence
from jaxtyping import Array, PRNGKeyArray
from typing import Any

import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr

from .attention import SpatialCrossAttention
from .unet import ResBlock
from .upsample import BilinearUpsample2d


class AttnUnetDown(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    resamples: list[nn.Conv2d]
    blocks: list[ResBlock]
    attn_blocks: list[SpatialCrossAttention | None]
    downs: list[nn.MaxPool2d]

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        *,
        key: PRNGKeyArray,
        block_args: dict[str, Any] | None = None,
    ):
        super().__init__()

        if block_args is None:
            block_args = {}

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        channels = base_channels

        self.blocks = []
        self.attn_blocks = []
        self.downs = []
        self.resamples = []

        for i, channel_mult in enumerate(channel_mults[1:]):
            new_channels = channel_mult * base_channels

            key, resample_key, res_key, attn_key = jr.split(key, 4)

            self.blocks.append(ResBlock(channels, key=res_key, **block_args))

            if i != 0:
                self.attn_blocks.append(
                    SpatialCrossAttention(channels, 8, channels // 4, key=attn_key)
                )
            else:
                self.attn_blocks.append(None)

            self.resamples.append(
                nn.Conv2d(channels, new_channels, 1, use_bias=False, key=resample_key)
            )

            channels = new_channels

            self.downs.append(nn.MaxPool2d(2, 2))

    def __call__(self, x: Array, context: Array | None = None) -> tuple[Array, list[Array]]:
        skips: list[Array] = []

        for block, resample, attn_block, down in zip(
            self.blocks, self.resamples, self.attn_blocks, self.downs
        ):
            x = block(x)

            if attn_block is not None:
                x = attn_block(x, context)

            skips.append(x)

            _c, h, w = x.shape

            assert h % 2 == 0 and w % 2 == 0, (
                f"spatial dims of shape {x.shape} are not divisible by 2"
            )

            x = resample(x)

            x = down(x)

        return x, skips


class AttnUnetUp(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    ups: list[BilinearUpsample2d]
    resamples: list[nn.Conv2d]
    blocks: list[ResBlock]
    attn_blocks: list[SpatialCrossAttention | None]

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        *,
        key: PRNGKeyArray,
        block_args: dict[str, Any] | None = None,
    ):
        super().__init__()

        if block_args is None:
            block_args = {}

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        self.ups = []
        self.resamples = []
        self.blocks = []
        self.attn_blocks = []

        channels = base_channels * channel_mults[-1]

        for i, channel_mult in enumerate(list(reversed(channel_mults))[1:]):
            new_channels = channel_mult * base_channels

            key, resample_key, res_key, attn_key = jr.split(key, 4)

            # self.ups.append(ConvUpsample2d(channels, channels, key=up_key))
            self.ups.append(BilinearUpsample2d())

            self.resamples.append(nn.Conv2d(channels, new_channels, 1, key=resample_key))

            channels = 2 * new_channels

            self.blocks.append(ResBlock(channels, key=res_key, **block_args))

            if i != len(channel_mults) - 2:
                self.attn_blocks.append(
                    SpatialCrossAttention(channels, 8, channels // 4, key=attn_key)
                )
            else:
                self.attn_blocks.append(None)

    def __call__(
        self,
        x: Array,
        skips: list[Array],
        context: Array | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Array:
        skips = skips.copy()

        for up, resample, block, attn_block in zip(
            self.ups, self.resamples, self.blocks, self.attn_blocks
        ):
            x = up(x)

            x = resample(x)

            skip = skips.pop()

            x = jnp.concat([skip, x], axis=0)

            x = block(x)

            if attn_block is not None:
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
        block_args: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        down_key, middle_key, up_key = jr.split(key, 3)

        self.down = AttnUnetDown(base_channels, channel_mults, key=down_key, block_args=block_args)

        middle_channels = base_channels * channel_mults[-1]

        res1_key, attn_key, res2_key = jr.split(middle_key, 3)

        self.middle_res1 = ResBlock(middle_channels, key=res1_key, **(block_args or {}))
        self.middle_attn = SpatialCrossAttention(
            middle_channels, 8, middle_channels // 4, key=attn_key
        )
        self.middle_res2 = ResBlock(middle_channels, key=res2_key, **(block_args or {}))

        self.up = AttnUnetUp(base_channels, channel_mults, key=up_key, block_args=block_args)

    def __call__(self, x: Array, context: Array | None = None) -> Array:
        _c, h, w = x.shape

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

        x = self.up(x, skips, context)

        return x
