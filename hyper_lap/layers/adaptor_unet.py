from jaxtyping import Array, PRNGKeyArray
from typing import Any, Optional, Sequence

import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr

from .unet import ResBlock
from .upsample import BilinearUpsample2d


class AdaptorUnetDown(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    blocks: list[ResBlock]
    adaptors: list[nn.Conv2d]
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

        self.blocks = []
        self.adaptors = []
        self.downs = []

        channels = base_channels

        for channel_mult in channel_mults[1:]:
            new_channels = channel_mult * base_channels

            key, block_key, adaptor_key = jr.split(key, 3)

            self.blocks.append(ResBlock(channels, new_channels, key=block_key, **block_args))

            channels = new_channels

            self.adaptors.append(
                nn.Conv2d(new_channels, new_channels, 1, use_bias=False, key=adaptor_key)
            )

            self.downs.append(nn.MaxPool2d(2, 2))

    def __call__(self, x: Array) -> tuple[Array, list[Array]]:
        skips: list[Array] = []

        for block, down in zip(self.blocks, self.downs):
            x = block(x)

            skips.append(x)

            c, h, w = x.shape

            assert (
                h % 2 == 0 and w % 2 == 0
            ), f"spatial dims of shape {x.shape} are not divisible by 2"

            x = down(x)

        return x, skips


class AdaptorUnetUp(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    ups: list[BilinearUpsample2d]
    blocks: list[ResBlock]
    adaptors: list[nn.Conv2d]

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

        self.ups = []
        self.blocks = []
        self.adaptors = []

        channels = base_channels * channel_mults[-1]

        for channel_mult in list(reversed(channel_mults))[1:]:
            new_channels = channel_mult * base_channels

            key, block_key, adaptor_key = jr.split(key, 3)

            self.ups.append(BilinearUpsample2d())

            self.blocks.append(ResBlock(2 * channels, new_channels, key=block_key, **block_args))

            self.adaptors.append(
                nn.Conv2d(new_channels, new_channels, 1, use_bias=False, key=adaptor_key)
            )

            channels = new_channels

    def __call__(self, x: Array, skips: list[Array]) -> Array:
        skips = skips.copy()

        for up, block in zip(self.ups, self.blocks):
            x = up(x)

            skip = skips.pop()

            x = jnp.concat([skip, x], axis=0)

            x = block(x)

        return x


class AdaptorUnetModule(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    down: AdaptorUnetDown
    middle: ResBlock
    up: AdaptorUnetUp

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

        self.down = AdaptorUnetDown(
            base_channels, channel_mults, key=down_key, block_args=block_args
        )

        middle_channels = base_channels * channel_mults[-1]

        self.middle = ResBlock(
            middle_channels, middle_channels, key=middle_key, **(block_args or {})
        )

        self.up = AdaptorUnetUp(base_channels, channel_mults, key=up_key, block_args=block_args)

    def __call__(self, x: Array) -> Array:
        c, h, w = x.shape

        down_factor = 2 ** len(self.channel_mults)

        assert (
            h % down_factor == 0
        ), f"spatial dims must be divisible by {down_factor}, but shape is {x.shape}"
        assert (
            w % down_factor == 0
        ), f"spatial dims must be divisible by {down_factor}, but shape is {x.shape}"

        x, skips = self.down(x)

        x = self.middle(x)

        x = self.up(x, skips)

        return x
