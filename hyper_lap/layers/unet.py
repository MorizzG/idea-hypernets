from collections.abc import Sequence
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Any

import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr

from .conv import ConvNormAct
from .upsample import BilinearUpsample2d


class ResBlock(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    kernel_size: int = eqx.field(static=True)

    groups: int = eqx.field(static=True)
    n_convs: int = eqx.field(static=True)

    use_weight_standardized_conv: bool = eqx.field(static=True)

    layers: list[ConvNormAct]

    res_conv: nn.Conv2d | nn.Identity

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        groups: int = 8,
        n_convs: int = 2,
        use_weight_standardized_conv: bool = False,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if n_convs < 1:
            raise ValueError("Must have at least one conv")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.groups = groups
        self.n_convs = n_convs
        self.use_weight_standardized_conv = use_weight_standardized_conv

        res_conv_key, *keys = jr.split(key, n_convs + 1)

        self.layers = [
            ConvNormAct(
                in_channels,
                out_channels,
                kernel_size,
                groups=groups,
                use_weight_standardized_conv=use_weight_standardized_conv,
                key=keys[0],
            )
        ]

        self.layers += [
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

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(
                in_channels, out_channels, 1, use_bias=False, key=res_conv_key
            )
        else:
            self.res_conv = nn.Identity()

    def __call__(self, x: Float[Array, "c h w d"]) -> Float[Array, "c h w d"]:
        res = x

        for layer in self.layers:
            x = layer(x)

        x += self.res_conv(res)

        return x


class UnetDown(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    blocks: list[ResBlock]
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

        self.blocks = []
        self.downs = []

        channels = base_channels

        for channel_mult in channel_mults[1:]:
            new_channels = channel_mult * base_channels

            key, block_key = jr.split(key)

            self.blocks.append(ResBlock(channels, new_channels, key=block_key, **block_args))

            channels = new_channels

            self.downs.append(nn.MaxPool2d(2, 2))

    def __call__(self, x: Array) -> tuple[Array, list[Array]]:
        skips: list[Array] = []

        for block, down in zip(self.blocks, self.downs):
            x = block(x)

            skips.append(x)

            _c, h, w = x.shape

            assert h % 2 == 0 and w % 2 == 0, (
                f"spatial dims of shape {x.shape} are not divisible by 2"
            )

            x = down(x)

        return x, skips


class UnetUp(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    ups: list[BilinearUpsample2d]
    blocks: list[ResBlock]

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
        self.blocks = []

        channels = base_channels * channel_mults[-1]

        for channel_mult in list(reversed(channel_mults))[1:]:
            new_channels = channel_mult * base_channels

            key, block_key = jr.split(key)

            self.ups.append(BilinearUpsample2d())

            self.blocks.append(ResBlock(2 * channels, new_channels, key=block_key, **block_args))

            channels = new_channels

    def __call__(self, x: Array, skips: list[Array]) -> Array:
        skips = skips.copy()

        for up, block in zip(self.ups, self.blocks):
            x = up(x)

            skip = skips.pop()

            x = jnp.concat([skip, x], axis=0)

            x = block(x)

        return x


class UnetModule(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    down: UnetDown
    middle: ResBlock
    up: UnetUp

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

        self.down = UnetDown(base_channels, channel_mults, key=down_key, block_args=block_args)

        middle_channels = base_channels * channel_mults[-1]

        self.middle = ResBlock(
            middle_channels, middle_channels, key=middle_key, **(block_args or {})
        )

        self.up = UnetUp(base_channels, channel_mults, key=up_key, block_args=block_args)

    def __call__(self, x: Array) -> Array:
        _c, h, w = x.shape

        down_factor = 2 ** len(self.channel_mults)

        assert h % down_factor == 0, (
            f"spatial dims must be divisible by {down_factor}, but shape is {x.shape}"
        )
        assert w % down_factor == 0, (
            f"spatial dims must be divisible by {down_factor}, but shape is {x.shape}"
        )

        x, skips = self.down(x)

        x = self.middle(x)

        x = self.up(x, skips)

        return x
