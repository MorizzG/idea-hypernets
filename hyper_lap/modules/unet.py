from jaxtyping import Array, Float, PRNGKeyArray
from typing import Any, Optional, Sequence

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr

from ._util import _channel_to_spatials


class ReLU(eqx.Module):
    def __call__(self, x: Array) -> Array:
        return jax.nn.relu(x)


class Upsample3d(eqx.Module):
    conv: nn.Conv3d

    def __init__(self, in_channels: int, out_channels: int, *, key: PRNGKeyArray):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, 4 * out_channels, 1, key=key)

    def __call__(
        self, x: Float[Array, "c h w d"], *, key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "c h w d"]:
        x = self.conv(x)

        x = _channel_to_spatials(x)

        return x


class ConvNormAct(eqx.Module):
    conv: nn.Conv3d
    # norm: nn.BatchNorm2d
    norm: nn.GroupNorm
    act: ReLU

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups=8,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, padding="SAME", use_bias=False, key=key
        )

        # self.norm = nn.BatchNorm2d(out_channels, "batch")
        self.norm = nn.GroupNorm(groups, out_channels)

        self.act = ReLU()

    def __call__(
        self, x: Float[Array, "c h w d"], *, key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "c h w d"]:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class Block(eqx.Module):
    res_conv: Optional[nn.Conv2d]

    use_res: bool = eqx.field(static=True)

    layers: nn.Sequential

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 8,
        n_convs: int = 2,
        use_res: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if n_convs < 1:
            raise ValueError("Must have at least one conv")

        self.use_res = use_res

        if use_res and in_channels != out_channels:
            key, consume = jr.split(key)

            self.res_conv = nn.Conv2d(in_channels, out_channels, 1, padding="SAME", key=consume)
        else:
            self.res_conv = None

        keys = jr.split(key, n_convs)

        layers = [ConvNormAct(in_channels, out_channels, kernel_size, groups=groups, key=keys[0])]

        layers += [
            ConvNormAct(out_channels, out_channels, kernel_size, groups=groups, key=key)
            for key in keys[1:]
        ]

        self.layers = nn.Sequential(layers)

    def __call__(
        self, x: Float[Array, "c h w d"], *, key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "c h w d"]:
        res = x

        for layer in self.layers:
            x = layer(x)

        if self.use_res:
            if self.res_conv is not None:
                res = self.res_conv(res)

            x += res

        return x


class UnetDown(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    blocks: list[Block]
    downs: list[nn.MaxPool3d]

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

        self.blocks = []
        self.downs = []

        for channel_mult in channel_mults[1:]:
            new_channels = channel_mult * base_channels

            key, block_key = jr.split(key, 2)

            self.blocks.append(Block(channels, new_channels, key=block_key, **block_args))

            channels = new_channels

            self.downs.append(nn.MaxPool3d(2, 2))

    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, list[Array]]:
        skips: list[Array] = []

        for block, down in zip(self.blocks, self.downs):
            x = block(x)

            skips.append(x)

            x = down(x)

        return x, skips


class UnetUp(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    blocks: list[Block]
    ups: list[Upsample3d]

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

        channels = base_channels * channel_mults[-1]

        self.blocks = []
        self.ups = []

        for channel_mult in list(reversed(channel_mults))[1:]:
            new_channels = channel_mult * base_channels

            key, block_key, up_key = jr.split(key, 3)

            self.ups.append(Upsample3d(channels, channels, key=up_key))

            self.blocks.append(Block(2 * channels, new_channels, key=block_key, **block_args))

            channels = new_channels

    def __call__(
        self, x: Array, skips: list[Array], *, key: Optional[PRNGKeyArray] = None
    ) -> Array:

        skips = skips.copy()

        for up, block in zip(self.ups, self.blocks):
            x = up(x)

            skip = skips.pop()

            x = jnp.concat([skip, x], axis=0)

            x = block(x)

        return x


class UnetModule(eqx.Module):
    down: UnetDown
    middle: Block
    up: UnetUp

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        *,
        key: PRNGKeyArray,
        block_args: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        down_key, middle_key, up_key = jr.split(key, 3)

        self.down = UnetDown(base_channels, channel_mults, key=down_key, block_args=block_args)

        middle_channels = base_channels * channel_mults[-1]

        self.middle = Block(middle_channels, middle_channels, key=middle_key, **(block_args or {}))

        self.up = UnetUp(base_channels, channel_mults, key=up_key, block_args=block_args)

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        x, skips = self.down(x)

        x = self.middle(x)

        x = self.up(x, skips)

        return x
