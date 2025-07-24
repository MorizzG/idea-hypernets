from jaxtyping import Array, Float, PRNGKeyArray
from typing import Any, Optional, Sequence

import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr

from .conv import ConvNormAct
from .upsample import BilinearUpsample2d


class Block(eqx.Module):
    """
    Block module for U-Nets.

    Groups n_convs convolutions, with same in_channels as out_channels.
    """

    channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)
    groups: int = eqx.field(static=True)
    n_convs: int = eqx.field(static=True)
    use_weight_standardized_conv: bool = eqx.field(static=True)

    layers: nn.Sequential

    def __init__(
        self,
        channels: int,
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

        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.n_convs = n_convs
        self.use_weight_standardized_conv = use_weight_standardized_conv

        keys = jr.split(key, n_convs)

        layers = [
            ConvNormAct(
                channels,
                channels,
                kernel_size,
                groups=groups,
                use_weight_standardized_conv=use_weight_standardized_conv,
                key=key,
            )
            for key in keys
        ]

        self.layers = nn.Sequential(layers)

    def __call__(
        self, x: Float[Array, "c h w d"], *, key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "c h w d"]:
        res = x

        for layer in self.layers:
            x = layer(x)

        x += res

        return x


class UnetDown(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    blocks: list[Block]
    downs: list[nn.Conv2d]

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

            key, down_key, block_key = jr.split(key, 3)

            self.blocks.append(Block(channels, key=block_key, **block_args))

            self.downs.append(
                nn.Conv2d(channels, new_channels, 2, stride=2, use_bias=False, key=down_key)
            )

            channels = new_channels

    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, list[Array]]:
        skips: list[Array] = []

        for block, down in zip(self.blocks, self.downs):
            x = block(x)

            skips.append(x)

            c, h, w = x.shape

            assert h % 2 == 0 and w % 2 == 0, (
                f"spatial dims of shape {x.shape} are not divisible by 2"
            )

            x = down(x)

        return x, skips


class UnetUp(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    ups: list[nn.ConvTranspose2d]
    blocks: list[Block]

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

        channels = base_channels * channel_mults[-1]

        for channel_mult in list(reversed(channel_mults))[1:]:
            new_channels = channel_mult * base_channels

            key, up_key, block_key = jr.split(key, 3)

            self.ups.append(
                nn.ConvTranspose2d(channels, new_channels, 2, stride=2, use_bias=False, key=up_key)
            )

            channels = 2 * new_channels

            self.blocks.append(Block(channels, key=block_key, **block_args))

    def __call__(
        self, x: Array, skips: list[Array], *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        skips = skips.copy()

        for up, block in zip(self.ups, self.blocks):
            x = up(x)

            skip = skips.pop()

            x = jnp.concat([skip, x], axis=0)

            # x = resample(x)

            x = block(x)

        return x


class UnetModule(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

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

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        down_key, middle_key, up_key = jr.split(key, 3)

        self.down = UnetDown(base_channels, channel_mults, key=down_key, block_args=block_args)

        middle_channels = base_channels * channel_mults[-1]

        self.middle = Block(middle_channels, key=middle_key, **(block_args or {}))

        self.up = UnetUp(base_channels, channel_mults, key=up_key, block_args=block_args)

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        c, h, w = x.shape

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
