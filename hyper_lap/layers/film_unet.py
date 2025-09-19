from jaxtyping import Array, Float, PRNGKeyArray
from typing import Any, Optional, Sequence

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr

from .activations import SiLU
from .conv import ConvNormAct, WeightStandardizedConv2d
from .upsample import BilinearUpsample2d


class FilmProjection(eqx.Module):
    first: nn.Linear
    second: nn.Linear

    def __init__(self, emb_size: int, channels: int, *, key: PRNGKeyArray):
        super().__init__()

        first_key, second_key = jr.split(key)

        self.first = nn.Linear(emb_size, 2 * channels, key=first_key)

        self.second = nn.Linear(2 * channels, 2 * channels, key=second_key)

    def __call__(self, emb: Float[Array, " emb_size"]) -> Float[Array, "2 channels 1 1"]:
        x = self.first(emb)
        x = jax.nn.silu(x)
        x = self.second(x)

        x = x.reshape(2, -1, 1, 1)

        return x


class ConvNormFilmAct(eqx.Module):
    conv: nn.Conv2d | WeightStandardizedConv2d
    norm: nn.GroupNorm
    film_proj: FilmProjection
    act: SiLU

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        emb_size: int,
        *,
        groups=8,
        use_weight_standardized_conv: bool,
        key: PRNGKeyArray,
    ):
        super().__init__()

        conv_key, proj_key = jr.split(key)

        if use_weight_standardized_conv:
            self.conv = WeightStandardizedConv2d(
                in_channels, out_channels, kernel_size, padding="SAME", use_bias=False, key=conv_key
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding="SAME", use_bias=False, key=conv_key
            )

        # self.norm = nn.BatchNorm2d(out_channels, "batch")
        self.norm = nn.GroupNorm(groups, out_channels)

        self.film_proj = FilmProjection(emb_size, out_channels, key=proj_key)

        self.act = SiLU()

    def __call__(
        self, x: Float[Array, "c h w d"], emb: Float[Array, " emb_size"]
    ) -> Float[Array, "c h w d"]:
        x = self.conv(x)

        x = self.norm(x)

        scale, shift = self.film_proj(emb)

        x = (scale + 1) * x + shift

        x = self.act(x)

        return x


class FilmResBlock(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    kernel_size: int = eqx.field(static=True)

    groups: int = eqx.field(static=True)
    n_convs: int = eqx.field(static=True)

    use_weight_standardized_conv: bool = eqx.field(static=True)

    emb_size: int = eqx.field(static=True)

    film_cna: ConvNormFilmAct

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
        emb_size: int,
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

        self.emb_size = emb_size

        res_conv_key, *keys = jr.split(key, n_convs + 1)

        self.film_cna = ConvNormFilmAct(
            in_channels,
            out_channels,
            kernel_size,
            emb_size,
            groups=groups,
            use_weight_standardized_conv=use_weight_standardized_conv,
            key=keys[0],
        )

        self.layers = [
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

    def __call__(self, x: Float[Array, "c h w d"], cond_emb: Array) -> Float[Array, "c h w d"]:
        res = x

        x = self.film_cna(x, cond_emb)

        for layer in self.layers:
            x = layer(x)

        x += self.res_conv(res)

        return x


class FilmUnetDown(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    blocks: list[FilmResBlock]
    downs: list[nn.MaxPool2d]

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        *,
        emb_size: int,
        key: PRNGKeyArray,
        block_args: Optional[dict[str, Any]] = None,
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

            self.blocks.append(
                FilmResBlock(channels, new_channels, emb_size=emb_size, key=block_key, **block_args)
            )

            channels = new_channels

            self.downs.append(nn.MaxPool2d(2, 2))

    def __call__(self, x: Array, cond: Array) -> tuple[Array, list[Array]]:
        skips: list[Array] = []

        for block, down in zip(self.blocks, self.downs):
            x = block(x, cond)

            skips.append(x)

            c, h, w = x.shape

            assert h % 2 == 0 and w % 2 == 0, (
                f"spatial dims of shape {x.shape} are not divisible by 2"
            )

            x = down(x)

        return x, skips


class FilmUnetUp(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    ups: list[BilinearUpsample2d]
    blocks: list[FilmResBlock]

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        *,
        emb_size: int,
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

            key, block_key = jr.split(key)

            self.ups.append(BilinearUpsample2d())

            self.blocks.append(
                FilmResBlock(
                    2 * channels, new_channels, emb_size=emb_size, key=block_key, **block_args
                )
            )

            channels = new_channels

    def __call__(self, x: Array, skips: list[Array], cond: Array) -> Array:
        skips = skips.copy()

        for up, block in zip(self.ups, self.blocks):
            x = up(x)

            skip = skips.pop()

            x = jnp.concat([skip, x], axis=0)

            x = block(x, cond)

        return x


class FilmUnetModule(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    emb_size: int = eqx.field(static=True)

    down: FilmUnetDown
    middle: FilmResBlock
    up: FilmUnetUp

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        emb_size: int,
        *,
        key: PRNGKeyArray,
        block_args: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        self.emb_size = emb_size

        down_key, middle_key, up_key = jr.split(key, 3)

        self.down = FilmUnetDown(
            base_channels, channel_mults, emb_size=emb_size, key=down_key, block_args=block_args
        )

        middle_channels = base_channels * channel_mults[-1]

        self.middle = FilmResBlock(
            middle_channels,
            middle_channels,
            emb_size=emb_size,
            key=middle_key,
            **(block_args or {}),
        )

        self.up = FilmUnetUp(
            base_channels, channel_mults, emb_size=emb_size, key=up_key, block_args=block_args
        )

    def __call__(self, x: Array, cond: Array) -> Array:
        c, h, w = x.shape

        down_factor = 2 ** len(self.channel_mults)

        assert h % down_factor == 0, (
            f"spatial dims must be divisible by {down_factor}, but shape is {x.shape}"
        )
        assert w % down_factor == 0, (
            f"spatial dims must be divisible by {down_factor}, but shape is {x.shape}"
        )

        x, skips = self.down(x, cond)

        x = self.middle(x, cond)

        x = self.up(x, skips, cond)

        return x
