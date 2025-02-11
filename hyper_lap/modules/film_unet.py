from jaxtyping import Array, Float, PRNGKeyArray
from typing import Any, Optional, Sequence

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr

from ._util import ReLU, SiLU
from .conv import ConvNormAct, WeightStandardizedConv2d
from .upsample import BilinearUpsample2d


class ConvNormFilmAct(eqx.Module):
    conv: nn.Conv2d | WeightStandardizedConv2d
    # norm: nn.BatchNorm2d
    norm: nn.GroupNorm
    act: ReLU | SiLU

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups=8,
        *,
        use_weight_standardized_conv: bool,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding="SAME", use_bias=False, key=key
        )

        if use_weight_standardized_conv:
            self.conv = WeightStandardizedConv2d(
                in_channels, out_channels, kernel_size, padding="SAME", use_bias=False, key=key
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding="SAME", use_bias=False, key=key
            )

        # self.norm = nn.BatchNorm2d(out_channels, "batch")
        self.norm = nn.GroupNorm(groups, out_channels, channelwise_affine=False)

        self.act = SiLU()

    def __call__(
        self,
        x: Float[Array, "c h w d"],
        scale_shift: Float[Array, "2 c"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "c h w d"]:
        x = self.conv(x)
        x = self.norm(x)

        # expand h and w axes
        scale_shift = jnp.expand_dims(scale_shift, (2, 3))

        scale = scale_shift[0]
        shift = scale_shift[1]

        x = (scale + 1) * x + shift

        x = self.act(x)

        return x


class FilmBlock(eqx.Module):
    emb_size: int = eqx.field(static=True)

    use_res: bool = eqx.field(static=True)

    emb_proj: nn.Linear

    res_conv: Optional[nn.Conv2d]

    film_cna: ConvNormFilmAct

    # layers: nn.Sequential
    layers: list[ConvNormAct]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 8,
        n_convs: int = 2,
        use_res: bool = False,
        *,
        emb_size: int,
        use_weight_standardized_conv: bool,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if n_convs < 1:
            raise ValueError("Must have at least one conv")

        self.emb_size = emb_size

        self.use_res = use_res

        key, emb_proj_key = jr.split(key)

        self.emb_proj = nn.Linear(emb_size, 2 * out_channels, key=emb_proj_key)

        if use_res and in_channels != out_channels:
            key, res_conv_key = jr.split(key)

            self.res_conv = nn.Conv2d(
                in_channels, out_channels, 1, padding="SAME", key=res_conv_key
            )
        else:
            self.res_conv = None

        keys = jr.split(key, n_convs)

        self.film_cna = ConvNormFilmAct(
            in_channels,
            out_channels,
            kernel_size,
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

    def __call__(
        self, x: Float[Array, "c h w d"], cond_emb: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "c h w d"]:
        res = x

        cond_emb = jax.nn.silu(cond_emb)

        scale_shift = self.emb_proj(cond_emb).reshape(2, -1)

        x = self.film_cna(x, scale_shift)

        for layer in self.layers[1:]:
            x = layer(x)

        if self.use_res:
            if self.res_conv is not None:
                res = self.res_conv(res)

            x += res

        return x


class FilmUnetDown(eqx.Module):
    base_channels: int = eqx.field(static=True)
    channel_mults: list[int] = eqx.field(static=True)

    blocks: list[FilmBlock]
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

        self.blocks = []
        self.downs = []

        for channel_mult in channel_mults[1:]:
            new_channels = channel_mult * base_channels

            key, block_key = jr.split(key, 2)

            self.blocks.append(FilmBlock(channels, new_channels, key=block_key, **block_args))

            channels = new_channels

            self.downs.append(nn.MaxPool2d(2, 2))

    def __call__(
        self, x: Array, cond: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, list[Array]]:
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

    blocks: list[FilmBlock]
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

        self.blocks = []
        self.ups = []

        channels = base_channels * channel_mults[-1]

        for channel_mult in list(reversed(channel_mults))[1:]:
            new_channels = channel_mult * base_channels

            key, block_key, up_key = jr.split(key, 3)

            # self.ups.append(ConvUpsample2d(channels, channels, key=up_key))
            self.ups.append(BilinearUpsample2d())

            self.blocks.append(FilmBlock(2 * channels, new_channels, key=block_key, **block_args))

            channels = new_channels

    def __call__(
        self, x: Array, skips: list[Array], cond: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
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
    middle: FilmBlock
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

        if block_args is None:
            block_args = {}

        block_args["emb_size"] = emb_size

        self.base_channels = base_channels
        self.channel_mults = list(channel_mults)

        self.emb_size = emb_size

        down_key, middle_key, up_key = jr.split(key, 3)

        self.down = FilmUnetDown(base_channels, channel_mults, key=down_key, block_args=block_args)

        middle_channels = base_channels * channel_mults[-1]

        self.middle = FilmBlock(middle_channels, middle_channels, key=middle_key, **block_args)

        self.up = FilmUnetUp(base_channels, channel_mults, key=up_key, block_args=block_args)

    def __call__(self, x: Array, cond: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
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
