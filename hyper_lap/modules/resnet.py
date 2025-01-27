from jaxtyping import Array, Float, PRNGKeyArray
from typing import Literal

import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
import jax.random as jr

from hyper_lap.modules._util import SiLU


class BasicBlock(eqx.Module):
    act: SiLU

    conv1: nn.Conv2d
    norm1: nn.GroupNorm

    conv2: nn.Conv2d
    norm2: nn.GroupNorm

    def __init__(self, in_channels: int, out_channels: int, *, key: PRNGKeyArray):
        super().__init__()

        key1, key2 = jr.split(key)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding="SAME", use_bias=False, key=key1
        )
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.act = SiLU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding="SAME", use_bias=False, key=key2
        )
        self.norm2 = nn.GroupNorm(8, out_channels)

    def __call__(
        self, x: Float[Array, "c h w"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "c h w"]:
        res = x

        x = self.conv1(x)
        x = self.norm1(x)

        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x += res

        return x


class BottleneckBlock(eqx.Module):
    act: SiLU

    conv1: nn.Conv2d
    norm1: nn.GroupNorm

    conv2: nn.Conv2d
    norm2: nn.GroupNorm

    conv3: nn.Conv2d
    norm3: nn.GroupNorm

    def __init__(self, in_channels: int, out_channels: int, *, key: PRNGKeyArray):
        super().__init__()

        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4")

        self.act = SiLU()

        compressed_channels = out_channels // 4

        key1, key2, key3 = jr.split(key, 3)

        self.conv1 = nn.Conv2d(in_channels, compressed_channels, 1, use_bias=False, key=key1)
        self.norm1 = nn.GroupNorm(8, compressed_channels)

        self.conv2 = nn.Conv2d(
            compressed_channels, compressed_channels, 3, padding="SAME", use_bias=False, key=key2
        )
        self.norm2 = nn.GroupNorm(8, compressed_channels)

        self.conv3 = nn.Conv2d(compressed_channels, out_channels, 1, use_bias=False, key=key3)
        self.norm3 = nn.GroupNorm(8, out_channels)

    def __call__(
        self, x: Float[Array, "c h w"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "c h w"]:
        res = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x += res

        return x


type BlockKind = Literal["basic", "bottleneck"]


class ResNet(eqx.Module):
    # act: SiLU

    # init_conv: nn.Conv2d
    # init_norm: nn.GroupNorm

    downs: list[nn.Sequential]
    layers: list[nn.Sequential]

    head: nn.Sequential

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        depths: list[int] | None = None,
        block_kind: BlockKind = "bottleneck",
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if depths is None:
            depths = [3, 4, 6, 3]

        if block_kind not in ["basic", "bottleneck"]:
            raise ValueError("block_kind must be 'basic' or 'bottleneck'")

        match block_kind:
            case "basic":
                channels = 64
            case "bottleneck":
                channels = 4 * 64
            case _:
                assert False

        init_conv_key, key = jr.split(key)

        # self.init_conv = nn.Conv2d(
        #     in_channels, channels, 7, stride=2, padding="SAME", use_bias=False, key=init_conv_key
        # )
        # self.init_norm = nn.GroupNorm(8, channels)

        init_conv = nn.Conv2d(
            in_channels,
            channels,
            7,
            stride=2,
            padding="SAME",
            use_bias=False,
            key=init_conv_key,
        )
        init_norm = nn.GroupNorm(8, channels)

        self.downs = [nn.Sequential([init_conv, init_norm, SiLU()])]

        def _make_layer(
            channels: int,
            depth: int,
            *,
            key: PRNGKeyArray,
        ) -> nn.Sequential:
            keys = jr.split(key, depth)

            match block_kind:
                case "basic":
                    layer = nn.Sequential(
                        [BasicBlock(channels, channels, key=keys[0])]
                        + [BasicBlock(channels, channels, key=key) for key in keys[1:]]
                    )
                case "bottleneck":
                    layer = nn.Sequential(
                        [BottleneckBlock(channels, channels, key=keys[0])]
                        + [BottleneckBlock(channels, channels, key=key) for key in keys[1:]]
                    )
                case _:
                    raise ValueError(f"invalid block kind {block_kind}")

            return layer

        key, layer_key = jr.split(key)

        self.layers = [_make_layer(channels, depths[0], key=layer_key)]

        for depth in depths[1:]:
            key, down_key, layer_key = jr.split(key, 3)

            self.downs.append(
                nn.Sequential(
                    [nn.Conv2d(channels, 2 * channels, 2, stride=2, padding="SAME", key=down_key)]
                )
            )

            channels = 2 * channels
            self.layers.append(_make_layer(channels, depth, key=layer_key))

        head_key1, head_key2 = jr.split(key)

        self.head = nn.Sequential(
            [
                nn.Linear(channels, 4 * num_classes, key=head_key1),
                SiLU(),
                nn.Linear(4 * num_classes, num_classes, key=head_key2),
            ]
        )

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, " d"]:
        for down, layer in zip(self.downs, self.layers):
            x = down(x)
            x = layer(x)

        x = jnp.mean(x, axis=(1, 2))

        x = self.head(x)

        return x
