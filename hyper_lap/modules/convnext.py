from typing import Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from hyper_lap.modules._util import SiLU


class ConvNextBlock(eqx.Module):
    dw_conv: nn.Conv2d

    norm: nn.LayerNorm

    pw_conv1: nn.Conv2d
    pw_conv2: nn.Conv2d

    def __init__(self, channels: int, *, key: PRNGKeyArray):
        super().__init__()

        dw_key, pw_key1, pw_key2 = jr.split(key, 3)

        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=7, padding="SAME", groups=channels, key=dw_key
        )

        self.norm = nn.LayerNorm(channels, eps=1e-6)

        self.pw_conv1 = nn.Conv2d(channels, 4 * channels, kernel_size=1, key=pw_key1)
        self.pw_conv2 = nn.Conv2d(4 * channels, channels, kernel_size=1, key=pw_key2)

    def __call__(
        self, x: Float[Array, "c h w"], *, key: PRNGKeyArray | None = None
    ) -> Float[Array, "c h w"]:
        res = x

        x = self.dw_conv(x)

        # vmap over H and W
        norm = self.norm
        norm = jax.vmap(norm, in_axes=-1, out_axes=-1)
        norm = jax.vmap(norm, in_axes=-1, out_axes=-1)
        x = norm(x)

        x = self.pw_conv1(x)

        x = jax.nn.swish(x)

        x = self.pw_conv2(x)

        x = x + res

        return x


class ConvNeXt(eqx.Module):
    stem_conv: nn.Conv2d
    stem_norm: nn.LayerNorm
    act: SiLU

    downs: list[nn.Sequential]
    stages: list[nn.Sequential]

    head_norm: nn.LayerNorm
    head: nn.Sequential

    def __init__(
        self,
        num_classes: int,
        base_channels: int,
        depths: Optional[list[int]] = None,
        in_channels: int = 1,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if depths is None:
            depths = [3, 3, 9, 3]

        if len(depths) != 4:
            raise ValueError("depths must have length 4")

        key, stem_key = jr.split(key)

        self.stem_conv = nn.Conv2d(
            in_channels, base_channels, kernel_size=4, stride=4, key=stem_key
        )

        self.stem_norm = nn.LayerNorm(base_channels, eps=1e-6)

        self.act = SiLU()

        # first down sampling is done by stem
        self.downs = [nn.Identity()]

        stage_key0, key = jr.split(key)

        self.stages = [
            nn.Sequential([ConvNextBlock(base_channels, key=stage_key0) for _ in range(depths[0])])
        ]

        channels = base_channels

        for i in range(1, 4):
            down_key, stage_key, key = jr.split(key, 3)

            norm = nn.LayerNorm(channels, eps=1e-6)
            norm = jax.vmap(norm, in_axes=-1, out_axes=-1)
            norm = jax.vmap(norm, in_axes=-1, out_axes=-1)

            down = nn.Conv2d(channels, 2 * channels, kernel_size=2, stride=2, key=down_key)

            channels = 2 * channels

            self.downs.append(nn.Sequential([norm, down]))

            stage = nn.Sequential(
                [ConvNextBlock(channels, key=stage_key) for _ in range(depths[i])]
            )

            self.stages.append(stage)

        self.head_norm = nn.LayerNorm(channels, eps=1e-6)

        # self.head = nn.Linear(channels, out_dim, key=key)

        head_key1, head_key2 = jr.split(key)

        self.head = nn.Sequential(
            [
                nn.Linear(channels, 4 * num_classes, key=head_key1),
                nn.Lambda(fn=jax.nn.swish),
                nn.Linear(4 * num_classes, num_classes, key=head_key2),
            ]
        )

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "d"]:
        c, h, w = x.shape

        assert h % 4 == 0 and w % 4 == 0, f"h and w must be divisible by 4, bu x.shape={x.shape}"

        x = self.stem_conv(x)

        stem_norm = self.stem_norm
        stem_norm = jax.vmap(stem_norm, in_axes=-1, out_axes=-1)
        stem_norm = jax.vmap(stem_norm, in_axes=-1, out_axes=-1)
        x = stem_norm(x)

        x = self.act(x)

        for down, stage in zip(self.downs, self.stages):
            x = down(x)
            x = stage(x)

        # global average pool
        x = x.mean(axis=(1, 2))

        x = self.head_norm(x)
        x = self.head(x)

        return x
