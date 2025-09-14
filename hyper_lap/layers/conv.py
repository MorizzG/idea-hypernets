from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional, Sequence, Union

import equinox as eqx
import equinox.nn as nn
import jax

from .activations import ReLU, SiLU


class WeightStandardizedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = (1, 1),
        padding: Union[str, int, Sequence[int], Sequence[tuple[int, int]]] = (0, 0),
        dilation: Union[int, Sequence[int]] = (1, 1),
        groups: int = 1,
        use_bias: bool = True,
        padding_mode: str = "ZEROS",
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            padding_mode=padding_mode,
            dtype=dtype,
            key=key,
        )

    def _super_forward(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        return super().__call__(x, key=key)

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        weight = self.weight

        mean = weight.mean()
        var = weight.var()

        weight_normed = (weight - mean) * jax.lax.rsqrt(var + 1e-5)

        self_normed = eqx.tree_at(lambda me: me.weight, self, weight_normed)

        return self_normed._super_forward(x, key=key)


class ConvNormAct(eqx.Module):
    conv: nn.Conv2d | WeightStandardizedConv2d
    # norm: nn.BatchNorm2d
    norm: nn.GroupNorm
    act: ReLU | SiLU

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 8,
        *,
        use_weight_standardized_conv: bool = False,
        key: PRNGKeyArray,
    ):
        super().__init__()

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

    def __call__(self, x: Float[Array, "c h w d"]) -> Float[Array, "c h w d"]:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x
