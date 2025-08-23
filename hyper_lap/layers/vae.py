from jaxtyping import Array, Float, PRNGKeyArray
from typing import Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import dataclass

from .activations import SiLU


class ResnetBlock(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    act: SiLU

    norm1: nn.GroupNorm
    conv1: nn.Conv2d

    norm2: nn.GroupNorm
    conv2: nn.Conv2d

    conv_shortcut: nn.Conv2d | None

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        groups: int,  # = 32,
        eps: float,  # = 1e-6,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        eps = 1e-6
        groups = 32

        conv1_key, conv2_key, shortcut_key = jr.split(key, 3)

        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, channelwise_affine=True)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="SAME", key=conv1_key
        )

        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="SAME", key=conv2_key
        )

        self.act = SiLU()

        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, 1, use_bias=True, key=shortcut_key
            )
        else:
            self.conv_shortcut = None

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c_out h_out w_out"]:
        res = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            res = self.conv_shortcut(res)

        x += res

        return x


class Attention(eqx.Module):
    group_norm: nn.GroupNorm | None

    to_q: nn.Linear
    to_k: nn.Linear
    to_v: nn.Linear

    to_out: nn.Linear

    def __init__(
        self,
        dim_in: int,
        heads: int,  # = 8,
        dim_head: int,  # = 64,
        bias: bool,  # = False,
        norm_num_groups: Optional[int],  # = None,
        eps: float,  # = 1e-5,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        inner_dim = dim_head * heads

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(norm_num_groups, dim_in, eps=1e-6)
        else:
            self.group_norm = None

        q_key, k_key, v_key, out_key = jr.split(key, 4)

        self.to_q = nn.Linear(dim_in, inner_dim, use_bias=bias, key=q_key)
        self.to_k = nn.Linear(dim_in, inner_dim, use_bias=bias, key=k_key)
        self.to_v = nn.Linear(dim_in, inner_dim, use_bias=bias, key=v_key)

        self.to_out = nn.Linear(inner_dim, dim_in, use_bias=True, key=out_key)

    def __call__(self, x: Float[Array, "c h w"]) -> Array:
        res = x

        if self.group_norm is not None:
            x = self.group_norm(x)

        c, h, w = x.shape

        x = x.reshape(c, h * w).transpose(1, 0)

        q = jax.vmap(self.to_q)(x)[:, None, :]
        k = jax.vmap(self.to_k)(x)[:, None, :]
        v = jax.vmap(self.to_v)(x)[:, None, :]

        x = jax.nn.dot_product_attention(q, k, v)

        x = jnp.squeeze(x, axis=1)

        x = jax.vmap(self.to_out)(x)

        x = x.transpose(1, 0).reshape(c, h, w)

        x += res

        return x


class UnetMidBlock2d(eqx.Module):
    attentions: list

    resnets: list[ResnetBlock]

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        attention_head_dim: int = 1,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        key, *attn_keys = jr.split(key, num_layers + 1)
        resnet_keys = jr.split(key, num_layers + 1)

        self.attentions = [
            Attention(
                in_channels,
                heads=in_channels // attention_head_dim,
                dim_head=attention_head_dim,
                eps=resnet_eps,
                norm_num_groups=resnet_groups,
                bias=True,
                key=key,
            )
            for key in attn_keys
        ]

        self.resnets = [
            ResnetBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                key=key,
            )
            for key in resnet_keys
        ]

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c_out h_out w_out"]:
        x = self.resnets[0](x)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)

            x = resnet(x)

        return x


class DownEncoderBlock2D(eqx.Module):
    resnets: list[ResnetBlock]

    downsample: nn.Conv2d | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        resnet_eps: float,
        resnet_groups: int,
        add_downsample: bool,  # = True,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        downsample_key, *keys = jr.split(key, num_layers + 1)

        self.resnets = [
            ResnetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                key=keys[0],
            )
        ]

        for key in keys[1:]:
            self.resnets.append(
                ResnetBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    key=key,
                )
            )

        if add_downsample:
            self.downsample = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                use_bias=True,
                key=downsample_key,
            )
        else:
            self.downsample = None

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c_out h_out w_out"]:
        for resnet in self.resnets:
            x = resnet(x)

        if self.downsample is not None:
            x = jnp.pad(x, ((0, 0), (0, 1), (0, 1)), mode="constant", constant_values=0)

            x = self.downsample(x)

        return x


class UpDecoderBlock2D(eqx.Module):
    resnets: list[ResnetBlock]

    upsample: nn.Conv2d | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_groups: int = 32,
        add_upsample: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        upsample_key, *keys = jr.split(key, num_layers + 1)

        self.resnets = [
            ResnetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                key=keys[0],
            )
        ]

        for key in keys[1:]:
            self.resnets.append(
                ResnetBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    key=key,
                )
            )

        if add_upsample:
            self.upsample = nn.Conv2d(
                out_channels, out_channels, 3, padding="SAME", key=upsample_key
            )
        else:
            self.upsample = None

    def __call__(self, x: Array) -> Array:
        for resnet in self.resnets:
            x = resnet(x)

        if self.upsample is not None:
            c, h, w = x.shape

            x = jax.image.resize(x, (c, 2 * h, 2 * w), method="nearest")

            x = self.upsample(x)

        return x


class Encoder(eqx.Module):
    layers_per_block: int = eqx.field(static=True)

    conv_in: nn.Conv2d

    down_blocks: list[DownEncoderBlock2D]

    mid_block: UnetMidBlock2d

    conv_norm_out: nn.GroupNorm
    conv_act: SiLU
    conv_out: nn.Conv2d

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.layers_per_block = layers_per_block

        block_out_channels = (128, 256, 512, 512)

        key, conv_in_key = jr.split(key)

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=1, key=conv_in_key
        )

        channels = block_out_channels[0]

        self.down_blocks = []

        for i, next_channels in enumerate(block_out_channels):
            is_final_block = i == len(block_out_channels) - 1

            key, down_key = jr.split(key)

            self.down_blocks.append(
                DownEncoderBlock2D(
                    num_layers=layers_per_block,
                    in_channels=channels,
                    out_channels=next_channels,
                    add_downsample=not is_final_block,
                    resnet_eps=1e-6,
                    resnet_groups=norm_num_groups,
                    key=down_key,
                )
            )

            channels = next_channels

        conv_key, mid_key = jr.split(key)

        self.mid_block = UnetMidBlock2d(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            key=mid_key,
        )

        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[-1])
        self.conv_act = SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[-1], 2 * out_channels, 3, padding="SAME", key=conv_key
        )

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c_out h_out w_out"]:
        x = self.conv_in(x)

        for down_block in self.down_blocks:
            x = down_block(x)

        x = self.mid_block(x)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x


class Decoder(eqx.Module):
    layers_per_block: int = eqx.field(static=True)

    conv_in: nn.Conv2d

    up_blocks: list[UpDecoderBlock2D]

    mid_block: UnetMidBlock2d

    conv_norm_out: nn.GroupNorm
    conv_act: SiLU
    conv_out: nn.Conv2d

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.layers_per_block = layers_per_block

        key, conv_in_key = jr.split(key)

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[-1], 3, padding="SAME", key=conv_in_key
        )

        key, mid_key = jr.split(key)

        self.mid_block = UnetMidBlock2d(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            key=mid_key,
        )

        channels = block_out_channels[-1]

        self.up_blocks = []

        for i, next_channels in enumerate(reversed(block_out_channels)):
            is_final_block = i == len(block_out_channels) - 1

            key, up_key = jr.split(key)

            self.up_blocks.append(
                UpDecoderBlock2D(
                    num_layers=layers_per_block + 1,
                    in_channels=channels,
                    out_channels=next_channels,
                    add_upsample=not is_final_block,
                    resnet_eps=1e-6,
                    resnet_groups=norm_num_groups,
                    key=up_key,
                )
            )

            channels = next_channels

        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[0])
        self.conv_act = SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding="SAME", key=key)

    def __call__(self, x: Array) -> Array:
        x = self.conv_in(x)

        x = self.mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x


@dataclass
class DiagonalGaussianDistribution:
    mean: Array
    std: Array

    def sample(self, key: PRNGKeyArray) -> Array:
        eps = jr.normal(key, self.mean.shape)

        return self.mean + self.std * eps


class VAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder

    quant_conv: nn.Conv2d
    post_quant_conv: nn.Conv2d

    def __init__(self, *, key: PRNGKeyArray):
        super().__init__()

        in_channels: int = 3
        out_channels: int = 3
        # down_block_types: tuple[str, ...] = 4 * ("DownEncoderBlock2D",)
        # up_block_types: tuple[str, ...] = 4 * ("UpDecoderBlock2D",)
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
        layers_per_block: int = 2
        # act_fn: str = "silu"
        latent_channels: int = 4
        norm_num_groups: int = 32
        # sample_size: int = 1024
        # scaling_factor: float = 0.13025
        # shift_factor: Optional[float] = None
        # latents_mean: Optional[tuple[float, ...]] = None
        # latents_std: Optional[tuple[float, ...]] = None
        # force_upcast: float = True
        # use_quant_conv: bool = True
        # use_post_quant_conv: bool = True
        # mid_block_add_attention: bool = True

        encoder_key, decoder_key, q_key, pq_key = jr.split(key, 4)

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            key=encoder_key,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            # mid_block_add_attention=mid_block_add_attention,
            key=decoder_key,
        )

        self.quant_conv = nn.Conv2d(
            2 * latent_channels, 2 * latent_channels, kernel_size=1, key=q_key
        )

        self.post_quant_conv = nn.Conv2d(
            latent_channels, latent_channels, kernel_size=1, key=pq_key
        )

    def encode(self, x: Float[Array, "c h w"]) -> DiagonalGaussianDistribution:
        enc = self.encoder(x)

        if self.quant_conv is not None:
            enc = self.quant_conv(enc)

        c, h, w = enc.shape

        mean, logvar = enc.reshape(2, c // 2, h, w)

        logvar = jnp.clip(logvar, -30.0, 20.0)

        std = jnp.exp(0.5 * logvar)

        posterior = DiagonalGaussianDistribution(mean=mean, std=std)

        return posterior

    def decode(self, z: Array) -> Float[Array, "c h w"]:
        y = z

        if self.post_quant_conv is not None:
            y = self.post_quant_conv(y)

        y = self.decoder(y)

        return y

    def __call__(
        self,
        x: Float[Array, "c h w"],
        deterministic: bool = False,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "c h w"]:
        if not deterministic and key is None:
            raise ValueError("Need to pass key if using non-deterministic mode")

        dist = self.encode(x)

        if deterministic:
            z = dist.mean
        else:
            assert key is not None

            z = dist.sample(key)

        y = self.decode(z)

        return y
