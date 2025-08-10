from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jr
from chex import assert_axis_dimension, assert_equal_shape, assert_rank, assert_shape

from hyper_lap.modules import SinusoidalPositionEmbeddings
from hyper_lap.modules.attention import Encoder


class PatchEncoder(eqx.Module):
    d_model: int = eqx.field(static=True)

    channels: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)

    patch_proj: nn.Linear

    pos_embs: SinusoidalPositionEmbeddings

    def __init__(self, channels: int, patch_size: int, d_model: int, *, key: PRNGKeyArray):
        super().__init__()

        self.d_model = d_model

        self.channels = channels
        self.patch_size = patch_size

        self.patch_proj = nn.Linear(channels * patch_size**2, d_model, key=key)

        self.pos_embs = SinusoidalPositionEmbeddings(d_model)

    def __call__(self, img: Float[Array, "c h w"]) -> Float[Array, "n_seq d_model"]:
        assert_rank(img, 3)

        c, h, w = img.shape

        assert c == self.channels, "channels doesn't match expected number of channels"

        assert h % self.patch_size == 0, "image size must be divisible by patch size"
        assert w % self.patch_size == 0, "image size must be divisible by patch size"

        # split image into patches along h and w axes
        x = img.reshape(
            c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size
        )

        # collect spatial axes at the front, channels and patch_sizes at the back
        x = x.transpose(1, 3, 0, 2, 4)

        # squash channels and patch_sizes together
        x = x.reshape(-1, c * self.patch_size**2)

        assert_shape(
            x, [h // self.patch_size * w // self.patch_size, self.channels * self.patch_size**2]
        )

        x = jax.vmap(self.patch_proj)(x)

        assert_shape(x, [h // self.patch_size * w // self.patch_size, self.d_model])

        pos_embeddings = self.pos_embs(x.shape[0])

        assert_equal_shape([pos_embeddings, x])

        x += pos_embeddings

        return x


class PatchDecoder(eqx.Module):
    h: int = eqx.field(static=True)
    w: int = eqx.field(static=True)

    d_model: int = eqx.field(static=True)

    channels: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)

    patch_unproj: nn.Linear

    def __init__(
        self, h: int, w: int, channels: int, patch_size: int, d_model: int, *, key: PRNGKeyArray
    ):
        super().__init__()

        self.h = h
        self.w = w

        self.d_model = d_model

        self.channels = channels
        self.patch_size = patch_size

        self.patch_unproj = nn.Linear(d_model, channels * patch_size**2, key=key)

    def __call__(self, x: Float[Array, "n_seq d_model"]) -> Float[Array, "c h w"]:
        assert_rank(x, 2)

        n_seq = x.shape[0]

        assert n_seq == (self.w // self.patch_size) * (self.h // self.patch_size)

        assert_axis_dimension(x, 1, self.d_model)

        x = jax.vmap(self.patch_unproj)(x)

        assert_shape(x, (n_seq, self.channels * self.patch_size**2))

        x = x.reshape(
            self.h // self.patch_size,
            self.w // self.patch_size,
            self.channels,
            self.patch_size,
            self.patch_size,
        )

        x = x.transpose(2, 0, 3, 1, 4)

        x = x.reshape(self.channels, self.h, self.w)

        return x


class VitSegmentator(eqx.Module):
    h: int = eqx.field(static=True)
    w: int = eqx.field(static=True)

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    patch_encoder: PatchEncoder
    patch_decoder: PatchDecoder

    transformer: Encoder

    def __init__(
        self,
        image_size: tuple[int, int] | int,
        patch_size: int,
        d_model: int,
        depth: int,
        *,
        num_heads: int = 8,
        dim_head: int | None = None,
        in_channels: int = 3,
        out_channels: int = 2,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if isinstance(image_size, int):
            h = image_size
            w = image_size
        elif isinstance(image_size, tuple):
            h, w = image_size
        else:
            raise ValueError(f"invalid image size {image_size}")

        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(f"patch size {patch_size} doesn't divide image dimensions ({h},{w})")

        if dim_head is None:
            dim_head = 2 * d_model // num_heads

        self.h = h
        self.w = w

        self.in_channels = in_channels
        self.out_channels = out_channels

        encoder_key, transformer_key = jr.split(key)

        self.patch_encoder = PatchEncoder(in_channels, patch_size, d_model, key=encoder_key)
        self.patch_decoder = PatchDecoder(h, w, out_channels, patch_size, d_model, key=encoder_key)

        self.transformer = Encoder(depth, d_model, num_heads, dim_head, key=transformer_key)

    def __call__(self, img: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        assert_shape(img, (self.in_channels, self.h, self.w))

        x = self.patch_encoder(img)

        x = self.transformer(x)

        y = self.patch_decoder(x)

        return y
