from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import assert_equal_shape, assert_rank, assert_shape
from equinox import nn

from .attention import Encoder


def sinudoidal_positional_encoding2d(n: int, m: int, dim_model: int) -> Float[Array, ""]:
    """
    2-dimensional sinusoidal positional encoding, constructed by applying 1d sinusoidal positional
    encodings, in the first direction on the first half of the model dimension and in the second
    direction on the second half.
    """
    assert dim_model % 4 == 0, "dim_model must be divisible by 4"

    t1 = jnp.arange(n)
    t2 = jnp.arange(m)

    quater_dim = dim_model // 4

    base_freqs = jnp.log(10_000) / (quater_dim - 1)
    base_freqs = jnp.exp(-jnp.arange(quater_dim) * base_freqs)

    freqs1 = t1[:, None, None] * base_freqs[None, None, :]
    freqs2 = t2[None, :, None] * base_freqs[None, None, :]

    embeddings1 = jnp.concat([jnp.cos(freqs1), jnp.sin(freqs1)], axis=-1)
    embeddings2 = jnp.concat([jnp.cos(freqs2), jnp.sin(freqs2)], axis=-1)

    embeddings1 = jnp.repeat(embeddings1, m, axis=1)
    embeddings2 = jnp.repeat(embeddings2, n, axis=0)

    embeddings = jnp.concat([embeddings1, embeddings2], axis=-1)

    return embeddings


class PatchEmbedder(eqx.Module):
    dim_model: int = eqx.field(static=True)

    channels: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)

    patch_proj: nn.Linear

    cls_token: Array

    def __init__(
        self, channels: int, patch_size: int, dim_model: int, *, key: PRNGKeyArray
    ) -> None:
        super().__init__()

        self.dim_model = dim_model

        self.channels = channels
        self.patch_size = patch_size

        proj_key, cls_key = jr.split(key)

        self.patch_proj = nn.Linear(channels * patch_size**2, dim_model, key=proj_key)

        self.cls_token = jr.normal(cls_key, [1, dim_model])

    def __call__(self, img: Float[Array, "c h w"]) -> Float[Array, "n d"]:
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
        x = x.reshape(*x.shape[:2], c * self.patch_size**2)

        assert_shape(
            x, [h // self.patch_size, w // self.patch_size, self.channels * self.patch_size**2]
        )

        x = jax.vmap(jax.vmap(self.patch_proj))(x)

        assert_shape(x, [h // self.patch_size, w // self.patch_size, self.dim_model])

        pos_embeddings = sinudoidal_positional_encoding2d(*x.shape)

        assert_equal_shape([pos_embeddings, x])

        x += pos_embeddings

        x = x.reshape(-1, self.dim_model)

        # concatenate class token to the front of the sequence
        x = jnp.concat([self.cls_token, x], axis=0)

        return x


class ViT(eqx.Module):
    dim_model: int = eqx.field(static=True)

    embedder: PatchEmbedder

    encoder: Encoder

    projection: nn.Linear

    def __init__(
        self,
        dim_model: int,
        channels: int,
        patch_size: int = 16,
        depth: int = 6,
        num_heads: int = 8,
        dim_head: int = 64,
        dim_out: int | None = None,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if dim_out is None:
            dim_out = dim_model

        self.dim_model = dim_model

        emb_key, encoder_key, proj_key = jr.split(key, 3)

        self.embedder = PatchEmbedder(channels, patch_size, dim_model, key=emb_key)

        self.encoder = Encoder(depth, dim_model, num_heads, dim_head, key=encoder_key)

        self.projection = nn.Linear(dim_model, dim_out, key=proj_key)

    def __call__(self, img: Float[Array, "c h w"]) -> Float[Array, " d"]:
        x = self.embedder(img)

        assert_rank(x, 2)
        assert x.shape[1] == self.dim_model

        x = self.encoder(x)

        cls_token = x[0]

        assert_shape(cls_token, [self.dim_model])

        out = self.projection(cls_token)

        return out
