from jaxtyping import Array, Float, PRNGKeyArray

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import assert_axis_dimension, assert_equal_shape, assert_rank, assert_shape


class SinusoidalPositionEmbeddings(eqx.Module):
    d_model: int = eqx.field(static=True)

    freqs: Array

    def __init__(self, d_model: int):
        super().__init__()

        assert d_model % 2 == 0, "dim must be divisible by 2"

        self.d_model = d_model

        half_dim = d_model // 2

        freqs = jnp.log(10_000) / (half_dim - 1)
        freqs = jnp.exp(-jnp.arange(half_dim) * freqs)

        self.freqs = freqs

    def __call__(self, n: int) -> Float[Array, "n d_model"]:
        t = jnp.arange(n)

        # embeddings = t[:, None] * self.freqs[None, :]
        embeddings = jnp.outer(t, self.freqs)

        embeddings = jnp.concat([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)

        assert_shape(embeddings, (n, self.d_model))

        embeddings = jax.lax.stop_gradient(embeddings)

        return embeddings


class FeedForward(eqx.Module):
    linear1: nn.Linear
    linear2: nn.Linear

    def __init__(self, dim_model: int, dim_hidden: int, *, key: PRNGKeyArray):
        super().__init__()

        key1, key2 = jr.split(key)

        self.linear1 = nn.Linear(dim_model, dim_hidden, key=key1)

        self.linear2 = nn.Linear(dim_hidden, dim_model, key=key2)

    def __call__(self, x: Float[Array, " d"]) -> Float[Array, " d"]:
        x = self.linear1(x)

        x = jax.nn.silu(x)

        x = self.linear2(x)

        return x


class Attention(eqx.Module):
    d_model: int = eqx.field(static=True)

    num_heads: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)

    query: nn.Linear
    key: nn.Linear
    value: nn.Linear

    out_proj: nn.Linear

    # scale_factor: float

    def __init__(self, d_model: int, num_heads: int, d_head: int, *, key: PRNGKeyArray):
        super().__init__()

        self.d_model = d_model

        self.num_heads = num_heads
        self.dim_head = d_head

        # self.scale_factor = dim_head**-0.5

        hidden_dim = num_heads * d_head

        q_key, k_key, v_key, out_key = jr.split(key, 4)

        self.query = nn.Linear(d_model, hidden_dim, use_bias=False, key=q_key)
        self.key = nn.Linear(d_model, hidden_dim, use_bias=False, key=k_key)
        self.value = nn.Linear(d_model, hidden_dim, use_bias=False, key=v_key)

        self.out_proj = nn.Linear(hidden_dim, d_model, use_bias=False, key=out_key)

    def transpose_for_scores(self, x: Float[Array, "n h"]) -> Float[Array, "n_h n d_h"]:
        n_seq, hidden_dim = x.shape

        assert hidden_dim == self.num_heads * self.dim_head

        return x.reshape(n_seq, self.num_heads, self.dim_head)  # .transpose(1, 0, 2)

    def __call__(
        self, x: Float[Array, "n d"], context: Float[Array, "m d"] | None = None
    ) -> Float[Array, "n d"]:
        assert_rank(x, 2)
        assert_axis_dimension(x, 1, self.d_model)

        if context is None:
            context = x

        assert_rank(context, 2)
        assert_axis_dimension(context, 1, self.d_model)

        n_seq = x.shape[0]
        m_seq = context.shape[0]

        # shape: [num_heads, context, dim_head]1
        q = self.transpose_for_scores(jax.vmap(self.query)(x))
        k = self.transpose_for_scores(jax.vmap(self.key)(context))
        v = self.transpose_for_scores(jax.vmap(self.value)(context))

        assert_shape(q, [n_seq, self.num_heads, self.dim_head])
        assert_shape([k, v], [m_seq, self.num_heads, self.dim_head])

        # # shape: [num_heads, context, context]
        # qk = self.scale_factor * (q @ k.mT)
        #
        # assert_shape(qk, [self.num_heads, n_seq, n_seq])
        #
        # # shape: [num_heads, context, context]
        # attn = jax.nn.softmax(qk, axis=-1)
        #
        # assert_shape(attn, [self.num_heads, n_seq, n_seq])
        #
        # # shape: [num_heads, context, dim_head]
        # out = attn @ v
        #
        # assert_shape(out, [self.num_heads, n_seq, self.dim_head])
        #
        # out = out.transpose(1, 0, 2).reshape(n_seq, -1)
        #
        # assert_shape(out, [n_seq, self.num_heads * self.dim_head])

        out = jax.nn.dot_product_attention(q, k, v)

        assert_shape(out, [n_seq, self.num_heads, self.dim_head])

        # squash heads back to hidden_dim
        out = out.reshape(n_seq, self.num_heads * self.dim_head)

        out = jax.vmap(self.out_proj)(out)

        assert_shape(out, [n_seq, self.d_model])

        return out


class ResAttentionBlock(eqx.Module):
    norm1: nn.LayerNorm
    attention: Attention

    norm2: nn.LayerNorm
    ff: FeedForward

    def __init__(self, d_model: int, num_heads: int, d_head: int, *, key: PRNGKeyArray):
        super().__init__()

        attn_key, ff_key = jr.split(key)

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = Attention(d_model, num_heads, d_head, key=attn_key)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, 4 * d_model, key=ff_key)

    def __call__(self, x: Float[Array, "n d"]) -> Float[Array, "n d"]:
        res = x

        x = jax.vmap(self.norm1)(x)
        x = self.attention(x)

        x += res

        res = x

        x = jax.vmap(self.norm2)(x)
        # vmap over sequence
        x = jax.vmap(self.ff)(x)

        x += res

        return x


class Encoder(eqx.Module):
    res_attn_blocks: list[ResAttentionBlock]

    def __init__(self, depth: int, d_model: int, num_heads: int, d_head: int, *, key: PRNGKeyArray):
        super().__init__()

        keys = jr.split(key, depth)

        self.res_attn_blocks = [
            ResAttentionBlock(d_model, num_heads, d_head, key=key) for key in keys
        ]

    def __call__(self, x: Float[Array, "n d"]) -> Float[Array, "n d"]:
        for res_attn_block in self.res_attn_blocks:
            x = res_attn_block(x)

        return x


class SpatialSelfAttention(eqx.Module):
    num_heads: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)

    scale_factor: float = eqx.field(static=True)

    qkv: nn.Conv2d

    out_proj: nn.Conv2d

    def __init__(self, channels: int, num_heads: int, dim_head: int, *, key: PRNGKeyArray):
        super().__init__()

        self.num_heads = num_heads
        self.dim_head = dim_head

        hidden_dim = num_heads * dim_head

        qkv_key, out_key = jr.split(key)

        self.qkv = nn.Conv2d(channels, 3 * hidden_dim, kernel_size=1, use_bias=False, key=qkv_key)

        self.out_proj = nn.Conv2d(hidden_dim, channels, kernel_size=1, key=out_key)

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        _c, h, w = x.shape

        # shape: [3, num_heads, hw, dim_head]
        q, k, v = self.qkv(x).reshape(3, self.num_heads, self.dim_head, h * w).transpose(0, 3, 1, 2)

        # assert_shape(qkv, [3, h * w, self.num_heads, self.dim_head])

        # q, k, v = qkv[:]

        # # all of shape [num_heads, hw, dim_head]
        # q, k, v = qkv[:]
        #
        # # of shape [num_heads, hw, hw]
        # qk = self.scale_factor * (q @ k.mT)
        #
        # attn = jax.nn.softmax(qk, axis=-1)
        #
        # # shape: [num_heads, hw, dim_head]
        # out = attn @ v
        #
        # out = out.transpose(0, 2, 1).reshape(-1, h, w)

        assert_shape([q, k, v], [h * w, self.num_heads, self.dim_head])

        out = jax.nn.dot_product_attention(q, k, v)

        assert_shape(out, [h * w, self.num_heads, self.dim_head])

        out = out.transpose(1, 2, 0).reshape(-1, h, w)

        out = self.out_proj(out)

        return out


class SpatialCrossAttention(eqx.Module):
    """
    Attention block that enables spatial cross-attention between an image and a context.
    """

    attn: Attention

    def __init__(self, channels: int, num_heads: int, dim_head: int, *, key: PRNGKeyArray):
        super().__init__()

        self.attn = Attention(channels, num_heads, dim_head, key=key)

    def __call__(
        self, img: Float[Array, "c h w"], context: Float[Array, "m d"] | None = None
    ) -> Float[Array, "c h w"]:
        c, h, w = img.shape

        x = img.reshape(c, h * w).transpose(1, 0)

        if context is None:
            context = x

        assert_equal_shape([x, context], dims=1)

        out = self.attn(x, context)

        out = out.transpose(1, 0).reshape(c, h, w)

        assert_equal_shape([img, out])

        return out
