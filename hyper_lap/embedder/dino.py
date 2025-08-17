from jaxtyping import Array, Float, Integer, PRNGKeyArray
from typing import Literal

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from chex import assert_axis_dimension, assert_rank, assert_shape
from dinov3_equinox import DinoVisionTransformer, dinov3_vitl16


class DinoEmbedder(eqx.Module):
    emb_size: int = eqx.field(static=True)

    dino: DinoVisionTransformer

    proj: nn.Linear

    def __init__(
        self,
        emb_size: int,
        *,
        dino: Literal["vitl16"] = "vitl16",
        weights_path: str,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.emb_size = emb_size

        match dino:
            case "vitl16":
                self.dino = dinov3_vitl16(key=jr.PRNGKey(0)).load_weights(weights_path)

            case _:
                raise ValueError(f"invalid dino {dino}")

        self.proj = nn.Linear(3 * 2 * self.dino.embed_dim, emb_size, key=key)

    def __call__(
        self,
        image: Float[Array, "3 h w"],
        label: Integer[Array, "h w"],
        _dataset_idx: Integer[Array, ""],
    ) -> Float[Array, " emb_size"]:
        c, h, w = image.shape

        assert c == 3
        assert_shape(label, [h, w])

        pos_masked_image = (label != 0) * image
        neg_masked_image = (label == 0) * image

        x = jnp.stack([image, pos_masked_image, neg_masked_image])

        assert x.shape[:2] == (3, 3)

        assert_shape(x, (3, 3, h, w))

        dino_out = jax.vmap(self.dino)(x)

        cls_token = dino_out["cls_token"]
        patch_tokens = dino_out["patch_tokens"]

        embed_dim = self.dino.embed_dim

        assert_shape(cls_token, (3, 1, embed_dim))
        assert_rank(patch_tokens, 3)
        assert_axis_dimension(patch_tokens, 0, 3)
        assert_axis_dimension(patch_tokens, 2, embed_dim)

        # same as classifier in DINOv3: mean over patch tokens, concat with cls token
        x = jnp.stack([cls_token[:, 0, :], patch_tokens.mean(axis=1)], axis=1)

        assert_shape(x, (3, 2, embed_dim))

        x = x.ravel()

        x = jax.lax.stop_gradient(x)

        x = self.proj(x)

        return x
