from jaxtyping import Array, Float, Integer, PRNGKeyArray
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from chex import assert_shape

from hyper_lap.modules.embedder import (
    ClipEmbedder,
    ConvNextEmbedder,
    LearnedEmbedding,
    ResNetEmbedder,
    ViTEmbedder,
)


class InputEmbedder(eqx.Module):
    emb_size: int = eqx.field(static=True)

    embedder: ViTEmbedder | ResNetEmbedder | ConvNextEmbedder | ClipEmbedder | LearnedEmbedding

    def __init__(
        self,
        emb_size: int,
        *,
        kind: Literal["vit", "convnext", "resnet", "clip", "learned"] = "resnet",
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.emb_size = emb_size

        if kind == "vit":
            self.embedder = ViTEmbedder(emb_size, key=key)
        elif kind == "convnext":
            self.embedder = ConvNextEmbedder(emb_size, key=key)
        elif kind == "resnet":
            self.embedder = ResNetEmbedder(emb_size, key=key)
        elif kind == "clip":
            self.embedder = ClipEmbedder(emb_size, key=key)
        elif kind == "learned":
            self.embedder = LearnedEmbedding(emb_size, key=key)
        else:
            raise ValueError(f"Unknown embedder: {kind}")

    def __call__(
        self, image: Float[Array, "c h w"], label: Integer[Array, "h w"]
    ) -> Float[Array, "*"]:
        c, h, w = image.shape

        assert c == 1 or c == 3
        assert_shape(label, (h, w))

        if c == 1:
            image = jnp.repeat(image, 3, 0)

        emb = self.embedder(image, label)

        return emb
