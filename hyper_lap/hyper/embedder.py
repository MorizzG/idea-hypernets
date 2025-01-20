from jaxtyping import Array, Float, Integer, PRNGKeyArray
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from chex import assert_shape

from hyper_lap.modules.convnext import ConvNeXt
from hyper_lap.modules.embedder import ClipEmbedder
from hyper_lap.modules.resnet import ResNet
from hyper_lap.modules.vit import ViT


class InputEmbedder(eqx.Module):
    emb_size: int

    embedder: ViT | ConvNeXt | ResNet | ClipEmbedder

    def __init__(
        self,
        emb_size: int,
        *,
        kind: Literal["vit", "convnext", "resnet", "clip"] = "resnet",
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.emb_size = emb_size

        if kind == "vit":
            self.embedder = ViT(512, 3, 16, 6, 8, 64, emb_size, key=key)
        elif kind == "convnext":
            self.embedder = ConvNeXt(emb_size, 96, in_channels=3, depths=[3, 3, 9, 3], key=key)
        elif kind == "resnet":
            self.embedder = ResNet(emb_size, in_channels=3, depths=[3, 4, 6, 3], key=key)
        elif kind == "clip":
            self.embedder = ClipEmbedder(emb_size, key=key)
        else:
            raise ValueError(f"Unknown embedder: {kind}")

    def __call__(
        self, image: Float[Array, "1 h w"], label: Integer[Array, "h w"]
    ) -> Float[Array, "*"]:
        c, h, w = image.shape

        assert c == 1
        assert_shape(label, (h, w))

        labels_onehot = jnp.zeros([2, h, w])
        labels_onehot = labels_onehot.at[label].set(1)

        combined = jnp.concatenate([image, labels_onehot], axis=0)

        assert_shape(combined, (3, h, w))

        emb = self.embedder(combined)

        return emb
