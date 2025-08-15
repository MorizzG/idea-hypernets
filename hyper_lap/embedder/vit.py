from jaxtyping import Array, Float, Integer, PRNGKeyArray
from typing import Literal

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import transformers
from chex import assert_shape
from transformers.models.clip import FlaxCLIPVisionModel

from hyper_lap.modules.convnext import ConvNeXt
from hyper_lap.modules.resnet import BlockKind, ResNet
from hyper_lap.modules.vit import ViT


class ViTEmbedder(eqx.Module):
    vit: ViT

    def __init__(self, emb_size: int, key: PRNGKeyArray):
        super().__init__()

        self.vit = ViT(512, 3, 16, 6, 8, 64, emb_size, key=key)

    def __call__(
        self,
        image: Float[Array, "3 h w"],
        label: Integer[Array, "h w"],
        _dataset_idx: Integer[Array, ""],
    ) -> Float[Array, " emb_size"]:
        c, h, w = image.shape

        assert c == 3
        assert_shape(label, [h, w])

        label = jnp.expand_dims((label != 0).astype(image.dtype), 0)

        combined = jnp.concatenate([image, label], axis=0)

        assert_shape(combined, (4, h, w))

        emb = self.vit(combined)

        return emb
