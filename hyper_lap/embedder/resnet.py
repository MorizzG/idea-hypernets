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


class ResNetEmbedder(eqx.Module):
    resnet: ResNet

    def __init__(
        self,
        emb_size: int,
        depths: list[int] | None = None,
        block_kind: BlockKind = "bottleneck",
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        self.resnet = ResNet(emb_size, in_channels=3, depths=depths, block_kind=block_kind, key=key)

    def __call__(
        self,
        image: Float[Array, "3 h w"],
        label: Integer[Array, "h w"],
        _dataset_idx: Integer[Array, ""],
    ) -> Float[Array, " emb_size"]:
        c, h, w = image.shape

        assert c == 3
        assert_shape(label, [h, w])

        image = image.mean(axis=0)

        pos_masked_image = (label != 0) * image
        neg_masked_image = (label == 0) * image

        input = jnp.stack([image, pos_masked_image, neg_masked_image])

        assert_shape(input, (3, h, w))

        emb = self.resnet(input)

        return emb
