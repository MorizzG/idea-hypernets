from jaxtyping import Array, Float, Integer, PRNGKeyArray
from typing import Literal

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import transformers
from chex import assert_shape
from transformers.models.clip import FlaxCLIPVisionModel

from hyper_lap.modules.convnext import ConvNeXt
from hyper_lap.modules.resnet import BlockKind, ResNet
from hyper_lap.modules.vit import ViT


class LearnedEmbedding(eqx.Module):
    emb_size: int = eqx.field(static=True)

    embedding: Array

    def __init__(self, emb_size: int, *, key: PRNGKeyArray):
        super().__init__()

        self.emb_size = emb_size

        self.embedding = jr.normal(key, (emb_size,))

    def __call__(self, image: Float[Array, "3 h w"], label: Integer[Array, "h w"]) -> Array:
        return self.embedding


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
        self, image: Float[Array, "3 h w"], label: Integer[Array, "h w"]
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


class ConvNextEmbedder(eqx.Module):
    convnext: ConvNeXt

    def __init__(self, emb_size: int, key: PRNGKeyArray):
        super().__init__()

        self.convnext = ConvNeXt(emb_size, 96, in_channels=3, depths=[3, 3, 9, 3], key=key)

    def __call__(
        self, image: Float[Array, "3 h w"], label: Integer[Array, "h w"]
    ) -> Float[Array, " emb_size"]:
        c, h, w = image.shape

        assert c == 3
        assert_shape(label, [h, w])

        image = image.mean(axis=0)

        pos_masked_image = (label != 0) * image
        neg_masked_image = (label == 0) * image

        input = jnp.stack([image, pos_masked_image, neg_masked_image])

        assert_shape(input, (3, h, w))

        emb = self.convnext(input)

        return emb


class ViTEmbedder(eqx.Module):
    vit: ViT

    def __init__(self, emb_size: int, key: PRNGKeyArray):
        super().__init__()

        self.vit = ViT(512, 3, 16, 6, 8, 64, emb_size, key=key)

    def __call__(
        self, image: Float[Array, "3 h w"], label: Integer[Array, "h w"]
    ) -> Float[Array, " emb_size"]:
        c, h, w = image.shape

        assert c == 3
        assert_shape(label, [h, w])

        label = jnp.expand_dims((label != 0).astype(image.dtype), 0)

        combined = jnp.concatenate([image, label], axis=0)

        assert_shape(combined, (4, h, w))

        emb = self.vit(combined)

        return emb


class ClipEmbedder(eqx.Module):
    num_layers: int = eqx.field(static=True)
    select_layer: int = eqx.field(static=True)
    pool: Literal["cls", "mean"] = eqx.field(static=True)

    clip_vision: FlaxCLIPVisionModel

    # tokenizer: CLIPTokenizerFast = eqx.field(static=True)
    # clip_text: FlaxCLIPTextModel = eqx.field(static=True)

    projection: nn.Linear | nn.Identity

    def __init__(
        self,
        emb_size: int,
        select_layer: int = -2,
        pool: Literal["cls", "mean"] = "cls",
        clip: Literal["openai"] = "openai",
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if pool not in ["cls", "mean"]:
            raise ValueError(f"Invalid pool {pool}")

        self.select_layer = select_layer
        self.pool = pool

        # logging.getLogger("transformers/modeling_flax_utils.py").setLevel(logging.ERROR)

        orig_verbosity = transformers.logging.get_verbosity()  # type: ignore

        transformers.logging.set_verbosity_error()  # type: ignore

        if clip == "openai":
            clip_vision = FlaxCLIPVisionModel.from_pretrained(
                "openai/clip-vit-large-patch14-336", from_pt=True
            )
            assert isinstance(clip_vision, FlaxCLIPVisionModel)
            self.clip_vision = clip_vision

            # tokenizer = CLIPTokenizerFast.from_pretrained(
            #     "openai/clip-vit-large-patch14-336", from_pt=True
            # )
            # assert isinstance(tokenizer, CLIPTokenizerFast), (
            #     f"expected CLIPTokenizerFast, found {type(tokenizer)}"
            # )
            # self.tokenizer = tokenizer

            # clip_text = FlaxCLIPTextModel.from_pretrained(
            #     "openai/clip-vit-large-patch14-336", from_pt=True
            # )
            # assert isinstance(clip_text, FlaxCLIPTextModel)
            # self.clip_text = clip_text
        else:
            raise RuntimeError(f"Unknown clip variant {clip}")

        transformers.logging.set_verbosity(orig_verbosity)  # type: ignore

        self.num_layers = len(self.clip_vision.params["vision_model"]["encoder"]["layers"])
        # self.num_layers = 24

        if not -self.num_layers <= select_layer < self.num_layers:
            raise ValueError(
                f"select_layer {select_layer} is out of range for {self.num_layers} layers"
            )

        self.projection = nn.Linear(3 * 1024, emb_size, key=key)
        # self.projection = nn.Identity(key=key)

    def __call__(
        self, image: Float[Array, "3 h w"], label: Integer[Array, "h w"]
    ) -> Float[Array, " emb_size"]:
        c, h, w = image.shape

        assert c == 3
        assert_shape(label, [h, w])

        pos_masked_image = (label != 0) * image
        neg_masked_image = (label == 0) * image

        input = jnp.stack([image, pos_masked_image, neg_masked_image])

        assert input.shape[:2] == (3, 3)

        if input.shape[-2:] != (336, 336):
            input = jax.image.resize(input, (3, 3, 336, 336), method="bicubic")

        assert_shape(input, (3, 3, 336, 336))

        output = self.clip_vision(input, output_hidden_states=True)

        hidden_states: list[Array] = output["hidden_states"][1:]  # type: ignore

        assert len(hidden_states) == self.num_layers

        hidden_state = hidden_states[self.select_layer]

        assert_shape([hidden_state], (3, 577, 1024))

        if self.pool == "cls":
            vision_emb = hidden_state[:, 0, :].ravel()
        elif self.pool == "mean":
            vision_emb = hidden_state.mean(axis=1).ravel()
        else:
            assert False

        assert_shape(vision_emb, (3 * 1024,))

        vision_emb = jax.lax.stop_gradient(vision_emb)

        # inputs = self.tokenizer(text=text, return_tensors="jax")

        # text_emb = self.clip_text(input_ids=inputs["input_ids"]).pooler_output[0]  # type: ignore

        emb = self.projection(vision_emb)

        return emb
