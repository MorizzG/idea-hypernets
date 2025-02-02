from jaxtyping import Array, Float, Integer, PRNGKeyArray
from typing import Literal

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import transformers
from chex import assert_shape
from jax import lax
from transformers.models.clip import FlaxCLIPVisionModel

from hyper_lap.modules.convnext import ConvNeXt
from hyper_lap.modules.resnet import ResNet
from hyper_lap.modules.vit import ViT


class LearnedEmbedding(eqx.Module):
    embedding: Array

    def __init__(self, emb_size: int, *, key: PRNGKeyArray):
        super().__init__()

        self.embedding = jr.normal(key, (emb_size,))

    def __call__(self, image: Float[Array, "3 h w"], label: Integer[Array, "h w"]) -> Array:
        return self.embedding


class ResNetEmbedder(eqx.Module):
    resnet: ResNet

    def __init__(self, emb_size: int, key: PRNGKeyArray):
        super().__init__()

        self.resnet = ResNet(
            emb_size, in_channels=4, depths=[2, 2, 2, 2], block_kind="bottleneck", key=key
        )

    def __call__(
        self, image: Float[Array, "3 h w"], label: Integer[Array, "h w"]
    ) -> Float[Array, " emb_size"]:
        c, h, w = image.shape

        assert c == 3
        assert_shape(label, [h, w])

        label = jnp.expand_dims((label != 0).astype(image.dtype), 0)

        combined = jnp.concatenate([image, label], axis=0)

        assert_shape(combined, (4, h, w))

        emb = self.resnet(combined)

        return emb


class ConvNextEmbedder(eqx.Module):
    convnext: ConvNeXt

    def __init__(self, emb_size: int, key: PRNGKeyArray):
        super().__init__()

        self.convnext = ConvNeXt(emb_size, 96, in_channels=4, depths=[3, 3, 9, 3], key=key)

    def __call__(
        self, image: Float[Array, "3 h w"], label: Integer[Array, "h w"]
    ) -> Float[Array, " emb_size"]:
        c, h, w = image.shape

        assert c == 3
        assert_shape(label, [h, w])

        label = jnp.expand_dims((label != 0).astype(image.dtype), 0)

        combined = jnp.concatenate([image, label], axis=0)

        assert_shape(combined, (4, h, w))

        emb = self.convnext(combined)

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

    # image_embedder: FlaxCLIPVisionModel = eqx.field(static=True)
    # label_embedder: FlaxCLIPVisionModel = eqx.field(static=True)

    clip: FlaxCLIPVisionModel = eqx.field(static=True)

    projection: nn.Linear

    def __init__(
        self,
        emb_size: int,
        select_layer: int = -2,
        pool: Literal["cls", "mean"] = "cls",
        *,
        key: PRNGKeyArray,
    ):
        super().__init__()

        if pool not in ["cls", "mean"]:
            raise ValueError(f"Invalid pool {pool}")

        self.select_layer = select_layer
        self.pool = pool

        # logging.getLogger("transformers/modeling_flax_utils.py").setLevel(logging.ERROR)

        orig_verbosity = transformers.logging.get_verbosity()

        transformers.logging.set_verbosity_error()

        self.clip = FlaxCLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14-336", from_pt=True
        )  # type: ignore

        transformers.logging.set_verbosity(orig_verbosity)

        self.num_layers = len(self.clip.params["vision_model"]["encoder"]["layers"])
        # self.num_layers = 24

        if not -self.num_layers <= select_layer < self.num_layers:
            raise ValueError(
                f"select_layer {select_layer} is out of range for {self.num_layers} layers"
            )

        self.projection = nn.Linear(2 * 1024, emb_size, key=key)

    def __call__(
        self, image: Float[Array, "3 h w"], label: Integer[Array, "h w"]
    ) -> Float[Array, " emb_size"]:
        c, h, w = image.shape

        assert c == 3
        assert_shape(label, [h, w])

        # label = jnp.expand_dims(label, 0).astype(image.dtype)
        # label = image * jnp.expand_dims(label, 0).astype(image.dtype)

        # duplicate image along new axis
        image_double = jnp.repeat(jnp.expand_dims(image, 0), 2, 0)

        # multiply second copy with label -> hadamard image
        image_double = image_double.at[1, ...].multiply(
            jnp.expand_dims(label, 0).astype(image.dtype)
        )

        # image = jax.image.resize(image, (3, 336, 336), method="bilinear")
        # label = jax.image.resize(label, (3, 336, 336), method="bilinear")
        image_double = jax.image.resize(image_double, (2, 3, 336, 336), method="bilinear")

        # image_output = self.clip(image[None, ...], output_hidden_states=True)
        # label_output = self.clip(label[None, ...], output_hidden_states=True)

        # ignore first hidden state, as it's just the input
        # image_hidden_states = image_output["hidden_states"][1:]  # type: ignore
        # label_hidden_states = label_output["hidden_states"][1:]  # type: ignore

        # assert (
        #     len(image_hidden_states) == self.num_layers
        #     and len(label_hidden_states) == self.num_layers
        # )

        # image_hidden_state = image_hidden_states[self.select_layer]
        # label_hidden_state = label_hidden_states[self.select_layer]

        output = self.clip(image_double, output_hidden_states=True)

        hidden_states: list[Array] = output["hidden_states"][1:]  # type: ignore

        assert len(hidden_states) == self.num_layers

        hidden_state = hidden_states[self.select_layer]

        assert_shape([hidden_state], (2, 577, 1024))

        # if self.pool == "cls":
        #     image_emb = image_hidden_state[0, 0, :]
        #     label_emb = label_hidden_state[0, 0, :]
        # elif self.pool == "mean":
        #     image_emb = image_hidden_state[0].mean(axis=0)
        #     label_emb = label_hidden_state[0].mean(axis=0)
        # else:
        #     assert False, f"self.pool has unexpected value {self.pool}"

        # assert_shape([image_emb, label_emb], (1024,))

        # output_emb = jnp.concat([image_emb, label_emb])

        if self.pool == "cls":
            output_emb = hidden_state[:, 0, :].ravel()
        elif self.pool == "mean":
            output_emb = hidden_state.mean(axis=1).ravel()
        else:
            assert False

        assert_shape(output_emb, (2 * 1024,))

        emb = self.projection(output_emb)

        return lax.stop_gradient(emb)
